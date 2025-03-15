from abc import ABC
from collections import defaultdict
from typing import Any, Union

from pandas import DataFrame
from pydantic import BaseModel
from sqlalchemy import Engine

from matchbox.client import _handler
from matchbox.client.helpers.cleaner import process
from matchbox.client.helpers.selector import query, select
from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.models.models import make_model
from matchbox.client.results import Results
from matchbox.common.logging import logger
from matchbox.common.sources import Source

DagNode = Union["ModelStep", Source]


class StepInput(BaseModel):
    """Input to a DAG step."""

    origin: DagNode
    select: dict[str, list[str]]
    cleaners: dict[str, dict[str, Any]] = {}
    threshold: float | None = None

    @property
    def name(self):
        if isinstance(self.origin, ModelStep):
            return self.origin.name
        else:
            return self.origin.address.full_name


class ModelStep(BaseModel, ABC):
    """Base step in DAG."""

    name: str
    description: str
    left: StepInput
    settings: dict[str, Any]
    sources: set[str] = set()

    def query(self, step_input: StepInput, engine: Engine) -> DataFrame:
        """Retrieve data for declared step input.

        Args:
            step_input: Declared input to this DAG step.
            engine: SQLAlchemy engine to use for retrieving the data.

        Returns:
            Pandas dataframe with retrieved results.
        """
        selector = select(
            step_input.select,
            engine=engine,
        )

        df_raw = query(
            selector,
            return_type="pandas",
            threshold=step_input.threshold,
            resolution_name=step_input.name,
        )

        return df_raw


class DedupeStep(ModelStep):
    """Deduplication step."""

    model_class: type[Deduper]

    def deduplicate(self, df: DataFrame) -> Results:
        """Define and run deduper on pre-processed data.

        Args:
            df_clean: Pandas dataframe with cleaned data.

        Returns:
            Results from running the model.
        """
        model = make_model(
            model_name=self.name,
            description=self.description,
            model_class=self.model_class,
            model_settings=self.settings,
            left_data=df,
            left_resolution=self.left.name,
        )

        res = model.run()
        return res

    def run(self, engine: Engine):
        """Run full deduping pipeline and store results.

        Args:
            engine: SQLAlchemy Engine to use when retrieving data.
        """

        df_raw = self.query(self.left, engine)
        df_clean = process(df_raw, self.left.cleaners)
        results = self.deduplicate(df_clean)
        results.to_matchbox()


class LinkStep(ModelStep):
    """Linking step."""

    model_class: type[Linker]
    right: "StepInput"

    def link(
        self,
        left_df: DataFrame,
        right_df: DataFrame,
    ):
        linker = make_model(
            model_name=self.name,
            description=self.description,
            model_class=self.model_class,
            model_settings=self.settings,
            left_data=left_df,
            left_resolution=self.left.name,
            right_data=right_df,
            right_resolution=self.right.name,
        )

        return linker.run()

    def run(self, engine: Engine):
        left_raw = self.query(self.left, engine)
        left_clean = process(left_raw, self.left.cleaners)

        right_raw = self.query(self.right, engine)
        right_clean = process(right_raw, self.right.cleaners)

        res = self.link(left_clean, right_clean)
        res.to_matchbox()


class Dag:
    """Self-sufficient pipeline of indexing, deduping and linking steps."""

    def __init__(self, engine: Engine):
        self.engine = engine

        self.nodes: dict[str, DagNode] = {}
        self.graph: dict[str, list[str]] = {}
        self.sequence: list[str] = []

    def _validate_node(self, name: str, node: DagNode):
        if name in self.nodes:
            raise ValueError(f"Name '{name}' is already taken in the DAG")

    def add_sources(self, *sources: Source):
        """Add sources to DAG.

        Args:
            sources: All sources to add.
        """
        for source in sources:
            self._validate_node(source.address.full_name, source)
            self.nodes[source.address.full_name] = source
            self.graph[source.address.full_name] = []

    def add_steps(self, *steps: ModelStep):
        """Add dedupers and linkers to DAG, and register sources available to steps.

        Args:
            steps: Dedupe and link steps.
        """

        def validate_input(step: ModelStep, step_input: StepInput):
            """Validate and update available sources for step input"""
            if step_input.name not in self.nodes:
                raise ValueError(f"Dependency {step_input.name} not available")

            origin = step_input.origin
            # Before adding sources, validate select statements
            if isinstance(origin, Source):
                if (
                    len(step_input.select) > 1
                    or list(step_input.select.keys())[0] != origin.address.full_name
                ):
                    raise ValueError(
                        f"Can only select from source {origin.address.full_name}"
                    )
                step.sources.add(step_input.origin.address.full_name)
            else:
                for source_name in step_input.select:
                    if source_name not in origin.sources:
                        raise ValueError(
                            f"Cannot select {source_name} from {step_input.name}"
                        )
                step.sources.update(origin.sources)

        for step in steps:
            self._validate_node(step.name, step)

            try:
                validate_input(step, step.left)

                if isinstance(step, LinkStep):
                    validate_input(step, step.right)
            except ValueError as e:
                # If the validation fails, reset this step's sources
                step.sources = set()
                raise e

            # Only add to DAG after everything is validated
            self.nodes[step.name] = step
            self.graph[step.name] = [step.left.name]
            if isinstance(step, LinkStep):
                self.graph[step.name].append(step.right.name)

    def prepare(self):
        """Determine order of execution of steps"""
        self.sequence = []

        inverse_graph = defaultdict(list)
        for node in self.graph:
            for neighbor in self.graph[node]:
                inverse_graph[neighbor].append(node)
        apex = {node for node in self.graph if node not in inverse_graph}
        if len(apex) > 1:
            raise ValueError("Some models or sources are disconnected")
        else:
            apex = apex.pop()

        def depth_first(node: str, sequence: list):
            sequence.append(node)
            for neighbour in self.graph[node]:
                if neighbour not in sequence:
                    depth_first(neighbour, sequence)

        inverse_sequence = []
        depth_first(apex, inverse_sequence)
        self.sequence = list(reversed(inverse_sequence))

    def run(self):
        """Run entire DAG."""
        self.prepare()

        for step_name in self.sequence:
            node = self.nodes[step_name]
            if isinstance(node, Source):
                _handler.index(source=node)
                logger.info(f"Indexed {node.address.full_name}")
            else:
                node.run(engine=self.engine)
                logger.info(f"Run {node.name} model pipeline")
