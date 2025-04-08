"""Objects to define a DAG which indexes, deduplicates and links data."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Union

from pandas import DataFrame
from pydantic import BaseModel, model_validator

from matchbox.client import _handler
from matchbox.client.helpers.cleaner import process
from matchbox.client.helpers.selector import Selector, query
from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.models.models import make_model
from matchbox.common.sources import Source

DAGNode = Union["IndexStep", "ModelStep"]
"""Type of node in the DAG. Either an indexing or model step."""


class IndexStep(BaseModel):
    """Index step."""

    source: Source
    batch_size: int | None = None

    @property
    def name(self) -> str:
        """Resolution name for this source."""
        return str(self.source.address)

    @property
    def sources(self) -> set[str]:
        """Return all sources to this step."""
        return {self.name}

    @property
    def inputs(self) -> list["StepInput"]:
        """Return all inputs to this step."""
        return []

    def run(self):
        """Run indexing step."""
        _handler.index(source=self.source, batch_size=self.batch_size)


class StepInput(BaseModel):
    """Input to a DAG step, generated by a previous node in the DAG."""

    prev_node: DAGNode
    select: dict[Source, list[str]]
    cleaners: dict[str, dict[str, Any]] = {}
    batch_size: int | None = None
    threshold: float | None = None

    @property
    def name(self):
        """Resolution name for node generating this input for the next step."""
        if isinstance(self.prev_node, Source):
            return self.prev_node.resolution_name
        else:
            return self.prev_node.name

    @model_validator(mode="after")
    def validate_all_input(self) -> "StepInput":
        """Verify select statement is valid given previous node."""
        if isinstance(self.prev_node, IndexStep):
            if (
                len(self.select) > 1
                or list(self.select.keys())[0] != self.prev_node.source
            ):
                raise ValueError(
                    f"Can only select from source {self.prev_node.source.address}"
                )
        else:
            for source in self.select:
                if str(source.address) not in self.prev_node.sources:
                    raise ValueError(
                        f"Cannot select {source.address} from {self.prev_node.name}."
                        f"Available sources are {self.prev_node.sources}."
                    )
        return self


class ModelStep(BaseModel, ABC):
    """Base step in DAG."""

    name: str
    description: str
    left: StepInput
    settings: dict[str, Any]
    truth: float
    sources: set[str] = set()

    @property
    @abstractmethod
    def inputs(self) -> list[StepInput]:
        """Return all inputs to this step."""
        raise NotImplementedError

    @model_validator(mode="after")
    def init_sources(self) -> "ModelStep":
        """Add sources inherited from all inputs."""
        for step_input in self.inputs:
            if isinstance(step_input.prev_node, Source):
                self.sources.add(str(step_input.prev_node.address))
            else:
                self.sources.update(step_input.prev_node.sources)

        return self

    def query(self, step_input: StepInput) -> DataFrame:
        """Retrieve data for declared step input.

        Args:
            step_input: Declared input to this DAG step.

        Returns:
            Pandas dataframe with retrieved results.
        """
        selectors = [
            Selector(engine=s.engine, address=s.address, fields=f)
            for s, f in step_input.select.items()
        ]
        df_raw = query(
            selectors,
            return_type="pandas",
            threshold=step_input.threshold,
            resolution_name=step_input.name,
            only_indexed=True,
            batch_size=step_input.batch_size,
            return_batches=False,
        )

        return df_raw


class DedupeStep(ModelStep):
    """Deduplication step."""

    model_class: type[Deduper]

    @property
    def inputs(self) -> list[StepInput]:
        """Return all inputs to this step."""
        return [self.left]

    def run(self):
        """Run full deduping pipeline and store results."""
        df_raw = self.query(self.left)
        df_clean = process(df_raw, self.left.cleaners)
        deduper = make_model(
            model_name=self.name,
            description=self.description,
            model_class=self.model_class,
            model_settings=self.settings,
            left_data=df_clean,
            left_resolution=self.left.name,
        )
        results = deduper.run()
        results.to_matchbox()
        deduper.truth = self.truth


class LinkStep(ModelStep):
    """Linking step."""

    model_class: type[Linker]
    right: "StepInput"

    @property
    def inputs(self) -> list[StepInput]:
        """Return all `StepInputs` to this step."""
        return [self.left, self.right]

    def run(self):
        """Run whole linking step."""
        left_raw = self.query(self.left)
        left_clean = process(left_raw, self.left.cleaners)

        right_raw = self.query(self.right)
        right_clean = process(right_raw, self.right.cleaners)

        linker = linker = make_model(
            model_name=self.name,
            description=self.description,
            model_class=self.model_class,
            model_settings=self.settings,
            left_data=left_clean,
            left_resolution=self.left.name,
            right_data=right_clean,
            right_resolution=self.right.name,
        )
        res = linker.run()
        res.to_matchbox()
        linker.truth = self.truth


class DAG:
    """Self-sufficient pipeline of indexing, deduping and linking steps."""

    def __init__(self):
        """Initialise DAG object."""
        self.nodes: dict[str, DAGNode] = {}
        self.graph: dict[str, list[str]] = {}
        self.sequence: list[str] = []

    def _validate_node(self, name: str) -> None:
        """Validate that a node name is unique in the DAG."""
        if name in self.nodes:
            raise ValueError(f"Name '{name}' is already taken in the DAG")

    def _validate_inputs(self, step: ModelStep) -> None:
        """Validate that all inputs to a step are already in the DAG."""
        for step_input in step.inputs:
            if step_input.name not in self.nodes:
                raise ValueError(f"Dependency {step_input.name} not added to DAG")

    def add_sources(
        self, *sources: Source, batch_size: int | None = None
    ) -> tuple[IndexStep]:
        """Add sources to DAG.

        Args:
            sources: All sources to add.
            batch_size: Batch size for indexing.
        """
        index_steps = tuple(
            IndexStep(source=source, batch_size=batch_size) for source in sources
        )
        self.add_steps(*index_steps)
        return index_steps

    def add_steps(self, *steps: ModelStep) -> None:
        """Add dedupers and linkers to DAG, and register sources available to steps.

        Args:
            steps: Dedupe and link steps.
        """
        for step in steps:
            self._validate_node(step.name)
            self._validate_inputs(step)
            self.nodes[step.name] = step
            self.graph[step.name] = [step_input.name for step_input in step.inputs]

    def prepare(self) -> None:
        """Determine order of execution of steps."""
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

    def run(self, start: str | None = None):
        """Run entire DAG."""
        self.prepare()

        start_index = 0
        if start:
            try:
                start_index = self.sequence.index(start)
            except ValueError as e:
                raise ValueError(f"Step {start} not in DAG") from e

        for step_name in self.sequence[start_index:]:
            node = self.nodes[step_name]
            node.run()
