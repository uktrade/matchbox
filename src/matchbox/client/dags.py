"""Objects to define a DAG which indexes, deduplicates and links data."""

import datetime
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
from matchbox.common.logging import logger
from matchbox.common.sources import Source

DAGNode = Union["IndexStep", "ModelStep"]
"""Type of node in the DAG. Either an indexing or model step."""


class IndexStep(BaseModel):
    """Index step."""

    source: Source
    batch_size: int | None = None
    last_run: datetime.datetime | None = None

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
    last_run: datetime.datetime | None = None

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

    def _build_inverse_graph(self) -> tuple[dict[str, list[str]], str]:
        """Build inverse graph and find the apex node.

        Returns:
            tuple: (inverse_graph, apex_node)

                * inverse_graph: Dictionary mapping nodes to their parent nodes
                * apex_node: The root node of the DAG

        Raises:
            ValueError: If the DAG has multiple disconnected components
        """
        inverse_graph = defaultdict(list)
        for node in self.graph:
            for neighbor in self.graph[node]:
                inverse_graph[neighbor].append(node)

        apex_nodes = {node for node in self.graph if node not in inverse_graph}
        if len(apex_nodes) > 1:
            raise ValueError("Some models or sources are disconnected")
        elif not apex_nodes:
            raise ValueError("No root node found, DAG might contain cycles")
        else:
            return inverse_graph, apex_nodes.pop()

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

        _, apex = self._build_inverse_graph()

        def depth_first(node: str, sequence: list):
            sequence.append(node)
            for neighbour in self.graph[node]:
                if neighbour not in sequence:
                    depth_first(neighbour, sequence)

        inverse_sequence = []
        depth_first(apex, inverse_sequence)
        self.sequence = list(reversed(inverse_sequence))

    def draw(
        self,
        start_time: datetime.datetime | None = None,
        doing: str | None = None,
        skipped: list[str] | None = None,
    ) -> str:
        """Create a string representation of the DAG as a tree structure.

        If `start_time` is provided, it will show the status of each node
        based on the last run time. The status indicators are:

        * ✅ Done
        * 🔄 Working
        * ⏸️ Awaiting
        * ⏭️ Skipped

        Args:
            start_time: Start time of the DAG run. Used to calculate node status.
            doing: Name of the node currently being processed (if any).
            skipped: List of node names that were skipped.

        Returns:
            String representation of the DAG with status indicators.
        """
        _, root_name = self._build_inverse_graph()
        skipped = skipped or []

        # Add status indicator if start_time is provided
        if start_time is not None:
            node = self.nodes.get(root_name)

            if root_name in skipped:
                status = "⏭️"
            elif doing and root_name == doing:
                status = "🔄"
            elif node and node.last_run and node.last_run > start_time:
                status = "✅"
            else:
                status = "⏸️"

            result = [f"{status} {root_name}"]
        else:
            result = [root_name]

        visited = set([root_name])

        def format_children(node: str, prefix=""):
            """Recursively format the children of a node."""
            children = []
            # Get all outgoing edges from this node
            for target in self.graph.get(node, []):
                if target not in visited:
                    children.append(target)
                    visited.add(target)

            # Format each child
            for i, child in enumerate(children):
                is_last = i == len(children) - 1

                # Add status indicator if start_time is provided
                if start_time is not None:
                    child_node = self.nodes.get(child)

                    if child in skipped:
                        status = "⏭️"
                    elif doing and child == doing:
                        status = "🔄"
                    elif (
                        child_node
                        and child_node.last_run
                        and child_node.last_run > start_time
                    ):
                        status = "✅"
                    else:
                        status = "⏸️"

                    child_display = f"{status} {child}"
                else:
                    child_display = child

                if is_last:
                    result.append(f"{prefix}└── {child_display}")
                    format_children(child, prefix + "    ")
                else:
                    result.append(f"{prefix}├── {child_display}")
                    format_children(child, prefix + "│   ")

        format_children(root_name)

        return "\n".join(result)

    def run(self, start: str | None = None):
        """Run entire DAG.

        Args:
            start: Name of the step to start from (if not from the beginning)
        """
        self.prepare()

        start_time = datetime.datetime.now()

        # Identify skipped nodes
        skipped_nodes = []
        if start:
            try:
                start_index = self.sequence.index(start)
                skipped_nodes = self.sequence[:start_index]
            except ValueError as e:
                raise ValueError(f"Step {start} not in DAG") from e
        else:
            start_index = 0

        logger.info("\n" + self.draw(start_time=start_time, skipped=skipped_nodes))

        for step_name in self.sequence[start_index:]:
            node = self.nodes[step_name]

            try:
                logger.info(
                    "\n"
                    + self.draw(
                        start_time=start_time, doing=node.name, skipped=skipped_nodes
                    )
                )
                node.run()
                node.last_run = datetime.datetime.now()
            except Exception as e:
                logger.error(f"❌ {node.name} failed: {e}")
                raise e

        logger.info("\n" + self.draw(start_time=start_time, skipped=skipped_nodes))
