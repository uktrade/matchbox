"""Objects to define a DAG which indexes, deduplicates and links data."""

import datetime
from collections import defaultdict
from typing import Self

from pyarrow import Table as ArrowTable

from matchbox.client import _handler
from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.sources import Source
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.logging import logger


class DAG:
    """Self-sufficient pipeline of indexing, deduping and linking steps."""

    def __init__(self, name: str, new: bool = False):
        """Initialises empty DAG."""
        self.name = name
        self.nodes: dict[str, Source | Model] = {}
        self.graph: dict[str, list[str]] = {}

    def _check_dag(self, dag: Self):
        if self != dag:
            raise ValueError("Cannot mix DAGs")

    def _add_step(self, step: Source | Model) -> None:
        """Validate and add sources and models to DAG."""
        self._check_dag(step.dag)

        if step.name in self.nodes:
            raise ValueError(f"Name '{step.name}' is already taken in the DAG")
        if isinstance(step, Model):
            self._check_dag(step.left_query.dag)
            if step.right_query:
                self._check_dag(step.right_query.dag)

            for resolution_name in step.dependencies:
                if resolution_name not in self.nodes:
                    raise ValueError(f"Step {resolution_name} not added to DAG")
            self.graph[step.name] = step.parents
        else:
            self.graph[step.name] = []

        self.nodes[step.name] = step

    @property
    def final_step(self) -> str:
        """Returns name of the root node in the DAG.

        Returns:
            The name of the root node in the DAG

        Raises:
            ValueError: If the DAG does not have a final step
        """
        if not self.nodes:
            raise ValueError("The DAG is empty.")

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
            return apex_nodes.pop()

    def source(self, *args, **kwargs) -> Source:
        """Create Source and add it to the DAG."""
        source = Source(*args, **kwargs, dag=self)
        self._add_step(source)

        return source

    def model(self, *args, **kwargs) -> Model:
        """Create Model and add it to the DAG."""
        model = Model(*args, **kwargs, dag=self)
        self._add_step(model)

        return model

    def query(self, *args, **kwargs) -> Query:
        """Create Query object."""
        return Query(*args, **kwargs, dag=self)

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
        root_name = self.final_step
        skipped = skipped or []

        def _get_node_status(name: str) -> str:
            """Determine the status indicator for a node."""
            if name in skipped:
                return "⏭️"
            elif doing and name == doing:
                return "🔄"
            elif (
                (node := self.nodes.get(name))
                and node.last_run
                and node.last_run > start_time
            ):
                return "✅"
            else:
                return "⏸️"

        # Add status indicator if start_time is provided
        if start_time is not None:
            status = _get_node_status(root_name)
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
                    status = _get_node_status(child)
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

    def run_and_sync(
        self,
        full_rerun: bool = False,
        start: str | None = None,
        finish: str | None = None,
    ):
        """Run entire DAG and send results to server."""
        start_time = datetime.datetime.now()

        # Determine order of execution steps
        root_node = self.final_step

        def depth_first(node: str, sequence: list):
            sequence.append(node)
            for neighbour in self.graph[node]:
                if neighbour not in sequence:
                    depth_first(neighbour, sequence)

        inverse_sequence = []
        depth_first(root_node, inverse_sequence)
        sequence = list(reversed(inverse_sequence))

        # Identify skipped nodes
        skipped_nodes = []
        if start:
            try:
                start_index = sequence.index(start)
                skipped_nodes = sequence[:start_index]
            except ValueError as e:
                raise ValueError(f"Step {start} not in DAG") from e
        else:
            start_index = 0

        # Determine end index
        if finish:
            try:
                end_index = sequence.index(finish) + 1
                skipped_nodes.extend(sequence[end_index:])
            except ValueError as e:
                raise ValueError(f"Step {finish} not in DAG") from e
        else:
            end_index = len(sequence)

        sequence = sequence[start_index:end_index]
        if not full_rerun:
            # Exclude nodes that already run, unless a full re-run is forced
            already_run = [node for node in sequence if self.nodes[node].last_run]
            skipped_nodes.extend(already_run)
            sequence = [node for node in sequence if node not in already_run]

        for step_name in sequence:
            node = self.nodes[step_name]

            logger.info(
                "\n"
                + self.draw(
                    start_time=start_time, doing=node.name, skipped=skipped_nodes
                )
            )
            try:
                node.run()
                node.sync()
            except Exception as e:
                logger.error(f"❌ {node.name} failed: {e}")
                raise e

        logger.info("\n" + self.draw(start_time=start_time, skipped=skipped_nodes))

    def lookup_key(
        self,
        from_source: str,
        to_sources: list[str],
        key: str,
        threshold: int | None = None,
    ) -> dict[str, list[str]]:
        """Matches IDs against the selected backend.

        Args:
            from_source: Name of source the provided key belongs to
            to_sources: Names of sources to find keys in
            key: The value to match from the source. Usually a primary key
            threshold (optional): The threshold to use for creating clusters.
                If None, uses the resolutions' default threshold
                If an integer, uses that threshold for the specified resolution, and the
                resolution's cached thresholds for its ancestors

        Examples:
            ```python
            dag.lookup_key(
                from_source="companies_house",
                to_sources=[
                    "datahub_companies",
                    "hmrc_exporters",
                ]
                key="8534735",
            )
            ```
        """
        matches = _handler.match(
            targets=to_sources,
            source=from_source,
            key=key,
            resolution=self.final_step,
            threshold=threshold,
        )

        to_sources_results = {m.target: list(m.target_id) for m in matches}
        # If no matches, _handler will raise
        return {from_source: list(matches[0].source_id), **to_sources_results}

    def extract_lookup(
        self,
        source_filter: list[str] | None = None,
        location_names: list[str] | None = None,
    ) -> ArrowTable:
        """Return matchbox IDs to source key mapping, optionally filtering.

        Args:
            source_filter: An optional list of source resolution names to filter by.
            location_names: An optional list of location names to filter by.
        """
        # Get all sources in scope of the resolution
        point_of_truth = self.final_step
        source_resolutions = _handler.get_leaf_source_resolutions(name=point_of_truth)

        if source_filter:
            source_resolutions = [
                s for s in source_resolutions if s.name in source_filter
            ]

        if location_names:
            source_resolutions = [
                s
                for s in source_resolutions
                if s.config.location_config.name in location_names
            ]

        if not source_resolutions:
            raise MatchboxResolutionNotFoundError("No compatible source was found")

        source_mb_ids: list[ArrowTable] = []
        source_to_key_field: dict[str, str] = {}

        for s in source_resolutions:
            # Get Matchbox IDs from backend
            source_mb_ids.append(
                _handler.query(
                    source=s.name, resolution=point_of_truth, return_leaf_id=False
                )
            )

            source_to_key_field[s.name] = s.config.key_field.name

        # Join Matchbox IDs to form mapping table
        mapping = source_mb_ids[0]
        mapping = mapping.rename_columns(
            {
                "key": source_resolutions[0].config.qualified_key(
                    source_resolutions[0].name
                )
            }
        )
        if len(source_resolutions) > 1:
            for s, mb_ids in zip(
                source_resolutions[1:], source_mb_ids[1:], strict=True
            ):
                mapping = mapping.join(
                    right_table=mb_ids, keys="id", join_type="full outer"
                )
                mapping = mapping.rename_columns(
                    {"key": s.config.qualified_key(s.name)}
                )

        return mapping
