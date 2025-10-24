"""Objects to define a DAG which indexes, deduplicates and links data."""

import datetime
import json
from collections import defaultdict
from typing import Any, Self

from pyarrow import Table as ArrowTable

from matchbox.client import _handler
from matchbox.client.locations import Location
from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.sources import Source
from matchbox.common.dtos import (
    Collection,
    CollectionName,
    ModelResolutionName,
    Resolution,
    ResolutionName,
    ResolutionPath,
    ResolutionType,
    Run,
    RunID,
    SourceResolutionName,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.common.transform import truth_int_to_float


class DAG:
    """Self-sufficient pipeline of indexing, deduping and linking steps."""

    def __init__(self, name: str):
        """Initialises empty DAG."""
        self.name: CollectionName = CollectionName(name)
        self._run: RunID | None = None
        self.nodes: dict[ResolutionName, Source | Model] = {}
        self.graph: dict[ResolutionName, list[ResolutionName]] = {}

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

            for resolution in step.dependencies:
                if resolution.name not in self.nodes:
                    raise ValueError(f"Step {resolution.name} not added to DAG")
            self.graph[step.name] = [parent.name for parent in step.parents]
        else:
            self.graph[step.name] = []

        self.nodes[step.name] = step

    @property
    def run(self) -> RunID:
        """Return run ID if available, else error."""
        if self._run:
            return self._run

        raise RuntimeError(
            "The DAG has not been connected to the server."
            "Start a new run or load a default one."
        )

    @run.setter
    def run(self, run_id: RunID) -> None:
        """Set run ID manually."""
        self._run = run_id

    @property
    def final_step(self) -> Source | Model:
        """Returns the root node in the DAG.

        Returns:
            The root node in the DAG

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
            return self.nodes[apex_nodes.pop()]

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

    def add_resolution(
        self,
        name: ResolutionName,
        resolution: Resolution,
    ) -> None:
        """Convert a resolution from the server to a Source or Model and add to DAG."""
        if resolution.resolution_type == ResolutionType.SOURCE:
            self.source(
                location=Location.from_config(resolution.config.location_config),
                name=SourceResolutionName(name),
                extract_transform=resolution.config.extract_transform,
                key_field=resolution.config.key_field,
                index_fields=resolution.config.index_fields,
                description=resolution.description,
                infer_types=False,
                validate_etl=False,
            )
        elif resolution.resolution_type == ResolutionType.MODEL:
            self.model(
                name=ModelResolutionName(name),
                description=resolution.description,
                model_class=resolution.config.model_class,
                model_settings=json.loads(resolution.config.model_settings),
                left_query=Query.from_config(resolution.config.left_query, dag=self),
                right_query=Query.from_config(resolution.config.right_query, dag=self)
                if resolution.config.right_query
                else None,
                truth=truth_int_to_float(resolution.truth),
            )
        else:
            raise ValueError(f"Unknown resolution type {resolution.resolution_type}")

    def get_source(self, name: ResolutionName) -> Source:
        """Get a source by name from the DAG.

        Args:
            name: The name of the source to retrieve.

        Returns:
            The Source object.

        Raises:
            ValueError: If the name doesn't exist in the DAG or isn't a Source.
        """
        if name not in self.nodes:
            raise ValueError(f"Source '{name}' not found in DAG")

        node = self.nodes[name]
        if not isinstance(node, Source):
            raise ValueError(
                f"Node '{name}' is not a Source, it's a {type(node).__name__}"
            )

        return node

    def get_model(self, name: ResolutionName) -> Model:
        """Get a model by name from the DAG.

        Args:
            name: The name of the model to retrieve.

        Returns:
            The Model object.

        Raises:
            ValueError: If the name doesn't exist in the DAG or isn't a Model.
        """
        if name not in self.nodes:
            raise ValueError(f"Model '{name}' not found in DAG")

        node = self.nodes[name]
        if not isinstance(node, Model):
            raise ValueError(
                f"Node '{name}' is not a Model, it's a {type(node).__name__}"
            )

        return node

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
        root_name = self.final_step.name
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

    def new_run(self) -> Self:
        """Start a new run."""
        try:
            collection = _handler.get_collection(self.name)
        except MatchboxCollectionNotFoundError:
            _handler.create_collection(self.name)
            collection = _handler.get_collection(self.name)

        # Delete non-default runs
        for run in collection.runs:
            if run != collection.default_run:
                _handler.delete_run(collection=self.name, run_id=run, certain=True)

        # Start a new run
        self.run = _handler.create_run(collection=self.name).run_id

        return self

    def set_client(self, client: Any) -> Self:
        """Assign a client to all sources at once."""
        for node in self.nodes.values():
            if isinstance(node, Source):
                node.location.set_client(client)

        return self

    def _load_run(self, run_id: RunID) -> Self:
        """Attach the specified run ID to the current DAG.

        Args:
            run_id: The ID of the run to attach
        """
        run: Run = _handler.get_run(collection=self.name, run_id=run_id)
        self.run: RunID = run_id

        def _len_dependencies(res_item: tuple[ResolutionName, Resolution]) -> int:
            return len(res_item[1].config.dependencies)

        sorted_resolutions: tuple[ResolutionName, Resolution] = sorted(
            run.resolutions.items(), key=_len_dependencies
        )

        for name, resolution in sorted_resolutions:
            self.add_resolution(name=name, resolution=resolution)

        return self

    def load_default(self) -> Self:
        """Attach to default run in this collection, loading all DAG nodes."""
        collection: Collection = _handler.get_collection(self.name)

        if not collection.default_run:
            raise RuntimeError("No default run set.")

        return self._load_run(collection.default_run)

    def load_pending(self) -> Self:
        """Attach to the pending run in this collection, loading all DAG nodes.

        Pending is defined as the last non-default run.
        """
        collection: Collection = _handler.get_collection(self.name)

        pending_runs: list[RunID] = [
            run_id for run_id in collection.runs if run_id != collection.default_run
        ]

        if not pending_runs:
            raise RuntimeError("No pending runs available.")

        return self._load_run(pending_runs[0])

    def run_and_sync(
        self,
        full_rerun: bool = False,
        start: str | None = None,
        finish: str | None = None,
    ):
        """Run entire DAG and send results to server."""
        start_time = datetime.datetime.now()

        # Determine order of execution steps
        root_node = self.final_step.name

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
                node.run(full_rerun=full_rerun)
                node.sync()
            except Exception as e:
                logger.error(f"❌ {node.name} failed: {e}")
                raise e

        logger.info("\n" + self.draw(start_time=start_time, skipped=skipped_nodes))

    def set_default(self) -> None:
        """Set the current run as the default for the collection.

        Makes it immutable, then moves the default pointer to it.
        """
        _handler.set_run_mutable(collection=self.name, run_id=self.run, mutable=False)
        _handler.set_run_default(collection=self.name, run_id=self.run, default=True)

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

        Returns:
            Dictionary mapping source names to list of keys within that source.

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
            targets=[
                ResolutionPath(name=target, collection=self.name, run=self.run)
                for target in to_sources
            ],
            source=ResolutionPath(name=from_source, collection=self.name, run=self.run),
            key=key,
            resolution=self.final_step.resolution_path,
            threshold=threshold,
        )

        to_sources_results = {m.target.name: list(m.target_id) for m in matches}
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
        # Get all sources in scope of the DAG run
        sources = {
            node_name: node
            for node_name, node in self.nodes.items()
            if isinstance(node, Source)
        }

        filtered_source_names = list(sources.keys())

        if source_filter:
            filtered_source_names = [
                s for s in filtered_source_names if s in source_filter
            ]

        if location_names:
            filtered_source_names = [
                s
                for s in filtered_source_names
                if sources[s].config.location_config.name in location_names
            ]

        if not filtered_source_names:
            raise MatchboxResolutionNotFoundError("No compatible source was found")

        source_mb_ids: list[ArrowTable] = []
        source_to_key_field: dict[str, str] = {}

        for source_name in filtered_source_names:
            # Get Matchbox IDs from backend
            source_mb_ids.append(
                _handler.query(
                    source=sources[source_name].resolution_path,
                    resolution=self.final_step.resolution_path,
                    return_leaf_id=False,
                )
            )

            source_to_key_field[source_name] = sources[source_name].key_field.name

        # Join Matchbox IDs to form mapping table
        mapping = source_mb_ids[0]
        mapping = mapping.rename_columns(
            {
                "key": sources[filtered_source_names[0]].config.qualified_key(
                    filtered_source_names[0]
                )
            }
        )
        if len(filtered_source_names) > 1:
            for source_name, mb_ids in zip(
                filtered_source_names[1:], source_mb_ids[1:], strict=True
            ):
                mapping = mapping.join(
                    right_table=mb_ids, keys="id", join_type="full outer"
                )
                mapping = mapping.rename_columns(
                    {"key": sources[source_name].config.qualified_key(source_name)}
                )

        return mapping
