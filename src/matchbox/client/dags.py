"""Objects to define a DAG which indexes, deduplicates and links data."""

import json
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, TypeAlias

import polars as pl
from platformdirs import user_cache_path

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.locations import Location
from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.results import ResolvedMatches
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
from matchbox.common.logging import log_mem_usage, logger, profile_time
from matchbox.common.transform import threshold_float_to_int, threshold_int_to_float


class DAGNodeExecutionStatus(StrEnum):
    """Enumeration of node execution statuses."""

    SKIPPED = "skipped"
    DONE = "done"
    DOING = "doing"


DAGExecutionStatus: TypeAlias = dict[str, DAGNodeExecutionStatus]

CACHE_DIR = user_cache_path("matchbox")


class DAG:
    """Self-sufficient pipeline of indexing, deduping and linking steps."""

    def __init__(self, name: str) -> None:
        """Initialises empty DAG."""
        self.name: CollectionName = CollectionName(name)
        self._run: RunID | None = None
        self.nodes: dict[ResolutionName, Source | Model] = {}
        self.graph: dict[ResolutionName, list[ResolutionName]] = {}

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_dir = tempfile.TemporaryDirectory(dir=str(CACHE_DIR))
        self.cache_path = Path(self._cache_dir.name)

    def _check_dag(self, dag: Self) -> None:
        """Check that the given DAG is the same as this one."""
        if self != dag:
            raise ValueError("Cannot mix DAGs")

    def _add_step(self, step: Source | Model) -> None:
        """Validate and add sources and models to DAG."""
        self._check_dag(step.dag)

        # We allow overwriting nodes, because:
        # - One might want to iterate on a local DAG without starting over
        # - One might need to load a pending run in a script which defines the same
        #   DAG, to re-run a specific node.
        #
        # We only mandate that a model's direct inputs not be changed. This ensures
        # that `self.graph` is not altered. It does not guarantee that the new node
        # works well with the rest of the DAG, which must be verified by the user.
        # For example, you could remove source fields that are needed downstream, or
        # you could query sources that are not available to a point of truth.
        # These issues are not checked when adding a node for the first time either.
        if step.name in self.nodes:
            if step.config.parents != self.nodes[step.name].config.parents:
                raise ValueError("Cannot re-assign name to model with different inputs")
            logger.info(f"Overwriting node '{step.name}'.")

        if isinstance(step, Model):
            self._check_dag(step.left_query.dag)
            if step.right_query:
                self._check_dag(step.right_query.dag)

            for resolution in step.config.dependencies:
                if resolution not in self.nodes:
                    raise ValueError(f"Step {resolution} not added to DAG")
            self.graph[step.name] = [parent for parent in step.config.parents]
        else:
            self.graph[step.name] = []

        self.nodes[step.name] = step

    @classmethod
    def list_all(cls) -> list[CollectionName]:
        """List available DAG names on the server."""
        return _handler.list_collections()

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
    def final_steps(self) -> list[Source | Model]:
        """Returns all apex nodes in the DAG.

        Returns:
            List of all apex nodes (nodes with no incoming edges).
            Returns empty list if DAG is empty.
        """
        if not self.nodes:
            return []

        # Find all nodes that appear as dependencies
        all_dependencies = set()
        for deps in self.graph.values():
            all_dependencies.update(deps)

        # Apex nodes are those that don't appear as anyone's dependency
        apex_node_names = [node for node in self.graph if node not in all_dependencies]
        return [self.nodes[name] for name in apex_node_names]

    @property
    def final_step(self) -> Source | Model:
        """Returns the root node in the DAG.

        Returns:
            The root node in the DAG

        Raises:
            ValueError: If the DAG does not have exactly one final step
        """
        steps = self.final_steps

        if len(steps) == 0:
            raise ValueError("No root node found, DAG might contain cycles")
        elif len(steps) > 1:
            raise ValueError("Some models or sources are unreachable")
        else:
            return steps[0]

    def source(self, *args: Any, **kwargs: Any) -> Source:
        """Create Source and add it to the DAG."""
        source = Source(*args, **kwargs, dag=self)
        self._add_step(source)

        return source

    def model(self, *args: Any, **kwargs: Any) -> Model:
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
                truth=threshold_int_to_float(resolution.truth),
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

    def query(self, *args: Any, **kwargs: Any) -> Query:
        """Create Query object."""
        return Query(*args, **kwargs, dag=self)

    def draw(self, status: DAGExecutionStatus | None = None) -> str:
        """Create a string representation of the DAG as a tree structure.

        If `status` is provided, it will show the status of each node.
        The status indicators are:

        * ✅ Done
        * 🔄 Working
        * ⏸️ Awaiting
        * ⏭️ Skipped

        Args:
            status: Object describing the status of each node.

        Returns:
            String representation of the DAG with status indicators.
        """
        # Handle empty DAG
        if not self.nodes:
            return "Empty DAG"

        apex_nodes = self.final_steps
        if not apex_nodes:
            return "No apex nodes found (possible cycle in DAG)"

        def _get_node_indicator(name: str) -> str:
            """Determine the status indicator for a node."""
            if name not in status:
                return "⏸️"
            elif status[name] == DAGNodeExecutionStatus.SKIPPED:
                return "⏭️"
            elif status[name] == DAGNodeExecutionStatus.DOING:
                return "🔄"
            elif status[name] == DAGNodeExecutionStatus.DONE:
                return "✅"

        # Header with collection and run info
        head_collection: str = f"Collection: {self.name}"
        head_run: str = f"└── Run: {self._run or '⛓️‍💥 Disconnected'}"

        result: list[str] = [head_collection, head_run, ""]
        visited = set()

        def format_children(node: str, prefix: str = "") -> None:
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

                # Add status indicator if status is provided
                if status is not None:
                    indicator = _get_node_indicator(child)
                    child_display = f"{indicator} {child}"
                else:
                    child_display = child

                if is_last:
                    result.append(f"{prefix}└── {child_display}")
                    format_children(child, prefix + "    ")
                else:
                    result.append(f"{prefix}├── {child_display}")
                    format_children(child, prefix + "│   ")

        # Draw each apex node
        for i, apex_node in enumerate(apex_nodes):
            root_name = apex_node.name

            if i > 0:
                result.append("")  # Blank line between disconnected components

            visited.add(root_name)

            if status is not None:
                indicator = _get_node_indicator(root_name)
                result.append(f"{indicator} {root_name}")
            else:
                result.append(root_name)

            format_children(root_name)

        return "\n".join(result)

    def __str__(self) -> str:
        """Return string representation of the DAG."""
        return self.draw()

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

    def set_client(self, client: Any) -> Self:  # noqa: ANN401
        """Assign a client to all sources at once."""
        for node in self.nodes.values():
            if isinstance(node, Source):
                node.location.set_client(client)

        return self

    def _load_run(self, run_id: RunID) -> Self:
        """Attach the specified run ID to the current DAG.

        Topologically sorts using Kahn's algorithm.

        Args:
            run_id: The ID of the run to attach
        """
        run: Run = _handler.get_run(collection=self.name, run_id=run_id)
        self.run: RunID = run_id

        resolutions: dict[ResolutionName, Resolution] = run.resolutions

        # Build dependency graph and track added
        added: set[ResolutionName] = set()
        deps_graph: dict[ResolutionName, set[ResolutionName]] = {
            name: {dep for dep in res.config.dependencies}
            for name, res in resolutions.items()
        }

        # Kahn's algorithm: iteratively add resolutions whose dependencies are satisfied
        while len(added) < len(resolutions):
            # Find resolutions that can be added (all dependencies satisfied)
            ready = [
                name
                for name in resolutions
                if name not in added and deps_graph[name].issubset(added)
            ]

            if not ready:
                # No progress possible - circular dependency or missing dependency
                missing = set(resolutions.keys()) - added
                raise RuntimeError(
                    f"Cannot resolve dependencies. Remaining resolutions: {missing}. "
                    "This may indicate circular dependencies or references to "
                    "non-existent resolutions."
                )

            # Add all ready resolutions (order within this batch doesn't matter)
            for name in ready:
                self.add_resolution(name=name, resolution=resolutions[name])
                added.add(name)

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
        start: str | None = None,
        finish: str | None = None,
        low_memory: bool = False,
        batch_size: int | None = None,
        profile: bool = False,
    ) -> None:
        """Run entire DAG and send results to server.

        Args:
            start: Name of first node to run
            finish: Name of last node to run
            low_memory: Whether to delete data for each node after it is run
            batch_size: The size used for internal batching. Overrides environment
                variable if set.
            profile: whether to log to INFO level the memory usage
        """
        if batch_size is None:
            batch_size = settings.batch_size

        # Determine order of execution steps
        root_nodes = self.final_steps

        def depth_first(node: str, sequence: list) -> None:
            sequence.append(node)
            for neighbour in self.graph[node]:
                if neighbour not in sequence:
                    depth_first(neighbour, sequence)

        inverse_sequence = []
        for root_node in root_nodes:
            depth_first(root_node.name, inverse_sequence)
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

        status = {
            step_name: DAGNodeExecutionStatus.SKIPPED for step_name in skipped_nodes
        }

        for step_name in sequence:
            node = self.nodes[step_name]
            status[step_name] = DAGNodeExecutionStatus.DOING
            logger.info("\n" + self.draw(status=status))
            try:
                if isinstance(node, Source):
                    node.run(batch_size=batch_size)
                else:
                    node.run()
                node.sync()
                if profile:
                    log_mem_usage()
            except Exception as e:
                logger.error(f"❌ {node.name} failed: {e}")
                raise e
            status[step_name] = DAGNodeExecutionStatus.DONE

            if low_memory:
                node.clear_data()
                logger.info("Cleared node data")
                if profile:
                    log_mem_usage()
        logger.info("\n" + self.draw(status=status))

    def set_default(self) -> None:
        """Set the current run as the default for the collection.

        Makes it immutable, then moves the default pointer to it.
        """
        # Trigger error if there isn't a single root
        _ = self.final_step

        # tries to get apex, error if it doesn't exist
        _handler.set_run_mutable(collection=self.name, run_id=self.run, mutable=False)
        _handler.set_run_default(collection=self.name, run_id=self.run, default=True)

    def lookup_key(
        self,
        from_source: str,
        to_sources: list[str],
        key: str,
        threshold: float | None = None,
    ) -> dict[str, list[str]]:
        """Matches IDs against the selected backend.

        Args:
            from_source: Name of source the provided key belongs to
            to_sources: Names of sources to find keys in
            key: The value to match from the source. Usually a primary key
            threshold (optional): The threshold to use for creating clusters.
                If None, uses the resolutions' default threshold
                If a float, uses that threshold for the specified resolution, and the
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
        if threshold:
            if not isinstance(threshold, float):
                raise ValueError("If passed, threshold must be a float")
            threshold = threshold_float_to_int(threshold)
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

    @profile_time(kwarg="node")
    def resolve(
        self,
        node: ResolutionName | None = None,
        source_filter: list[str] | None = None,
        location_names: list[str] | None = None,
        threshold: float | None = None,
    ) -> ResolvedMatches:
        """Returns ResolvedMatches, optionally filtering.

        Args:
            node: Name of source or model to resolve within DAG.
                If not provided, will look for an apex.
            source_filter: An optional list of source resolution names to filter by.
            location_names: An optional list of location names to filter by.
            threshold (optional): The threshold to use for creating clusters.
                If None, uses the resolutions' default threshold
                If a float, uses that threshold for the specified resolution, and the
                resolution's cached thresholds for its ancestors
        """
        if threshold:
            if not isinstance(threshold, float):
                raise ValueError("If passed, threshold must be a float")
            threshold = threshold_float_to_int(threshold)
        point_of_truth = self.nodes[node] if node else self.final_step

        available_sources = {
            node_name: self.get_source(node_name)
            for node_name in point_of_truth.sources
        }

        filtered_source_names = list(available_sources.keys())

        if source_filter:
            filtered_source_names = [
                s for s in filtered_source_names if s in source_filter
            ]

        if location_names:
            filtered_source_names = [
                s
                for s in filtered_source_names
                if available_sources[s].config.location_config.name in location_names
            ]

        if not filtered_source_names:
            raise MatchboxResolutionNotFoundError("No compatible source was found")

        resolved_sources: list[Source] = []
        query_results: list[pl.DataFrame] = []
        for source_name in filtered_source_names:
            resolved_sources.append(available_sources[source_name])
            query_results.append(
                pl.from_arrow(
                    _handler.query(
                        source=available_sources[source_name].resolution_path,
                        resolution=point_of_truth.resolution_path,
                        return_leaf_id=True,
                        threshold=threshold,
                    )
                )
            )

        return ResolvedMatches(sources=resolved_sources, query_results=query_results)
