"""Objects to define a DAG which indexes, deduplicates and links data."""

import datetime
from collections import defaultdict
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine

from matchbox.client.collections import Version
from matchbox.client.models.models import make_model
from matchbox.client.queries import Selector, clean, query
from matchbox.common.dags import DAG, DedupeStep, IndexStep, Step, StepInput
from matchbox.common.graph import ResolutionName, SourceResolutionName
from matchbox.common.logging import logger
from matchbox.common.sources import RelationalDBLocation, SourceConfig, SourceField


def find_apex(dag: DAG) -> tuple[dict[str, list[str]], str]:
    """Find apex of DAG.

    Raises:
        ValueError: If the DAG has multiple disconnected components
    """
    inverse_graph = defaultdict(list)
    for node in dag.graph:
        for neighbor in dag.graph[node]:
            inverse_graph[neighbor].append(node)

    apex_nodes = {node for node in dag.graph if node not in inverse_graph}
    if len(apex_nodes) > 1:
        raise ValueError("Some models or sources are disconnected")
    elif not apex_nodes:
        raise ValueError("No root node found, DAG might contain cycles")
    else:
        return apex_nodes.pop()


def running_sequence(dag: DAG) -> list[str]:
    """Determine order of execution of steps."""
    apex = find_apex(dag)

    def depth_first(node: str, sequence: list):
        sequence.append(node)
        for neighbour in dag.graph[node]:
            if neighbour not in sequence:
                depth_first(neighbour, sequence)

    inverse_sequence = []
    depth_first(apex, inverse_sequence)
    return list(reversed(inverse_sequence))


def query_step_input(step_input: StepInput) -> pl.DataFrame:
    """Retrieve data for step input.

    Args:
        step_input: Declared input to this DAG step.

    Returns:
        Polars dataframe with retrieved results.
    """
    selectors: list[Selector] = []

    for source, fields in step_input.select.items():
        field_lookup: dict[str, SourceField] = {
            field.name: field for field in source.index_fields
        }

        selected_fields: list[SourceField] = []
        for field in fields:
            selected_fields.append(field_lookup[field])

        selectors.append(Selector(source=source, fields=selected_fields))

    return query(
        selectors,
        return_leaf_id=False,
        return_type="polars",
        threshold=step_input.threshold,
        resolution=step_input.name,
        batch_size=step_input.batch_size,
        combine_type=step_input.combine_type,
    )


def run_step(version: Version, step: Step) -> Any:
    """Run appropriate logic depending on type of step."""
    if isinstance(step, IndexStep):
        data_hashes = version.add_source(
            source_config=SourceConfig, batch_size=step.batch_size
        )

        return data_hashes

    left_raw = query_step_input(step.left)
    left_clean = clean(left_raw, step.left.cleaning_dict)

    if isinstance(step, DedupeStep):
        model = make_model(
            name=step.name,
            description=step.description,
            model_class=step.model_class,
            model_settings=step.settings,
            left_data=left_clean,
            left_resolution=step.left.name,
        )

    else:
        right_raw = query_step_input(step.right)
        right_clean = clean(right_raw, step.right.cleaning_dict)

        model = make_model(
            name=step.name,
            description=step.description,
            model_class=step.model_class,
            model_settings=step.settings,
            left_data=left_clean,
            left_resolution=step.left.name,
            right_data=right_clean,
            right_resolution=step.right.name,
        )

    results = model.run()
    results.to_matchbox()
    model.truth = step.truth
    return results


def draw(
    dag: DAG,
    start_time: datetime.datetime | None = None,
    doing: ResolutionName | None = None,
    skipped: list[ResolutionName] | None = None,
) -> str:
    """Create a string representation of the DAG as a tree structure.

    If `start_time` is provided, it will show the status of each node
    based on the last run time. The status indicators are:

    * ✅ Done
    * 🔄 Working
    * ⏸️ Awaiting
    * ⏭️ Skipped

    Args:
        dag: DAG to draw.
        start_time: Start time of the DAG run. Used to calculate node status.
        doing: Name of the node currently being processed (if any).
        skipped: List of node names that were skipped.

    Returns:
        String representation of the DAG with status indicators.
    """
    root_name = find_apex(dag)
    skipped = skipped or []

    def _get_node_status(name: ResolutionName) -> str:
        """Determine the status indicator for a node."""
        if name in skipped:
            return "⏭️"
        elif doing and name == doing:
            return "🔄"
        elif (
            (node := dag.nodes.get(name))
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

    def format_children(node: ResolutionName, prefix=""):
        """Recursively format the children of a node."""
        children = []
        # Get all outgoing edges from this node
        for target in dag.graph.get(node, []):
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


class DAGDebugOptions(BaseModel):
    """Debug configuration options for DAG."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: ResolutionName | None = None
    finish: ResolutionName | None = None
    override_sources: dict[SourceResolutionName, pl.DataFrame] = {}
    keep_outputs: bool = False


def run_dag(dag: DAG, debug_options: DAGDebugOptions | None = None) -> dict[str, Any]:
    """Run entire DAG.

    Args:
        dag: DAG to run.
        debug_options: configuration options for debug run
    """
    debug_options = debug_options or DAGDebugOptions()
    start_time = datetime.datetime.now()

    sequence = running_sequence(dag)

    # Identify skipped nodes
    skipped_nodes = []
    if debug_options.start:
        try:
            start_index = sequence.index(debug_options.start)
            skipped_nodes = sequence[:start_index]
        except ValueError as e:
            raise ValueError(f"Step {debug_options.start} not in DAG") from e
    else:
        start_index = 0

    # Determine end index
    if debug_options.finish:
        try:
            end_index = sequence.index(debug_options.finish) + 1
            skipped_nodes.extend(sequence[end_index:])
        except ValueError as e:
            raise ValueError(f"Step {debug_options.finish} not in DAG") from e
    else:
        end_index = len(sequence)

    # Create debug warehouse if needed
    if len(debug_options.override_sources):
        debug_sqlite_uri = "sqlite:///:memory:"
        debug_engine = create_engine(debug_sqlite_uri)
        debug_location = RelationalDBLocation(name="__DEBUG__", client=debug_engine)

    debug_outputs = {}
    for step_name in sequence[start_index:end_index]:
        node = dag.nodes[step_name]
        if step_name in debug_options.override_sources and isinstance(node, IndexStep):
            with debug_engine.connect() as conn:
                debug_options.override_sources[step_name].write_database(
                    table_name=step_name,
                    connection=conn,
                    if_table_exists="replace",
                )
            node = IndexStep(
                source_config=SourceConfig(
                    location=debug_location,
                    name=step_name,
                    extract_transform=f"select * from {step_name}",
                    key_field=node.source_config.key_field,
                    index_fields=node.source_config.index_fields,
                )
            )

        logger.info(
            "\n" + draw(start_time=start_time, doing=node.name, skipped=skipped_nodes)
        )
        try:
            if debug_options.keep_outputs:
                debug_outputs[step_name] = run_step(node)
            else:
                run_step(node)
            node.last_run = datetime.datetime.now()
        except Exception as e:
            logger.error(f"❌ {node.name} failed: {e}")
            raise e

    logger.info("\n" + draw(start_time=start_time, skipped=skipped_nodes))
    return debug_outputs
