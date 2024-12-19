from typing import NamedTuple

import pandas as pd
import pyarrow as pa
import rustworkx as rx
from sqlalchemy import Engine, and_, case, func, select
from sqlalchemy.orm import Session

from matchbox.common.graph import ResolutionNodeType
from matchbox.common.results import (
    ClusterResults,
    ModelMetadata,
    ModelType,
    ProbabilityResults,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
)


class SourceInfo(NamedTuple):
    """Information about a model's sources."""

    left: int
    right: int | None
    left_ancestors: set[int]
    right_ancestors: set[int] | None


def _get_model_parents(
    engine: Engine, resolution_id: int
) -> tuple[bytes, bytes | None]:
    """Get the model's immediate parent models."""
    parent_query = (
        select(Resolutions.resolution_id, Resolutions.type)
        .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.parent)
        .where(ResolutionFrom.child == resolution_id)
        .where(ResolutionFrom.level == 1)
    )

    with engine.connect() as conn:
        parents = conn.execute(parent_query).fetchall()

    if len(parents) == 1:
        return parents[0][0], None
    elif len(parents) == 2:
        p1, p2 = parents
        p1_id, p1_type = p1
        p2_id, p2_type = p2
        # Put dataset first if it exists
        if p1_type == ResolutionNodeType.DATASET:
            return p1_id, p2_id
        elif p2_type == ResolutionNodeType.DATASET:
            return p2_id, p1_id
        # Both models, maintain original order
        return p1_id, p2_id
    else:
        raise ValueError(f"Model has unexpected number of parents: {len(parents)}")


def _get_source_info(engine: Engine, resolution_id: int) -> SourceInfo:
    """Get source resolutions and their ancestry information."""
    left_id, right_id = _get_model_parents(engine=engine, resolution_id=resolution_id)

    with Session(engine) as session:
        left = session.get(Resolutions, left_id)
        right = session.get(Resolutions, right_id) if right_id else None

        left_ancestors = {left_id} | {m.resolution_id for m in left.ancestors}
        if right:
            right_ancestors = {right_id} | {m.resolution_id for m in right.ancestors}
        else:
            right_ancestors = None

    return SourceInfo(
        left=left_id,
        right=right_id,
        left_ancestors=left_ancestors,
        right_ancestors=right_ancestors,
    )


def get_model_probabilities(
    engine: Engine, resolution: Resolutions
) -> ProbabilityResults:
    """
    Recover the model's ProbabilityResults.

    For each probability this model assigned:
    - Get its two immediate children
    - Filter for children that aren't parents of other clusters this model scored
    - Determine left/right by tracing ancestry to source resolutions using query helpers

    Args:
        engine: SQLAlchemy engine
        resolution: Resolution of type model to query

    Returns:
        ProbabilityResults containing the original pairwise probabilities
    """
    if resolution.type != ResolutionNodeType.MODEL:
        raise ValueError("Expected resolution of type model")

    source_info: SourceInfo = _get_source_info(
        engine=engine, resolution_id=resolution.resolution_id
    )

    with Session(engine) as session:
        left = session.get(Resolutions, source_info.left)
        right = (
            session.get(Resolutions, source_info.right) if source_info.right else None
        )

        metadata = ModelMetadata(
            name=resolution.name,
            description=resolution.description or "",
            type=ModelType.DEDUPER if source_info.right is None else ModelType.LINKER,
            left_source=left.name,
            right_source=right.name if source_info.right else None,
        )

        # First get all clusters this resolution assigned probabilities to
        resolution_clusters = (
            select(Probabilities.cluster)
            .where(Probabilities.resolution == resolution.resolution_id)
            .cte("resolution_clusters")
        )

        # Get clusters that are parents in Contains for resolution's probabilities
        resolution_parents = (
            select(Contains.parent)
            .join(resolution_clusters, Contains.child == resolution_clusters.c.cluster)
            .cte("resolution_parents")
        )

        # Get valid pairs (those with exactly 2 children)
        # where neither child is a parent in the resolution's hierarchy
        valid_pairs = (
            select(Contains.parent)
            .join(
                Probabilities,
                and_(
                    Probabilities.cluster == Contains.parent,
                    Probabilities.resolution == resolution.resolution_id,
                ),
            )
            .where(~Contains.child.in_(select(resolution_parents)))
            .group_by(Contains.parent)
            .having(func.count() == 2)
            .cte("valid_pairs")
        )

        # Join to get children and probabilities
        pairs = (
            select(
                Contains.parent.label("id"),
                func.array_agg(
                    case(
                        (
                            Contains.child.in_(list(source_info.left_ancestors)),
                            Contains.child,
                        ),
                        (
                            Contains.child.in_(list(source_info.right_ancestors))
                            if source_info.right_ancestors
                            else Contains.child.notin_(
                                list(source_info.left_ancestors)
                            ),
                            Contains.child,
                        ),
                    )
                ).label("children"),
                func.min(Probabilities.probability).label("probability"),
            )
            .join(valid_pairs, valid_pairs.c.parent == Contains.parent)
            .join(
                Probabilities,
                and_(
                    Probabilities.cluster == Contains.parent,
                    Probabilities.resolution == resolution.resolution_id,
                ),
            )
            .group_by(Contains.parent)
        ).cte("pairs")

        # Final select to properly split out left and right
        final_select = select(
            pairs.c.id,
            pairs.c.children[1].label("left_id"),
            pairs.c.children[2].label("right_id"),
            pairs.c.probability,
        )

        results = session.execute(final_select).fetchall()

        df = pd.DataFrame(
            results, columns=["id", "left_id", "right_id", "probability"]
        ).astype(
            {
                "id": pd.ArrowDtype(pa.int32()),
                "left_id": pd.ArrowDtype(pa.int32()),
                "right_id": pd.ArrowDtype(pa.int32()),
                "probability": pd.ArrowDtype(pa.float32()),
            }
        )

        return ProbabilityResults(dataframe=df, metadata=metadata)


def _get_all_leaf_descendants(graph: rx.PyDiGraph, node_id: int) -> set[int]:
    """Get all leaf descendant node IDs of a given node in the graph."""
    descendants = set()
    to_process = [node_id]

    while to_process:
        current = to_process.pop()
        children = [edge[1] for edge in graph.out_edges(current)]

        if not children:
            descendants.add(current)
        else:
            to_process.extend(children)

    return descendants


def get_model_clusters(engine: Engine, resolution: Resolutions) -> ClusterResults:
    """
    Recover the model's Clusters.

    Clusters are the connected components of the model at every threshold.

    While they're stored in a hierarchical structure, we need to recover the
    original components, where all child IDs are leaf Clusters.

    Args:
        engine: SQLAlchemy engine
        model: Resolution of type model to query

    Returns:
        A ClusterResults object containing connected components and model metadata
    """
    if resolution.type != ResolutionNodeType.MODEL:
        raise ValueError("Expected resolution of type model")

    source_info: SourceInfo = _get_source_info(
        engine=engine, resolution_id=resolution.resolution_id
    )

    with Session(engine) as session:
        # Build metadata
        left = session.get(Resolutions, source_info.left)
        right = (
            session.get(Resolutions, source_info.right) if source_info.right else None
        )

        metadata = ModelMetadata(
            name=resolution.name,
            description=resolution.description or "",
            type=ModelType.DEDUPER if source_info.right is None else ModelType.LINKER,
            left_source=left.name,
            right_source=right.name if source_info.right else None,
        )

        # Get all clusters and their relationships for this resolution
        hierarchy_query = (
            select(Contains.parent, Contains.child, Probabilities.probability)
            .join(
                Probabilities,
                and_(
                    Probabilities.cluster == Contains.parent,
                    Probabilities.resolution == resolution.resolution_id,
                ),
            )
            .order_by(Probabilities.probability.desc())
        )

        hierarchy = session.execute(hierarchy_query).fetchall()

        # Get all leaf nodes (clusters with no children) and their IDs
        leaf_query = select(Clusters.cluster_id, Clusters.source_pk).where(
            ~Clusters.cluster_id.in_(select(Contains.parent).distinct())
        )
        leaf_nodes = {
            row.cluster_id: row.source_pk[0] if row.source_pk else None
            for row in session.execute(leaf_query)
        }

        # Get unique thresholds and components at each threshold
        threshold_query = (
            select(Probabilities.cluster, Probabilities.probability)
            .where(Probabilities.resolution == resolution.resolution_id)
            .order_by(Probabilities.probability.desc())
        )
        threshold_components = session.execute(threshold_query).fetchall()

    # Build directed graph of the full hierarchy
    graph = rx.PyDiGraph()
    nodes: dict[bytes, int] = {}  # node_hash -> node_id

    def get_node_id(id: bytes) -> int:
        if id not in nodes:
            nodes[id] = graph.add_node(id)
        return nodes[id]

    for parent, child, prob in hierarchy:
        parent_id = get_node_id(parent)
        child_id = get_node_id(child)
        graph.add_edge(parent_id, child_id, prob)

    # Process each threshold level
    components: list[tuple[bytes, bytes, float]] = []
    seen_combinations = set()

    threshold_groups = {}
    for comp, thresh in threshold_components:
        if thresh not in threshold_groups:
            threshold_groups[thresh] = []
        threshold_groups[thresh].append(comp)

    # Process thresholds in descending order
    for threshold in sorted(threshold_groups.keys(), reverse=True):
        for component in threshold_groups[threshold]:
            component_id = get_node_id(component)

            leaf_ids = _get_all_leaf_descendants(graph, component_id)

            leaf_hashes = {
                graph.get_node_data(leaf_id)
                for leaf_id in leaf_ids
                if graph.get_node_data(leaf_id) in leaf_nodes
            }

            for leaf in leaf_hashes:
                if leaf_nodes[leaf] is not None:
                    relation = (component, leaf, threshold)
                    if relation not in seen_combinations:
                        components.append(relation)
                        seen_combinations.add(relation)

    df = pd.DataFrame(components, columns=["parent", "child", "threshold"]).astype(
        {
            "parent": pd.ArrowDtype(pa.int32()),
            "child": pd.ArrowDtype(pa.int32()),
            "threshold": pd.ArrowDtype(pa.float32()),
        }
    )

    return ClusterResults(
        dataframe=df,
        metadata=metadata,
    )
