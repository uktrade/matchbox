"""Shared fusion utilities for resolver execution and query-time recomputation."""

from collections.abc import Iterable

import polars as pl

from matchbox.common.dtos import FusionStrategy
from matchbox.common.transform import DisjointSet


def _as_component_assignments(assignments: pl.DataFrame) -> pl.DataFrame:
    """Normalise assignments to a stable ``cluster_id``/``node_id`` shape."""
    if assignments.height == 0:
        return pl.DataFrame(schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64})

    return assignments.select("cluster_id", "node_id").cast(
        {
            "cluster_id": pl.UInt64,
            "node_id": pl.UInt64,
        }
    )


def _union_components(
    model_edges: Iterable[pl.DataFrame],
    resolver_assignments: Iterable[pl.DataFrame],
) -> pl.DataFrame:
    """Fuse model edges and resolver assignments into connected components."""
    djs = DisjointSet[int]()
    all_nodes: set[int] = set()

    for edges in model_edges:
        if edges.height == 0:
            continue

        for left_id, right_id in edges.select("left_id", "right_id").iter_rows():
            left = int(left_id)
            right = int(right_id)
            all_nodes.update((left, right))
            djs.union(left, right)

    for assignments in resolver_assignments:
        assignment_df = _as_component_assignments(assignments)
        if assignment_df.height == 0:
            continue

        grouped_nodes = assignment_df.group_by("cluster_id").agg(
            pl.col("node_id").unique().sort()
        )

        for node_ids in grouped_nodes["node_id"]:
            component = [int(node_id) for node_id in node_ids]
            if not component:
                continue
            all_nodes.update(component)
            anchor = component[0]
            djs.add(anchor)
            for other in component[1:]:
                djs.union(anchor, other)

    for node_id in all_nodes:
        djs.add(node_id)

    components = [sorted(component) for component in djs.get_components()]
    components = sorted(components, key=lambda values: values[0])

    rows: list[dict[str, int]] = []
    for cluster_id, component in enumerate(components, start=1):
        for node_id in component:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "node_id": node_id,
                }
            )

    if not rows:
        return pl.DataFrame(schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64})

    return pl.DataFrame(rows).cast({"cluster_id": pl.UInt64, "node_id": pl.UInt64})


def fuse_components(
    *,
    strategy: FusionStrategy,
    model_edges: Iterable[pl.DataFrame],
    resolver_assignments: Iterable[pl.DataFrame],
) -> pl.DataFrame:
    """Fuse model and resolver inputs into materialised component assignments."""
    if strategy != FusionStrategy.UNION:
        raise ValueError("Only FusionStrategy.UNION is implemented")

    return _union_components(
        model_edges=model_edges,
        resolver_assignments=resolver_assignments,
    )
