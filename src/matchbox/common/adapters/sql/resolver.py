"""Resolver cluster canonicalisation, shared across relational backends."""

from collections.abc import Iterable

import pyarrow as pa
from sqlalchemy import Row, func, select
from sqlalchemy.sql.expression import TableClause
from sqlalchemy.sql.selectable import Select, Subquery

from matchbox.common.adapters.sql import tables
from matchbox.common.transform import hash_cluster_leaves


def build_expanded_leaves_subquery(
    incoming_cluster_assignments: TableClause,
) -> Subquery:
    """Expand child assignments to leaf-level cluster IDs per parent cluster."""
    return (
        select(
            incoming_cluster_assignments.c.parent_id,
            func.coalesce(
                tables.Contains.c.leaf,
                incoming_cluster_assignments.c.child_id,
            ).label("leaf_id"),
        )
        .distinct()
        .select_from(
            # Clusters with no children in contains resolve to themselves
            incoming_cluster_assignments.outerjoin(
                tables.Contains,
                tables.Contains.c.root == incoming_cluster_assignments.c.child_id,
            )
        )
        .subquery("expanded_leaves")
    )


def build_leaf_hash_groups_query(expanded_leaves: Subquery) -> Select:
    """Build a query grouping leaf hashes per parent cluster.

    Uses an inner join to the clusters table, so unknown leaf IDs are
    silently dropped.
    """
    return (
        select(
            expanded_leaves.c.parent_id,
            func.array_agg(tables.Clusters.c.cluster_hash).label("leaf_hashes"),
        )
        .select_from(
            expanded_leaves.join(
                tables.Clusters,
                tables.Clusters.c.cluster_id == expanded_leaves.c.leaf_id,
            )
        )
        .group_by(expanded_leaves.c.parent_id)
    )


def hash_resolver_parents(
    rows: Iterable[Row],
) -> pa.Table:
    """Compute a single composite hash per parent cluster from leaf hashes.

    Args:
        rows: (parent_id, leaf_hashes) rows, as returned by executing
            build_leaf_hash_groups_query.

    Returns:
        Arrow table with columns (parent_id: int64, cluster_hash: binary).
    """
    parent_ids: list[int] = []
    cluster_hashes: list[bytes] = []
    for parent_id, leaf_hashes in rows:
        parent_ids.append(int(parent_id))
        cluster_hashes.append(hash_cluster_leaves([bytes(h) for h in leaf_hashes]))

    return pa.table(
        {
            "parent_id": parent_ids,
            "cluster_hash": cluster_hashes,
        }
    )
