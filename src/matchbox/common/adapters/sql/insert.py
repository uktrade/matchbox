"""Ingest select builders, shared across relational backends.

literal() values use the generic BigInteger type, never a
dialect-specific one (e.g. postgresql.BIGINT).
"""

from sqlalchemy import BigInteger, exists, func, literal, select
from sqlalchemy.sql.expression import TableClause
from sqlalchemy.sql.selectable import Select, Subquery

from matchbox.common.adapters.sql import tables


def select_new_cluster_hashes(incoming: TableClause, hash_col: str = "hash") -> Select:
    """Distinct hashes in incoming absent from Clusters (anti-join)."""
    hash_column = incoming.c[hash_col]
    return (
        select(hash_column)
        .distinct()
        .where(~exists(select(1).where(tables.Clusters.c.cluster_hash == hash_column)))
    )


def select_cluster_map(incoming_hashes: TableClause) -> Subquery:
    """(parent_id, cluster_id) by joining staged hashes to Clusters."""
    return (
        select(incoming_hashes.c.parent_id, tables.Clusters.c.cluster_id)
        .select_from(
            incoming_hashes.join(
                tables.Clusters,
                tables.Clusters.c.cluster_hash == incoming_hashes.c.cluster_hash,
            )
        )
        .subquery("cluster_map")
    )


def select_key_expansion(incoming: TableClause, source_config_id: int) -> Select:
    """(cluster_id, source_config_id, key) with keys unnested."""
    return select(
        tables.Clusters.c.cluster_id,
        literal(source_config_id, BigInteger).label("source_config_id"),
        func.unnest(incoming.c["keys"]).label("key"),
    ).select_from(
        incoming.join(
            tables.Clusters, tables.Clusters.c.cluster_hash == incoming.c.hash
        )
    )


def select_contains_pairs(expanded_leaves: Subquery, cluster_map: Subquery) -> Select:
    """Distinct (root, leaf) pairs for the Contains insert."""
    return (
        select(
            cluster_map.c.cluster_id.label("root"),
            expanded_leaves.c.leaf_id.label("leaf"),
        )
        .select_from(
            expanded_leaves.join(
                cluster_map,
                expanded_leaves.c.parent_id == cluster_map.c.parent_id,
            )
        )
        .where(cluster_map.c.cluster_id != expanded_leaves.c.leaf_id)
        .distinct()
    )


def select_resolver_membership(step_id: int, cluster_map: Subquery) -> Select:
    """Distinct (step_id, cluster_id) for ResolverClusters."""
    return select(
        literal(step_id, BigInteger).label("step_id"),
        cluster_map.c.cluster_id,
    ).distinct()
