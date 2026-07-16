"""Query and match builders, shared across relational backends.

These functions build SQL statements from a pre-computed lineage: an
ordered list of (step_id, source_config_id) tuples, highest priority
first, rather than deriving it themselves from a live database. Callers
resolve lineage however suits their engine and pass the result in.

An entry with no source_config_id is a resolver node. The rest are
sources. Priority is encoded by list order, highest first.
"""

from typing import Literal

from sqlalchemy import and_, func, join, select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import CTE, Select, Subquery

from matchbox.common.adapters.sql import tables


def build_unified_query(
    lineage: list[tuple[int, int | None]],
    level: Literal["leaf", "key"] = "leaf",
    include_source_config_id: bool = False,
) -> Select:
    """Build a query that projects records to root IDs through the hierarchy.

    Args:
        lineage: Ordered (step_id, source_config_id) tuples, highest
            priority first.
        level: "leaf" deduplicates to one row per leaf cluster. "key" adds
            the source key, producing one row per (leaf, key) pair.
        include_source_config_id: whether to include the source config ID
            in the selection.
    """
    # Separate lineage entries into resolver steps and source config IDs.
    # Entries without a source_config_id are resolver nodes, the rest are sources
    resolver_ids: list[int] = []
    source_config_ids: list[int] = []
    for step_id, source_config_id in lineage:
        if source_config_id is None:
            resolver_ids.append(step_id)
        else:
            source_config_ids.append(source_config_id)

    # cluster_keys is the base table, resolver subqueries are LEFT JOINed onto it
    from_clause = tables.ClusterSourceKey
    projected_roots: list[ColumnElement[int]] = []

    for resolver_id in resolver_ids:
        # For each resolver, build a subquery that maps leaf cluster IDs to their
        # root cluster IDs at that step level via the contains table.
        assignments: Subquery = (
            select(
                tables.Contains.c.leaf.label("leaf_id"),
                tables.Contains.c.root.label("root_id"),
            )
            .select_from(tables.Contains)
            .join(
                tables.ResolverClusters,
                and_(
                    tables.ResolverClusters.c.cluster_id == tables.Contains.c.root,
                    tables.ResolverClusters.c.step_id == resolver_id,
                ),
            )
            .subquery(f"resolver_assignments_{resolver_id}")
        )
        # LEFT JOIN so that records not claimed by this resolver are still returned
        from_clause = join(
            from_clause,
            assignments,
            assignments.c.leaf_id == tables.ClusterSourceKey.c.cluster_id,
            isouter=True,
        )
        projected_roots.append(assignments.c.root_id)

    # COALESCE across all resolver root columns
    # First non-null wins, giving higher-priority resolvers precedence
    # Falls back to the source cluster ID when no resolver has claimed the record
    root_projection: ColumnElement[int] = (
        func.coalesce(*projected_roots, tables.ClusterSourceKey.c.cluster_id)
        if projected_roots
        else tables.ClusterSourceKey.c.cluster_id
    )

    selection: list[ColumnElement] = [
        root_projection.label("root_id"),
        tables.ClusterSourceKey.c.cluster_id.label("leaf_id"),
    ]
    if level == "key":
        # "key" level adds the source key, producing more rows than "leaf" because
        # multiple keys can share the same leaf cluster
        selection.append(tables.ClusterSourceKey.c.key)
    if include_source_config_id:
        selection.append(
            tables.ClusterSourceKey.c.source_config_id.label("source_config_id")
        )

    query_stmt = (
        select(*selection)
        .select_from(from_clause)
        .where(tables.ClusterSourceKey.c.source_config_id.in_(source_config_ids))
    )

    # At "leaf" level, deduplicate rows introduced because multiple keys share
    # the same leaf cluster
    if level == "leaf":
        query_stmt = query_stmt.distinct()

    return query_stmt


def build_target_cluster_cte(
    key: str,
    source_config_id: int,
    lineage: list[tuple[int, int | None]],
) -> CTE:
    """Build the target cluster CTE for a source key."""
    # Reuse the unified query at key level, filtered to this source config,
    # so we get the resolved root cluster for the given key
    source_projection = build_unified_query(
        lineage=lineage,
        level="key",
        include_source_config_id=True,
    ).subquery("source_projection")

    return (
        select(source_projection.c.root_id.label("cluster_id"))
        .where(
            and_(
                source_projection.c.source_config_id == source_config_id,
                source_projection.c.key == key,
            )
        )
        # Exactly one cluster per key
        # LIMIT 1 avoids a redundant scan
        .limit(1)
        .cte("target_cluster")
    )


def build_matching_leaves_cte(
    source_and_target_ids: list[int],
    lineage: list[tuple[int, int | None]],
    target_cluster_cte: CTE,
) -> CTE:
    """Build the matching keys CTE for a resolved cluster."""
    # Project all source + target keys through the hierarchy, then filter to those
    # whose resolved root matches the target cluster
    full_projection = build_unified_query(
        lineage=lineage,
        level="key",
        include_source_config_id=True,
    ).subquery("full_projection")

    return (
        select(
            full_projection.c.root_id.label("cluster_id"),
            full_projection.c.source_config_id,
            full_projection.c.key,
        )
        .where(
            and_(
                full_projection.c.source_config_id.in_(source_and_target_ids),
                full_projection.c.root_id == target_cluster_cte.c.cluster_id,
            )
        )
        .distinct()
        .cte("matching_leaves")
    )


def resolver_membership_subquery(
    step_id: int,
    alias: str = "resolver_membership",
) -> Subquery:
    """Build root_id/leaf_id membership rows for a resolver."""
    # First branch: root clusters count as their own leaf (self-membership)
    roots_query = select(
        tables.ResolverClusters.c.cluster_id.label("root_id"),
        tables.ResolverClusters.c.cluster_id.label("leaf_id"),
    ).where(tables.ResolverClusters.c.step_id == step_id)

    # Second branch: all clusters contained within a root via the contains table
    leaves_query = (
        select(
            tables.ResolverClusters.c.cluster_id.label("root_id"),
            tables.Contains.c.leaf.label("leaf_id"),
        )
        .select_from(tables.ResolverClusters)
        .join(
            tables.Contains,
            tables.Contains.c.root == tables.ResolverClusters.c.cluster_id,
        )
        .where(tables.ResolverClusters.c.step_id == step_id)
    )

    # UNION deduplicates in case a root cluster also appears as a leaf
    return roots_query.union(leaves_query).subquery(alias)
