"""Utilities for querying and matching in the PostgreSQL backend."""

from typing import Literal, TypeVar

import pyarrow as pa
from sqlalchemy import (
    CTE,
    ColumnElement,
    FromClause,
    and_,
    func,
    join,
    outerjoin,
    select,
    union_all,
)
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import Match
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
)
from matchbox.common.graph import ResolutionName, SourceResolutionName
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql

T = TypeVar("T")


def get_source_config(name: SourceResolutionName, session: Session) -> SourceConfigs:
    """Converts the named source to a SourceConfigs ORM object."""
    source_config = (
        session.query(SourceConfigs)
        .join(Resolutions, Resolutions.resolution_id == SourceConfigs.resolution_id)
        .filter(Resolutions.name == name)
        .first()
    )
    if source_config is None:
        raise MatchboxResolutionNotFoundError(name=name)

    return source_config


def _build_probability_subquery(
    res_id: int, threshold_val: float, subquery_name: str
) -> Select:
    """Build a probability subquery for a given resolution.

    When a leaf belongs to multiple clusters that meet the threshold,
    this selects the cluster with probability CLOSEST to the threshold
    (i.e., the most conservative/weakest clustering decision).
    """
    # Create unique aliases for this subquery to avoid parameter conflicts
    contains_alias = aliased(Contains, name=f"contains_{subquery_name}")
    probabilities_alias = aliased(Probabilities, name=f"prob_{subquery_name}")

    return (
        select(
            contains_alias.leaf.label("leaf"),
            contains_alias.root.label(f"cluster_{subquery_name}"),
            probabilities_alias.probability,
        )
        .select_from(
            join(
                contains_alias,
                probabilities_alias,
                contains_alias.root == probabilities_alias.cluster_id,
            )
        )
        .where(
            and_(
                probabilities_alias.resolution_id == res_id,
                probabilities_alias.probability >= threshold_val,
            )
        )
        .distinct(contains_alias.leaf)
        .order_by(
            contains_alias.leaf,
            probabilities_alias.probability.asc(),
            contains_alias.root.asc(),
        )
        .subquery(f"prob_{subquery_name}")
    )


def build_unified_query(
    resolution: Resolutions,
    sources: list[SourceConfigs] | None = None,
    threshold: int | None = None,
    level: Literal["leaf", "key"] = "leaf",
    get_hashes: bool = False,
) -> Select:
    """Build a query to resolve cluster assignments across resolution hierarchies.

    This function creates SQL that determines which cluster each source record belongs
    to by traversing up a resolution hierarchy and applying priority-based cluster
    selection.

    The query uses `COALESCE` to implement a priority system where higher-level
    resolutions can "claim" records, with lower levels only processing unclaimed
    records:

    ```sql
    COALESCE(highest_priority_cluster, medium_priority_cluster, ..., source_cluster)
    ```

    1. **Lineage discovery**: Queries the resolution hierarchy to find all ancestor
        resolutions, ordered by priority (lowest level = highest priority)
    2. **Source filtering**: When `sources` is provided, constrains results to only
        include clusters from those specific source configurations
    3. **Threshold application**: Applies probability thresholds to determine which
        clusters qualify at each resolution level
    4. **Subquery construction**: For each model resolution in the lineage, builds
        a subquery that finds qualifying clusters via the Contains→Probabilities
        join. Each joined subquery adds a new cluster column which is then merged
        via...
    5. **`COALESCE` assembly**: Joins all subqueries to source data and uses `COALESCE`
        to select the highest-priority cluster assignment for each record

    The level changes the data returned:

    * `"leaf"`: Returns both root and leaf cluster IDs. For unmerged source
        clusters, the root and leaf properties will be the same.
    * `"key"`: In addition to the above, it also returns the source key. This will give
        more rows than `"leaf"` because it needs a row for every key attached to a leaf.

    Additionally, if `get_hashes` is set to True, the root and leaf hashes are returned.
    """
    # Get ordered lineage (already sorted by priority)
    lineage = resolution.get_lineage(sources=sources, threshold=threshold)

    # Separate to model and source resolutions
    model_resolutions: list[tuple[int, float]] = []
    source_config_ids: list[int] = []
    for resolution_id, source_config_id, truth in lineage:
        if truth is None:
            source_config_ids.append(source_config_id)
        else:
            model_resolutions.append((resolution_id, truth))

    # Build source config filter
    if sources:
        # If sources are provided, filter to those source configs
        source_config_ids = set(sc.source_config_id for sc in sources) & set(
            source_config_ids
        )
        source_filter = ClusterSourceKey.source_config_id.in_(source_config_ids)
    elif resolution.type == "source":
        # If querying a source resolution with no sources filter,
        # filter to just that source
        source_filter = (
            ClusterSourceKey.source_config_id
            == resolution.source_config.source_config_id
        )
    else:
        # No sources provided, filter to lineage source configs
        source_filter = ClusterSourceKey.source_config_id.in_(source_config_ids)

    # Create proper aliases for clusters tables
    leaf_clusters = aliased(Clusters, name="leaf_clusters")
    root_clusters = aliased(Clusters, name="root_clusters")

    # `ClusterSourceKey` is the basis for all subsequent joins
    from_clause: FromClause = ClusterSourceKey

    # Handle source-only case (no model resolutions in lineage)
    if not model_resolutions:
        # We always must select from `ClusterSourceKey`, as it points to source clusters
        selection = [
            ClusterSourceKey.cluster_id.label("root_id"),
            ClusterSourceKey.cluster_id.label("leaf_id"),
        ]

        if level == "key":
            selection.append(
                ClusterSourceKey.key,
            )

        if get_hashes:
            selection += [
                leaf_clusters.cluster_hash.label("root_hash"),
                leaf_clusters.cluster_hash.label("leaf_hash"),
            ]

            from_clause = join(
                from_clause,
                leaf_clusters,
                ClusterSourceKey.cluster_id == leaf_clusters.cluster_id,
            )

    else:  # Querying from a resolution not at the bottom
        # Build subqueries for each model resolution
        # Note both subqueries and cluster_columns are in priority order
        subqueries: list[Select] = []
        cluster_columns: list[ColumnElement] = []

        for i, (res_id, threshold_val) in enumerate(model_resolutions):
            subquery = _build_probability_subquery(res_id, threshold_val, str(i))
            subqueries.append(subquery)
            cluster_columns.append(subquery.c[f"cluster_{i}"])

        # To get hashes we need to join `Clusters`, here for leaves
        # and later for roots
        if get_hashes:
            from_clause: FromClause = join(
                from_clause,
                leaf_clusters,
                ClusterSourceKey.cluster_id == leaf_clusters.cluster_id,
            )

        # Add LEFT JOINs for each model resolution subquery
        for subquery in subqueries:
            from_clause = outerjoin(
                from_clause,
                subquery,
                ClusterSourceKey.cluster_id == subquery.c.leaf,
            )

        # Build final cluster ID
        # First non-null wins (highest priority first)
        final_root_cluster_id = func.coalesce(
            *cluster_columns, ClusterSourceKey.cluster_id
        )

        selection = [
            final_root_cluster_id.label("root_id"),
            ClusterSourceKey.cluster_id.label("leaf_id"),
        ]

        if level == "key":
            selection.append(
                ClusterSourceKey.key,
            )

        if get_hashes:
            selection += [
                root_clusters.cluster_hash.label("root_hash"),
                leaf_clusters.cluster_hash.label("leaf_hash"),
            ]

            from_clause: FromClause = from_clause.outerjoin(
                root_clusters, final_root_cluster_id == root_clusters.cluster_id
            )

    query = select(*selection).select_from(from_clause).where(source_filter)

    # Because we start from `ClusterSourceKey`, we must remove duplicates caused
    # by distinct keys on the same leaf
    if level == "leaf":
        query = query.distinct()

    return query


def _build_target_cluster_cte(
    key: str,
    source_config_id: int,
    resolution: Resolutions,
    threshold: int | None,
) -> Select:
    """Build the target_cluster CTE.

    Follows very similar logic to `build_unified_query`, but with filtering
    specifically for a single key and source_config_id.
    """
    # Get ordered lineage
    lineage = resolution.get_lineage(threshold=threshold)
    model_resolutions = [
        (res_id, truth) for res_id, _, truth in lineage if truth is not None
    ]

    # Build subqueries for each model resolution
    # Note both subqueries and cluster_columns are in priority order
    subqueries = []
    cluster_columns = []
    for i, (res_id, threshold_val) in enumerate(model_resolutions):
        subquery = _build_probability_subquery(res_id, threshold_val, str(i))
        subqueries.append(subquery)
        cluster_columns.append(subquery.c[f"cluster_{i}"])

    # Build FROM clause starting with cluster_keys
    from_clause = ClusterSourceKey

    # Add LEFT JOINs for each resolution subquery
    for subquery in subqueries:
        from_clause = outerjoin(
            from_clause,
            subquery,
            ClusterSourceKey.cluster_id == subquery.c.leaf,
        )

    # Build final cluster ID using COALESCE - first non-null wins
    final_cluster_id = func.coalesce(*cluster_columns, ClusterSourceKey.cluster_id)

    return (
        select(final_cluster_id.label("cluster_id"))
        .select_from(from_clause)
        .where(
            and_(
                ClusterSourceKey.key == key,
                ClusterSourceKey.source_config_id == source_config_id,
            )
        )
    )


def _build_matching_leaves_cte(
    target_source_config_ids: list[int],
    resolution: Resolutions,
    threshold: int | None,
    target_cluster_cte: CTE,
) -> Select:
    """Find all keys from target sources that belong to the target cluster.

    Given a target cluster ID, find all keys from my target sources that belong to
    it through ANY path in the hierarchy.

    Uses `UNION ALL` to combine multiple ways a key can belong to the target cluster:

        1. **Direct membership**: Keys that directly belong to the target cluster ID
        2. **Hierarchy membership**: For each model resolution in the lineage, keys that
            are connected to the target cluster through Contains→Probabilities chains

    The target cluster ID comes from the target_cluster_cte, and we search for all
    keys from the specified target sources that are related to it through any path
    in the resolution hierarchy.

    Returns a union of all matching keys with their cluster_id, key, and
    source_config_id.
    """
    # Get ordered lineage and extract model resolutions
    lineage = resolution.get_lineage(threshold=threshold)
    model_resolutions = [
        (res_id, truth) for res_id, _, truth in lineage if truth is not None
    ]

    # Start with direct members branch
    branches = [
        select(
            ClusterSourceKey.cluster_id,
            ClusterSourceKey.key,
            ClusterSourceKey.source_config_id,
        )
        .select_from(
            ClusterSourceKey, target_cluster_cte
        )  # CROSS JOIN with target_cluster
        .where(
            and_(
                ClusterSourceKey.source_config_id.in_(target_source_config_ids),
                ClusterSourceKey.cluster_id == target_cluster_cte.c.cluster_id,
            )
        )
    ]

    # Add branches for each model resolution
    for res_id, threshold_val in model_resolutions:
        branch = (
            select(
                ClusterSourceKey.cluster_id,
                ClusterSourceKey.key,
                ClusterSourceKey.source_config_id,
            )
            .select_from(
                join(
                    join(
                        ClusterSourceKey,
                        Contains,
                        ClusterSourceKey.cluster_id == Contains.leaf,
                    ),
                    Probabilities,
                    and_(
                        Contains.root == Probabilities.cluster_id,
                        Probabilities.resolution_id == res_id,
                        Probabilities.probability >= threshold_val,
                    ),
                ),
                target_cluster_cte,  # CROSS JOIN with target_cluster
            )
            .where(
                and_(
                    ClusterSourceKey.source_config_id.in_(target_source_config_ids),
                    Probabilities.cluster_id == target_cluster_cte.c.cluster_id,
                )
            )
        )
        branches.append(branch)

    # Combine all branches with UNION ALL
    if len(branches) == 1:
        return branches[0]

    return union_all(*branches)


def _build_match_query(
    key: str,
    source_config_id: int,
    target_source_config_ids: list[int],
    resolution: Resolutions,
    threshold: int | None,
) -> Select:
    """Build a match query to find all keys that cluster with a given input key.

    This function creates SQL that identifies which cluster an input key belongs to,
    then finds all other keys from specified target sources that belong to the same
    cluster through the resolution hierarchy.

    The query uses two CTEs to solve the matching problem:

        1. **Target cluster identification**: Determines which cluster the input key
            belongs to at the specified resolution level
        2. **Matching leaves discovery**: Finds all keys from target sources that
            belong to the same target cluster

    The overall process:

        1. **Target cluster CTE**: Uses the same `COALESCE` hierarchy logic as
            `build_unified_query` to resolve which cluster the input key belongs to.
            This handles the full resolution hierarchy with proper priority ordering.
        2. **Matching leaves CTE**: Builds a `UNION ALL` query with multiple branches:

            - **Direct members**: Keys that belong directly to the target cluster
            - **Hierarchy branches**: For each model resolution, finds keys that are
                connected to the target cluster through the Contains→Probabilities joins

        3. **Final assembly**: `LEFT JOINs` the target cluster with matching leaves,
            ensuring we always get the cluster ID even if no target matches exist
    """
    target_cluster_cte = _build_target_cluster_cte(
        key=key,
        source_config_id=source_config_id,
        resolution=resolution,
        threshold=threshold,
    ).cte("target_cluster")

    # Include both source and target configs to get all keys in the cluster
    all_source_config_ids = [source_config_id] + target_source_config_ids
    matching_leaves_cte = _build_matching_leaves_cte(
        target_source_config_ids=all_source_config_ids,
        resolution=resolution,
        threshold=threshold,
        target_cluster_cte=target_cluster_cte,
    ).cte("matching_leaves")

    return (
        select(
            matching_leaves_cte.c.cluster_id.label("cluster"),
            matching_leaves_cte.c.source_config_id,
            matching_leaves_cte.c.key,
        )
        .select_from(matching_leaves_cte)
        .distinct()
    )


def query(
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    return_leaf_id: bool = False,
    limit: int = None,
) -> pa.Table:
    """Queries Matchbox to retrieve linked data for a source.

    Retrieves all linked data for a given source, resolving through hierarchy if needed.

    * Simple case: If querying the same resolution as the source, just select cluster
        IDs and keys directly from ClusterSourceKey
    * Hierarchy case: Uses the unified query builder to traverse up the resolution
        hierarchy, applying COALESCE priority logic to determine which parent cluster
        each source record belongs to
    * Priority resolution: When multiple model resolutions could assign a record to
        different clusters, COALESCE ensures higher-priority resolutions win

    Returns all records with their final resolved cluster IDs.
    """
    with MBDB.get_session() as session:
        source_config: SourceConfigs = get_source_config(source, session)
        source_resolution: Resolutions = session.get(
            Resolutions, source_config.resolution_id
        )

        if resolution:
            truth_resolution: Resolutions = (
                session.query(Resolutions)
                .filter(Resolutions.name == resolution)
                .first()
            )
            if truth_resolution is None:
                raise MatchboxResolutionNotFoundError(name=resolution)
        else:
            truth_resolution: Resolutions = source_resolution

        id_query: Select = build_unified_query(
            resolution=truth_resolution,
            sources=[source_config],
            threshold=threshold,
            level="key",
        )

        if limit:
            id_query = id_query.limit(limit)

        with MBDB.get_adbc_connection() as conn:
            stmt: str = compile_sql(id_query)
            logger.debug(f"Query SQL: \n {stmt}")
            id_results = sql_to_df(
                stmt=stmt, connection=conn.dbapi_connection, return_type="arrow"
            ).rename_columns({"root_id": "id"})

        selection = ["id", "key"]
        if return_leaf_id:
            selection.append("leaf_id")

        return id_results.select(selection)


def get_parent_clusters_and_leaves(
    resolution: Resolutions,
) -> dict[int, dict[str, list[dict]]]:
    """Query clusters and their leaves for all parent resolutions.

    For a given resolution, find all its parent resolutions and return complete
    cluster compositions.

    * Parent discovery: Queries ResolutionFrom to find all direct parent
        resolutions (level 1)
    * Cluster building: For each parent, runs the full unified query to get all
        cluster assignments with both root and leaf information
    * Aggregation: Collects all leaf nodes belonging to each root cluster across all
        parent resolutions

    Return a dictionary mapping cluster IDs to their complete leaf compositions
    and metadata.
    """
    with MBDB.get_session() as session:
        # Get direct parent resolution IDs
        parent_ids: list[int] = [
            row[0]
            for row in session.execute(
                select(ResolutionFrom.parent)
                .where(ResolutionFrom.child == resolution.resolution_id)
                .where(ResolutionFrom.level == 1)
            ).all()
        ]

        if not parent_ids:
            return {}

        # For each parent, get all cluster assignments
        all_clusters: dict[int, dict] = {}

        for parent_id in parent_ids:
            parent_resolution: Resolutions = session.get(Resolutions, parent_id)
            if parent_resolution is None:
                continue

            parent_assignments: Select = build_unified_query(
                resolution=parent_resolution, get_hashes=True
            )

            for row in session.execute(parent_assignments):
                root_id = row.root_id
                if root_id not in all_clusters:
                    all_clusters[root_id] = {
                        "root_hash": row.root_hash,
                        "leaves": [],
                        "probability": None,
                    }
                all_clusters[root_id]["leaves"].append(
                    {"leaf_id": row.leaf_id, "leaf_hash": row.leaf_hash}
                )

        return all_clusters


def match(
    key: str,
    source: SourceResolutionName,
    targets: list[SourceResolutionName],
    resolution: ResolutionName,
    threshold: int | None = None,
) -> list[Match]:
    """Matches an ID in a source resolution and returns the keys in the targets.

    Given a specific key in a source, find what it matches to in target sources
    through a resolution hierarchy.

    * Target cluster identification: Uses COALESCE priority CTE to determine which
        cluster the input key belongs to at the resolution level
    * Matching leaves discovery: Builds UNION ALL query with branches for:
        * Direct cluster members (source-only case)
        * Members connected through each model resolution in the hierarchy
    * Cross-reference: Joins the target cluster with all possible matching leaves,
        filtering for the requested target sources

    Organises matches by source configuration and returns structured Match objects
    for each target.
    """
    with MBDB.get_session() as session:
        # Get configurations
        source_config: SourceConfigs = get_source_config(source, session)
        truth_resolution: Resolutions | None = session.execute(
            select(Resolutions).where(Resolutions.name == resolution)
        ).scalar()
        if truth_resolution is None:
            raise MatchboxResolutionNotFoundError(name=resolution)

        target_configs: list[SourceConfigs] = [
            get_source_config(target, session) for target in targets
        ]
        target_source_config_ids = [tc.source_config_id for tc in target_configs]

        # Build and execute the match query
        matches_query = _build_match_query(
            key=key,
            source_config_id=source_config.source_config_id,
            target_source_config_ids=target_source_config_ids,
            resolution=truth_resolution,
            threshold=threshold,
        )

        logger.debug(f"Match SQL: \n {compile_sql(matches_query)}")
        matches = session.execute(matches_query).all()

        # Organise matches by source config
        cluster: int | None = None
        matches_by_source_id: dict[int, set] = {}

        for cluster_id, source_config_id_result, key_in_source in matches:
            if cluster is None:
                cluster = cluster_id

            # Skip NULL results from LEFT JOIN (when no target matches)
            if source_config_id_result is not None and key_in_source is not None:
                if source_config_id_result not in matches_by_source_id:
                    matches_by_source_id[source_config_id_result] = set()
                matches_by_source_id[source_config_id_result].add(key_in_source)

        # Build result objects
        result: list[Match] = []
        for target, target_config in zip(targets, target_configs, strict=False):
            match_obj = Match(
                cluster=cluster,
                source=source,
                source_id=matches_by_source_id.get(
                    source_config.source_config_id, set()
                ),
                target=target,
                target_id=matches_by_source_id.get(
                    target_config.source_config_id, set()
                ),
            )
            result.append(match_obj)

        return result
