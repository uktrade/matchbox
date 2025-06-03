"""Utilities for querying and matching in the PostgreSQL backend."""

from typing import Any, TypeVar

import pyarrow as pa
from sqlalchemy import and_, case, func, literal, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ResolutionName, SourceResolutionName
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.common.sources import Match
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
        raise MatchboxSourceNotFoundError(name=name)

    return source_config


def _resolve_thresholds(
    lineage_truths: dict[int, float | None],
    resolution: Resolutions,
    threshold: int | None,
) -> dict[int, float | None]:
    """Resolves final thresholds for each resolution in the lineage based on user input.

    Args:
        lineage_truths: Dict from resolution_id -> cached truth
        resolution: The target resolution being used for clustering
        threshold: User-supplied threshold value

    Returns:
        Dict mapping resolution_id to their final threshold values
    """
    resolved_thresholds = {}

    for resolution_id, default_truth in lineage_truths.items():
        # Source
        if default_truth is None:
            resolved_thresholds[resolution_id] = None
            continue

        # Model
        if threshold is None:
            resolved_thresholds[resolution_id] = default_truth
        elif isinstance(threshold, int):
            resolved_thresholds[resolution_id] = (
                threshold
                if resolution_id == resolution.resolution_id
                else default_truth
            )
        else:
            raise ValueError(f"Invalid threshold type: {type(threshold)}")

    return resolved_thresholds


def _resolve_hierarchy_assignments(
    resolution: Resolutions,
    sources: list[Resolutions] | None = None,
    threshold: int | None = None,
) -> select:
    """Resolve cluster assignments across a resolution's lineage.

    Simplified version that leverages ORM relationships and existing methods.
    """
    # Step 1: Get lineage using existing method
    if sources:
        # Get lineage to specific sources only
        lineage_truths = {}
        for source in sources:
            try:
                source_lineage = resolution.get_lineage_to_source(source=source)
                lineage_truths.update(source_lineage)
            except ValueError:
                continue

        if not lineage_truths:
            return _empty_result()
    else:
        # Get full lineage
        lineage_truths = resolution.get_lineage()

    # Step 2: Build source config filter if needed
    source_config_filter = True
    if sources:
        source_config_ids = []
        for source in sources:
            if source.source_config:
                source_config_ids.append(source.source_config.source_config_id)

        if source_config_ids:
            source_config_filter = ClusterSourceKey.source_config_id.in_(
                source_config_ids
            )

    # Step 3: Resolve thresholds
    resolved_thresholds = _resolve_thresholds(lineage_truths, resolution, threshold)

    # Step 4: Build the unified query
    return _build_unified_query(resolved_thresholds, source_config_filter, resolution)


def _build_source_query(
    source_conditions: list[tuple[Any, int, int]], source_config_filter: Any
) -> select:
    """Build query for source resolutions (no hierarchy traversal)."""
    source_priority_case = case(
        *[(condition, priority) for condition, priority, _ in source_conditions],
        else_=9999,
    ).label("priority")

    source_deciding_case = case(
        *[(condition, res_id) for condition, _, res_id in source_conditions], else_=None
    ).label("deciding_resolution_id")

    return (
        select(
            ClusterSourceKey.key.label("leaf_key"),
            Clusters.cluster_id.label("leaf_id"),
            Clusters.cluster_hash.label("leaf_hash"),
            Clusters.cluster_id.label("root_id"),  # No hierarchy for sources
            Clusters.cluster_hash.label("root_hash"),  # No hierarchy for sources
            ClusterSourceKey.source_config_id,
            source_priority_case,
            source_deciding_case,
        )
        .select_from(ClusterSourceKey)
        .join(Clusters, ClusterSourceKey.cluster_id == Clusters.cluster_id)
        .join(
            SourceConfigs,
            SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
        )
        .where(
            and_(
                or_(*[condition for condition, _, _ in source_conditions]),
                source_config_filter,
            )
        )
    )


def _build_model_query(
    model_conditions: list[tuple[Any, int, int]], source_config_filter: Any
) -> select:
    """Build query for model resolutions (with hierarchy traversal)."""
    RootClusters = Clusters.__table__.alias("root_clusters")

    model_priority_case = case(
        *[(condition, priority) for condition, priority, _ in model_conditions],
        else_=9999,
    ).label("priority")

    model_deciding_case = case(
        *[(condition, res_id) for condition, _, res_id in model_conditions], else_=None
    ).label("deciding_resolution_id")

    return (
        select(
            ClusterSourceKey.key.label("leaf_key"),
            Clusters.cluster_id.label("leaf_id"),
            Clusters.cluster_hash.label("leaf_hash"),
            func.coalesce(Contains.root, Clusters.cluster_id).label("root_id"),
            func.coalesce(RootClusters.c.cluster_hash, Clusters.cluster_hash).label(
                "root_hash"
            ),
            ClusterSourceKey.source_config_id,
            model_priority_case,
            model_deciding_case,
        )
        .select_from(ClusterSourceKey)
        .join(Clusters, ClusterSourceKey.cluster_id == Clusters.cluster_id)
        .join(
            SourceConfigs,
            SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
        )
        .outerjoin(Contains, Contains.leaf == Clusters.cluster_id)
        .outerjoin(RootClusters, Contains.root == RootClusters.c.cluster_id)
        .outerjoin(
            Probabilities,
            Probabilities.cluster == func.coalesce(Contains.root, Clusters.cluster_id),
        )
        .where(
            and_(
                or_(*[condition for condition, _, _ in model_conditions]),
                source_config_filter,
            )
        )
    )


def _build_unified_query(
    resolved_thresholds: dict[int, float | None],
    source_config_filter: Any,
    resolution: Resolutions,
) -> select:
    """Build a single unified query instead of separate source/model queries."""
    # Separate source and model resolutions
    source_conditions = []
    model_conditions = []

    for res_id, threshold_val in resolved_thresholds.items():
        if threshold_val is None:
            # Source resolution - direct cluster assignment, no hierarchy
            priority = 999
            condition = SourceConfigs.resolution_id == res_id
            source_conditions.append((condition, priority, res_id))
        else:
            # Model resolution - probability-based assignment with hierarchy
            priority = _get_resolution_priority(res_id, resolution)
            prob_conditions = [
                Probabilities.resolution == res_id,
                Probabilities.role_flag >= 1,
                Probabilities.probability >= threshold_val,
            ]
            condition = and_(*prob_conditions)
            model_conditions.append((condition, priority, res_id))

    # Build separate queries WITHOUT ranking
    queries = []

    if source_conditions:
        queries.append(_build_source_query(source_conditions, source_config_filter))

    if model_conditions:
        queries.append(_build_model_query(model_conditions, source_config_filter))

    if not queries:
        return _empty_result()

    # Union all queries
    if len(queries) == 1:
        combined = queries[0]
    else:
        combined = queries[0]
        for query in queries[1:]:
            combined = combined.union_all(query)

    all_decisions = combined.cte("all_decisions")

    # Apply ranking AFTER union
    ranked = select(
        all_decisions,
        func.row_number()
        .over(
            partition_by=all_decisions.c.leaf_key,
            order_by=[
                all_decisions.c.priority.asc(),
                all_decisions.c.deciding_resolution_id.desc(),
            ],
        )
        .label("rank"),
    ).cte("ranked_decisions")

    # Return only rank=1 per key
    return select(
        ranked.c.root_id,
        ranked.c.root_hash,
        ranked.c.leaf_id,
        ranked.c.leaf_hash,
        ranked.c.leaf_key,
        ranked.c.source_config_id,
    ).where(ranked.c.rank == 1)


def _get_resolution_priority(
    resolution_id: int, context_resolution: Resolutions
) -> int:
    """Get priority level for a resolution (lower = higher priority)."""
    with MBDB.get_session() as session:
        priority_query = select(ResolutionFrom.level).where(
            and_(
                ResolutionFrom.parent == resolution_id,
                ResolutionFrom.child == context_resolution.resolution_id,
            )
        )
        level = session.execute(priority_query).scalar()
        return level if level is not None else 0


def _empty_result() -> select:
    """Return empty result with correct column structure."""
    return select(
        literal(None).label("root_id"),
        literal(None).label("root_hash"),
        literal(None).label("leaf_id"),
        literal(None).label("leaf_hash"),
        literal(None).label("leaf_key"),
        literal(None).label("source_config_id"),
    ).where(False)


def _resolve_cluster_hierarchy(
    source_config: SourceConfigs,
    truth_resolution: Resolutions,
    threshold: int | None = None,
) -> Select:
    """Resolves the final cluster assignments for all records in a source.

    Args:
        source_config: SourceConfig object of the source to query
        truth_resolution: Resolution object representing the point of truth
        threshold: Optional threshold value

    Returns:
        SQLAlchemy Select statement that will resolve to (id, key) pairs, where
        id is the highest-priority cluster id and key is the original record key
    """
    with MBDB.get_session() as session:
        source_resolution = session.get(Resolutions, source_config.resolution_id)
        if source_resolution is None:
            raise MatchboxSourceNotFoundError()

        # If truth_resolution is the same as source resolution,
        # just return clusters directly
        if truth_resolution.resolution_id == source_config.resolution_id:
            return (
                select(
                    ClusterSourceKey.cluster_id.label("id"),
                    ClusterSourceKey.key.label("key"),
                )
                .select_from(ClusterSourceKey)
                .where(
                    ClusterSourceKey.source_config_id == source_config.source_config_id
                )
            )

        # Use the common abstraction to resolve hierarchy assignments
        hierarchy_assignments = _resolve_hierarchy_assignments(
            resolution=truth_resolution,
            sources=[source_resolution],
            threshold=threshold,
        ).cte("hierarchy_assignments")

        # Filter to our specific source config and return in the expected format
        return select(
            hierarchy_assignments.c.root_id.label("id"),
            hierarchy_assignments.c.leaf_key.label("key"),
        ).where(
            hierarchy_assignments.c.source_config_id == source_config.source_config_id
        )


def get_clusters_with_leaves(
    resolution: Resolutions,
) -> dict[int, dict[str, list[dict]]]:
    """Query clusters and their leaves for all parent resolutions.

    Args:
        resolution: Resolution object whose parents proposed the clusters
            we need to recover

    Returns:
        Dict mapping cluster_id to a dict with cluster info and leaves list
    """
    with MBDB.get_session() as session:
        # Get parent resolution IDs
        parent_ids = [
            row[0]
            for row in session.execute(
                select(ResolutionFrom.parent)
                .where(ResolutionFrom.child == resolution.resolution_id)
                .where(ResolutionFrom.level == 1)
            ).all()
        ]

        if not parent_ids:
            return {}

        # For each parent, get all cluster assignments it endorses
        all_clusters = {}
        cluster_leaves: dict[int, set[tuple[int, bytes]]] = {}

        for parent_id in parent_ids:
            parent_resolution = session.get(Resolutions, parent_id)
            if parent_resolution is None:
                continue

            # Use the common abstraction to get all assignments this parent endorses
            parent_assignments = _resolve_hierarchy_assignments(
                resolution=parent_resolution,
                sources=None,  # All sources in parent's lineage
                threshold=None,  # Use default thresholds
            )

            # Execute and collect results for this parent
            for row in session.execute(parent_assignments):
                root_id = row.root_id

                if root_id not in cluster_leaves:
                    cluster_leaves[root_id] = set()
                    all_clusters[root_id] = {
                        "root_hash": row.root_hash,
                        "leaves": [],
                        "probability": None,
                    }

                cluster_leaves[root_id].add((row.leaf_id, row.leaf_hash))

        # Convert tuple sets to dict lists
        for cluster_id, leaf_tuples in cluster_leaves.items():
            all_clusters[cluster_id]["leaves"] = [
                {"leaf_id": leaf_id, "leaf_hash": leaf_hash}
                for leaf_id, leaf_hash in leaf_tuples
            ]

        return all_clusters


def query(
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int = None,
) -> pa.Table:
    """Queries Matchbox and the SourceConfig warehouse to retrieve linked data.

    Takes the dictionaries of tables and fields outputted by selectors and
    queries database for them. If a "point of truth" resolution is supplied, will
    attach the clusters this data belongs to.

    To accomplish this, the function:

    * Iterates through each selector, and
        * Retrieves its data in Matchbox according to the optional point of truth,
        including its hash and cluster hash
        * Retrieves its raw data from its SourceConfig's warehouse
        * Joins the two together
    * Unions the results, one row per item of data in the warehouses

    Returns:
        A table containing the requested data from each table, unioned together,
        with the hash key of each row in Matchbox
    """
    with MBDB.get_session() as session:
        source_config = get_source_config(source, session)
        source_resolution = session.get(Resolutions, source_config.resolution_id)

        if resolution:
            truth_resolution = (
                session.query(Resolutions)
                .filter(Resolutions.name == resolution)
                .first()
            )
            if truth_resolution is None:
                raise MatchboxResolutionNotFoundError(name=resolution)
        else:
            truth_resolution = source_resolution

        id_query = _resolve_cluster_hierarchy(
            source_config=source_config,
            truth_resolution=truth_resolution,
            threshold=threshold,
        )

        if limit:
            id_query = id_query.limit(limit)

        with MBDB.get_adbc_connection() as conn:
            stmt = compile_sql(id_query)
            logger.debug(f"Query SQL: \n {stmt}")

            mb_ids = sql_to_df(
                stmt=stmt,
                connection=conn,
                return_type="arrow",
            )

        return mb_ids


def match(
    key: str,
    source: SourceResolutionName,
    targets: list[SourceResolutionName],
    resolution: ResolutionName,
    threshold: int | None = None,
) -> list[Match]:
    """Matches an ID in a source resolution and returns the keys in the targets.

    To accomplish this, the function:
    * Uses the resolution lineage to determine what cluster the source key belongs to
    * Finds all other keys in that same cluster across all target sources
    * Returns the results as Match objects, one per target
    """
    with MBDB.get_session() as session:
        # Get source config and truth resolution
        source_config = get_source_config(source, session)
        truth_resolution = (
            session.query(Resolutions).filter(Resolutions.name == resolution).first()
        )
        if truth_resolution is None:
            raise MatchboxResolutionNotFoundError(name=resolution)

        # Get all target source configs
        target_configs = []
        for target in targets:
            target_config = get_source_config(target, session)
            target_configs.append(target_config)

        # Use our hierarchy resolution to get all cluster assignments
        # This gives us what cluster each key belongs to
        # according to the truth resolution
        hierarchy_assignments = _resolve_hierarchy_assignments(
            resolution=truth_resolution,
            sources=None,  # Get assignments for all sources
            threshold=threshold,
        ).cte("hierarchy_assignments")

        # Find what cluster our source key belongs to
        source_key_cluster = (
            select(hierarchy_assignments.c.root_id)
            .where(
                and_(
                    hierarchy_assignments.c.leaf_key == key,
                    hierarchy_assignments.c.source_config_id
                    == source_config.source_config_id,
                )
            )
            .scalar_subquery()
        )

        # Find all keys in the same cluster across all sources
        cluster_matches = select(
            hierarchy_assignments.c.root_id.label("cluster"),
            hierarchy_assignments.c.source_config_id.label("source_config"),
            hierarchy_assignments.c.leaf_key.label("key"),
        ).where(hierarchy_assignments.c.root_id == source_key_cluster)

        logger.debug(f"Match SQL: \n {compile_sql(cluster_matches)}")

        matches = session.execute(cluster_matches).all()

        # Organize matches by source config
        cluster = None
        matches_by_source_id: dict[int, set] = {}
        for cluster_id, source_config_id, key_in_source in matches:
            if cluster is None:
                cluster = cluster_id
            if source_config_id not in matches_by_source_id:
                matches_by_source_id[source_config_id] = set()
            matches_by_source_id[source_config_id].add(key_in_source)

        # Build result objects for each target
        result = []
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
