"""Utilities for querying and matching in the PostgreSQL backend."""

from typing import TypeVar

import pyarrow as pa
from sqlalchemy import BIGINT, and_, cast, func, literal, null, select, union
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import CTE, Select

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    ModelResolutionName,
    ResolutionName,
    SourceResolutionName,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.common.sources import Match
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceIdentifiers,
    Contains,
    Probabilities,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql

T = TypeVar("T")


def _get_dataset_source(
    source: SourceResolutionName, session: Session
) -> SourceConfigs:
    """Converts the named address of source to a SourceConfigs ORM object."""
    source = (
        session.query(SourceConfigs)
        .join(Resolutions, Resolutions.resolution_id == SourceConfigs.resolution_id)
        .where(Resolutions.name == source)
        .first()
    )
    if source is None:
        raise MatchboxSourceNotFoundError(name=source)

    return source


def _resolve_thresholds(
    lineage_truths: dict[str, float],
    resolution: Resolutions,
    threshold: int | None,
) -> dict[int, float]:
    """Resolves final thresholds for each resolution in the lineage based on user input.

    Args:
        lineage_truths: Dict from with resolution hash -> cached truth
        resolution: The target resolution being used for clustering
        threshold: User-supplied threshold value

    Returns:
        Dict mapping resolution hash to their final threshold values
    """
    resolved_thresholds = {}

    for resolution_id, default_truth in lineage_truths.items():
        # Dataset
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


def _get_valid_clusters_for_resolution(resolution_id: int, threshold: int) -> Select:
    """Get clusters that meet the threshold for a specific resolution."""
    return select(Probabilities.cluster.label("cluster")).where(
        and_(
            Probabilities.resolution == resolution_id,
            Probabilities.probability >= threshold,
        )
    )


def _union_valid_clusters(lineage_thresholds: dict[int, float]) -> Select:
    """Creates a CTE of clusters that are valid for any resolution in the lineage.

    Each resolution may have a different threshold.
    """
    valid_clusters = None

    for resolution_id, threshold in lineage_thresholds.items():
        if threshold is None:
            # This is a dataset - get all its clusters through ClusterSourceIdentifiers
            resolution_valid = (
                select(ClusterSourceIdentifiers.cluster_id.label("cluster"))
                .join(
                    SourceConfigs,
                    SourceConfigs.source_id == ClusterSourceIdentifiers.source_id,
                )
                .where(SourceConfigs.resolution_id == resolution_id)
                .distinct()
            )
        else:
            # This is a model - get clusters meeting threshold
            resolution_valid = _get_valid_clusters_for_resolution(
                resolution_id, threshold
            )

        if valid_clusters is None:
            valid_clusters = resolution_valid
        else:
            valid_clusters = union(valid_clusters, resolution_valid)

    if valid_clusters is None:
        # Handle empty lineage case
        return select(cast(null(), BIGINT).label("cluster")).where(False)

    return valid_clusters.cte("valid_clusters").prefix_with("MATERIALIZED")


def _build_valid_contains(valid_clusters_cte: CTE, name: str) -> CTE:
    """Filters the Contains table to only include relationships with valid parents."""
    valid_contains = select(Contains.child, Contains.parent).where(
        Contains.parent.in_(select(valid_clusters_cte.c.cluster))
    )

    return valid_contains.cte(name).prefix_with("MATERIALIZED")


def _resolve_cluster_hierarchy(
    dataset_source: SourceConfigs,
    truth_resolution: Resolutions,
    threshold: int | None = None,
) -> Select:
    """Resolves the final cluster assignments for all records in a dataset.

    Args:
        dataset_source: SourceConfig object of the dataset to query
        truth_resolution: Resolution object representing the point of truth
        threshold: Optional threshold value

    Returns:
        SQLAlchemy Select statement that will resolve to (hash, id) pairs, where
        hash is the ultimate parent cluster hash and id is the original record ID
    """
    with MBDB.get_session() as session:
        dataset_resolution = session.get(Resolutions, dataset_source.resolution_id)
        if dataset_resolution is None:
            raise MatchboxSourceNotFoundError()

        try:
            lineage_truths = truth_resolution.get_lineage_to_dataset(
                dataset=dataset_resolution
            )
        except ValueError as e:
            raise MatchboxResolutionNotFoundError(
                f"Invalid resolution lineage: {str(e)}"
            ) from e

        thresholds = _resolve_thresholds(
            lineage_truths=lineage_truths,
            resolution=truth_resolution,
            threshold=threshold,
        )

        # Get clusters and contains valid across all resolutions in lineage
        valid_clusters = _union_valid_clusters(thresholds)
        valid_contains = _build_valid_contains(valid_clusters, name="valid_contains")

        # Get base mapping of IDs to clusters
        mapping_base = (
            select(
                Clusters.cluster_id.label("cluster_id"),
                ClusterSourceIdentifiers.source_identifier.label("source_identifier"),
            )
            .join(
                ClusterSourceIdentifiers,
                ClusterSourceIdentifiers.cluster_id == Clusters.cluster_id,
            )
            .where(
                and_(
                    Clusters.cluster_id.in_(select(valid_clusters.c.cluster)),
                    ClusterSourceIdentifiers.source_id == dataset_source.source_id,
                )
            )
            .cte("mapping_base")
            .prefix_with("MATERIALIZED")
        )

        # Build recursive hierarchy CTE
        hierarchy = (
            # Base case: direct parents
            select(
                mapping_base.c.cluster_id.label("original_cluster"),
                mapping_base.c.cluster_id.label("child"),
                valid_contains.c.parent.label("parent"),
                literal(1).label("level"),
            )
            .select_from(mapping_base)
            .join(
                valid_contains,
                valid_contains.c.child == mapping_base.c.cluster_id,
                isouter=True,
            )
            .cte("hierarchy", recursive=True)
        )

        # Recursive case
        recursive = (
            select(
                hierarchy.c.original_cluster,
                hierarchy.c.parent.label("child"),
                valid_contains.c.parent.label("parent"),
                (hierarchy.c.level + 1).label("level"),
            )
            .select_from(hierarchy)
            .join(valid_contains, valid_contains.c.child == hierarchy.c.parent)
            .where(hierarchy.c.parent.is_not(None))  # Only recurse on non-leaf nodes
        )

        hierarchy = hierarchy.union_all(recursive)

        # Get highest parents
        highest_parents = (
            select(
                hierarchy.c.original_cluster,
                hierarchy.c.parent.label("highest_parent"),
                hierarchy.c.level,
            )
            .distinct(hierarchy.c.original_cluster)
            .order_by(hierarchy.c.original_cluster, hierarchy.c.level.desc())
            .cte("highest_parents")
        )

        # Final mapping with coalesced results
        final_mapping = (
            select(
                mapping_base.c.source_identifier,
                func.coalesce(
                    highest_parents.c.highest_parent, mapping_base.c.cluster_id
                ).label("final_parent"),
            )
            .select_from(mapping_base)
            .join(
                highest_parents,
                highest_parents.c.original_cluster == mapping_base.c.cluster_id,
                isouter=True,
            )
            .cte("final_mapping")
        )

        # Final select statement
        return select(
            final_mapping.c.final_parent.label("id"), final_mapping.c.source_identifier
        )


def query(
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int = None,
) -> pa.Table:
    """Create a source identifier to Matchbox ID lookup for one source resolution."""
    with MBDB.get_session() as session:
        dataset_source = _get_dataset_source(source, session)
        dataset_resolution = session.get(Resolutions, dataset_source.resolution_id)

        if resolution:
            truth_resolution = (
                session.query(Resolutions)
                .filter(Resolutions.name == resolution)
                .first()
            )
            if truth_resolution is None:
                raise MatchboxResolutionNotFoundError(name=resolution)
        else:
            truth_resolution = dataset_resolution

        id_query = _resolve_cluster_hierarchy(
            dataset_source=dataset_source,
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


def _build_unnested_clusters() -> CTE:
    """Create CTE that unnests cluster IDs for easier joining."""
    return (
        select(
            Clusters.cluster_id,
            ClusterSourceIdentifiers.source_id.label("dataset"),
            ClusterSourceIdentifiers.identifier,
        )
        .select_from(Clusters)
        .join(
            ClusterSourceIdentifiers,
            ClusterSourceIdentifiers.cluster_id == Clusters.cluster_id,
        )
        .cte("unnested_clusters")
        .prefix_with("MATERIALIZED")
    )


def _find_source_cluster(
    unnested_clusters: CTE, source_dataset_id: int, identifier: str
) -> Select:
    """Find the initial cluster containing the source primary key."""
    return (
        select(unnested_clusters.c.cluster_id)
        .select_from(unnested_clusters)
        .where(
            and_(
                unnested_clusters.c.dataset == source_dataset_id,
                unnested_clusters.c.identifier == identifier,
            )
        )
        .scalar_subquery()
    )


def _build_hierarchy_up(
    source_cluster: Select,
    contains_table: CTE | Contains,
) -> CTE:
    """Build recursive CTE that finds all parent clusters.

    Args:
        source_cluster: Subquery that finds starting cluster
        contains_table: Contains or CTE of valid clusters to filter by
    """
    if isinstance(contains_table, CTE):
        child_col = contains_table.c.child
        parent_col = contains_table.c.parent
    else:
        child_col = contains_table.child
        child_col = contains_table.parent

    # Base case: direct parents
    base = (
        select(
            source_cluster.label("original_cluster"),
            source_cluster.label("child"),
            parent_col.label("parent"),
            literal(1).label("level"),
        )
        .select_from(contains_table)
        .where(child_col == source_cluster)
    )

    hierarchy_up = base.cte("hierarchy_up", recursive=True)

    # Recursive case
    recursive = (
        select(
            hierarchy_up.c.original_cluster,
            hierarchy_up.c.parent.label("child"),
            parent_col.label("parent"),
            (hierarchy_up.c.level + 1).label("level"),
        )
        .select_from(hierarchy_up)
        .join(contains_table, child_col == hierarchy_up.c.parent)
    )

    return hierarchy_up.union_all(recursive)


def _find_highest_parent(hierarchy_up: CTE) -> Select:
    """Find the topmost parent cluster from the hierarchy."""
    return (
        select(hierarchy_up.c.parent)
        .order_by(hierarchy_up.c.level.desc())
        .limit(1)
        .scalar_subquery()
    )


def _build_hierarchy_down(
    highest_parent: Select, unnested_clusters: CTE, contains_table: CTE | Contains
) -> CTE:
    """Build recursive CTE that finds all child clusters and their IDs.

    Args:
        highest_parent: Subquery that finds top cluster
        unnested_clusters: CTE with unnested cluster IDs
        contains_table: Contains or CTE of valid clusters to filter by
    """
    if isinstance(contains_table, CTE):
        child_col = contains_table.c.child
        parent_col = contains_table.c.parent
    else:
        child_col = contains_table.child
        child_col = contains_table.parent

    # Base case: Get both direct children and their IDs
    base = (
        select(
            highest_parent.label("parent"),
            child_col.label("child"),
            literal(1).label("level"),
            unnested_clusters.c.dataset.label("dataset"),
            unnested_clusters.c.source_identifier.label("source_identifier"),
        )
        .select_from(contains_table)
        .join_from(
            contains_table,
            unnested_clusters,
            unnested_clusters.c.cluster_id == child_col,
            isouter=True,
        )
        .where(parent_col == highest_parent)
    )

    hierarchy_down = base.cte("hierarchy_down", recursive=True)

    # Recursive case: Get both intermediate nodes AND their leaf records
    recursive = (
        select(
            hierarchy_down.c.parent,
            child_col.label("child"),
            (hierarchy_down.c.level + 1).label("level"),
            unnested_clusters.c.dataset.label("dataset"),
            unnested_clusters.c.source_identifier.label("source_identifier"),
        )
        .select_from(hierarchy_down)
        .join_from(
            hierarchy_down,
            contains_table,
            parent_col == hierarchy_down.c.child,
        )
        .join_from(
            contains_table,
            unnested_clusters,
            unnested_clusters.c.cluster_id == child_col,
            isouter=True,
        )
        .where(
            hierarchy_down.c.source_identifier.is_(None)
        )  # Only recurse on non-leaf nodes
    )

    return hierarchy_down.union_all(recursive)


def _build_match_query(
    identifier: str,
    dataset_source: SourceConfigs,
    resolution: ModelResolutionName,
    session: Session,
    threshold: int | None = None,
) -> Select:
    """Builds the SQL query that powers the match function."""
    # Get truth resolution
    truth_resolution = (
        session.query(Resolutions).filter(Resolutions.name == resolution).first()
    )
    if truth_resolution is None:
        raise MatchboxResolutionNotFoundError(name=resolution)

    # Get resolution lineage and resolve thresholds
    lineage_truths = truth_resolution.get_lineage()
    thresholds = _resolve_thresholds(
        lineage_truths=lineage_truths,
        resolution=truth_resolution,
        threshold=threshold,
    )

    # Get valid clusters across all resolutions
    valid_clusters = _union_valid_clusters(thresholds)

    # Pre-filter Contains table if valid_clusters is provided
    contains_table = Contains
    if valid_clusters is not None:
        contains_table = _build_valid_contains(valid_clusters, name="valid_contains_up")

    # Build the query components
    unnested = _build_unnested_clusters()
    source_cluster = _find_source_cluster(
        unnested, dataset_source.source_id, identifier
    )
    hierarchy_up = _build_hierarchy_up(source_cluster, contains_table)
    highest = _find_highest_parent(hierarchy_up)
    hierarchy_down = _build_hierarchy_down(highest, unnested, contains_table)

    # Get all matched IDs
    final_stmt = (
        select(
            hierarchy_down.c.parent.label("cluster"),
            hierarchy_down.c.dataset,
            hierarchy_down.c.source_identifier,
        )
        .distinct()
        .select_from(hierarchy_down)
    )

    return final_stmt


def match(
    targets: list[SourceResolutionName],
    source: SourceResolutionName,
    identifier: str,
    resolution: ModelResolutionName,
    threshold: int | None = None,
) -> list[Match]:
    """Matches an ID in a source dataset and returns the keys in the targets.

    To accomplish this, the function:

    * Reconstructs the resolution lineage from the specified resolution
    * Iterates through each target, and
        * Retrieves its cluster hash according to the resolution
        * Retrieves all other IDs in the cluster in the source dataset
        * Retrieves all other IDs in the cluster in the target dataset
    * Returns the results as Match objects, one per target
    """
    with MBDB.get_session() as session:
        # Get all matches for source_identifier in all possible targets
        dataset_source = _get_dataset_source(source, session)

        match_stmt = _build_match_query(
            identifier=identifier,
            dataset_source=dataset_source,
            resolution=resolution,
            session=session,
            threshold=threshold,
        )

        logger.debug(f"Match SQL: \n {compile_sql(match_stmt)}")

        matches = session.execute(match_stmt).all()

        # Return matches in target sources only
        cluster = None
        matches_by_source_id: dict[int, set] = {}
        for cluster_id, source_id, id_in_source in matches:
            if cluster is None:
                cluster = cluster_id
            if source_id not in matches_by_source_id:
                matches_by_source_id[source_id] = set()
            matches_by_source_id[source_id].add(id_in_source)

        result = []
        for target in targets:
            target_source = _get_dataset_source(target, session)
            match_obj = Match(
                cluster=cluster,
                source=source,
                source_identifier=matches_by_source_id.get(
                    dataset_source.source_id, set()
                ),
                target=target,
                target_identifier=matches_by_source_id.get(
                    target_source.source_id, set()
                ),
            )
            result.append(match_obj)

        return result
