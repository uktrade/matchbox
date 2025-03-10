from typing import TYPE_CHECKING, Any, TypeVar

import pyarrow as pa
from sqlalchemy import BIGINT, Engine, and_, cast, func, literal, null, select, union
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import CTE, Select

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.sources import Match, SourceAddress
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    Resolutions,
    Sources,
)

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any

T = TypeVar("T")


def _get_dataset_resolution(
    source_name_address: SourceAddress, session: Session
) -> Resolutions:
    """Converts the the named address of source to a Resolutions ORM object."""
    dataset_resolution = (
        session.query(Resolutions)
        .join(Sources, Sources.resolution_id == Resolutions.resolution_id)
        .filter(
            Sources.full_name == source_name_address.full_name,
            Sources.warehouse_hash == source_name_address.warehouse_hash,
        )
        .first()
    )
    if dataset_resolution is None:
        raise MatchboxSourceNotFoundError(
            address=str(source_name_address),
        )

    return dataset_resolution


def _resolve_thresholds(
    lineage_truths: dict[str, float],
    resolution: Resolutions,
    threshold: float | None,
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
        elif isinstance(threshold, float):
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
            # This is a dataset - get all its clusters directly
            resolution_valid = select(Clusters.cluster_id.label("cluster")).where(
                Clusters.dataset == resolution_id
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

    return valid_clusters.cte("valid_clusters")


def _resolve_cluster_hierarchy(
    dataset_id: int,
    resolution: Resolutions,
    engine: Engine,
    threshold: float | None = None,
) -> Select:
    """Resolves the final cluster assignments for all records in a dataset.

    Args:
        dataset_id: ID of the dataset to query
        resolution: Resolution object representing the point of truth
        engine: Engine for database connection
        threshold: Optional threshold value

    Returns:
        SQLAlchemy Select statement that will resolve to (hash, id) pairs, where
        hash is the ultimate parent cluster hash and id is the original record ID
    """
    with Session(engine) as session:
        dataset_resolution = session.get(Resolutions, dataset_id)
        if dataset_resolution is None:
            raise MatchboxSourceNotFoundError()

        try:
            lineage_truths = resolution.get_lineage_to_dataset(
                dataset=dataset_resolution
            )
        except ValueError as e:
            raise MatchboxResolutionNotFoundError(
                f"Invalid resolution lineage: {str(e)}"
            ) from e

        thresholds = _resolve_thresholds(
            lineage_truths=lineage_truths,
            resolution=resolution,
            threshold=threshold,
        )

        # Get clusters valid across all resolutions in lineage
        valid_clusters = _union_valid_clusters(thresholds)

        # Get base mapping of IDs to clusters
        mapping_0 = (
            select(
                Clusters.cluster_id.label("cluster_id"),
                func.unnest(Clusters.source_pk).label("source_pk"),
            )
            .where(
                and_(
                    Clusters.cluster_id.in_(select(valid_clusters.c.cluster)),
                    Clusters.dataset == dataset_id,
                    Clusters.source_pk.isnot(None),
                )
            )
            .cte("mapping_0")
        )

        # Build recursive hierarchy CTE
        hierarchy = (
            # Base case: direct parents
            select(
                mapping_0.c.cluster_id.label("original_cluster"),
                mapping_0.c.cluster_id.label("child"),
                Contains.parent.label("parent"),
                literal(1).label("level"),
            )
            .select_from(mapping_0)
            .join(Contains, Contains.child == mapping_0.c.cluster_id)
            .where(Contains.parent.in_(select(valid_clusters.c.cluster)))
            .cte("hierarchy", recursive=True)
        )

        # Recursive case
        recursive = (
            select(
                hierarchy.c.original_cluster,
                hierarchy.c.parent.label("child"),
                Contains.parent.label("parent"),
                (hierarchy.c.level + 1).label("level"),
            )
            .select_from(hierarchy)
            .join(Contains, Contains.child == hierarchy.c.parent)
            .where(Contains.parent.in_(select(valid_clusters.c.cluster)))
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
                mapping_0.c.source_pk,
                func.coalesce(
                    highest_parents.c.highest_parent, mapping_0.c.cluster_id
                ).label("final_parent"),
            )
            .select_from(mapping_0)
            .join(
                highest_parents,
                highest_parents.c.original_cluster == mapping_0.c.cluster_id,
                isouter=True,
            )
            .cte("final_mapping")
        )

        # Final select statement
        return select(
            final_mapping.c.final_parent.label("id"), final_mapping.c.source_pk
        )


def query(
    engine: Engine,
    source_address: SourceAddress,
    resolution_name: str | None = None,
    threshold: float | None = None,
    limit: int = None,
) -> pa.Table:
    """Queries Matchbox and the Source warehouse to retrieve linked data.

    Takes the dictionaries of tables and fields outputted by selectors and
    queries database for them. If a "point of truth" resolution is supplied, will
    attach the clusters this data belongs to.

    To accomplish this, the function:

    * Iterates through each selector, and
        * Retrieves its data in Matchbox according to the optional point of truth,
        including its hash and cluster hash
        * Retrieves its raw data from its Source's warehouse
        * Joins the two together
    * Unions the results, one row per item of data in the warehouses

    Returns:
        A table containing the requested data from each table, unioned together,
        with the hash key of each row in Matchbox
    """
    with Session(engine) as session:
        dataset_resolution = _get_dataset_resolution(source_address, session)

        if resolution_name:
            resolution = (
                session.query(Resolutions)
                .filter(Resolutions.name == resolution_name)
                .first()
            )
            if resolution is None:
                raise MatchboxResolutionNotFoundError(resolution_name=resolution_name)
        else:
            resolution = dataset_resolution

        id_query = _resolve_cluster_hierarchy(
            dataset_id=dataset_resolution.resolution_id,
            resolution=resolution,
            threshold=threshold,
            engine=engine,
        )

        if limit:
            id_query = id_query.limit(limit)

        mb_ids = sql_to_df(id_query, engine, return_type="arrow")

        return mb_ids


def _build_unnested_clusters() -> CTE:
    """Create CTE that unnests cluster IDs for easier joining."""
    return (
        select(
            Clusters.cluster_id,
            Clusters.dataset,
            func.unnest(Clusters.source_pk).label("source_pk"),
        )
        .select_from(Clusters)
        .cte("unnested_clusters")
    )


def _find_source_cluster(
    unnested_clusters: CTE, source_dataset_id: int, source_pk: str
) -> Select:
    """Find the initial cluster containing the source primary key."""
    return (
        select(unnested_clusters.c.cluster_id)
        .select_from(unnested_clusters)
        .where(
            and_(
                unnested_clusters.c.dataset == source_dataset_id,
                unnested_clusters.c.source_pk == source_pk,
            )
        )
        .scalar_subquery()
    )


def _build_hierarchy_up(
    source_cluster: Select, valid_clusters: CTE | None = None
) -> CTE:
    """Build recursive CTE that finds all parent clusters.

    Args:
        source_cluster: Subquery that finds starting cluster
        valid_clusters: Optional CTE of valid clusters to filter by
    """
    # Base case: direct parents
    base = (
        select(
            source_cluster.label("original_cluster"),
            source_cluster.label("child"),
            Contains.parent.label("parent"),
            literal(1).label("level"),
        )
        .select_from(Contains)
        .where(Contains.child == source_cluster)
    )

    # Add valid clusters filter if provided
    if valid_clusters is not None:
        base = base.where(Contains.parent.in_(select(valid_clusters.c.cluster)))

    hierarchy_up = base.cte("hierarchy_up", recursive=True)

    # Recursive case
    recursive = (
        select(
            hierarchy_up.c.original_cluster,
            hierarchy_up.c.parent.label("child"),
            Contains.parent.label("parent"),
            (hierarchy_up.c.level + 1).label("level"),
        )
        .select_from(hierarchy_up)
        .join(Contains, Contains.child == hierarchy_up.c.parent)
    )

    # Add valid clusters filter to recursive part if provided
    if valid_clusters is not None:
        recursive = recursive.where(
            Contains.parent.in_(select(valid_clusters.c.cluster))
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
    highest_parent: Select, unnested_clusters: CTE, valid_clusters: CTE | None = None
) -> CTE:
    """Build recursive CTE that finds all child clusters and their IDs.

    Args:
        highest_parent: Subquery that finds top cluster
        unnested_clusters: CTE with unnested cluster IDs
        valid_clusters: Optional CTE of valid clusters to filter by
    """
    # Base case: Get both direct children and their IDs
    base = (
        select(
            highest_parent.label("parent"),
            Contains.child.label("child"),
            literal(1).label("level"),
            unnested_clusters.c.dataset.label("dataset"),
            unnested_clusters.c.source_pk.label("source_pk"),
        )
        .select_from(Contains)
        .join_from(
            Contains,
            unnested_clusters,
            unnested_clusters.c.cluster_id == Contains.child,
            isouter=True,
        )
        .where(Contains.parent == highest_parent)
    )

    # Add valid clusters filter if provided
    if valid_clusters is not None:
        base = base.where(Contains.child.in_(select(valid_clusters.c.cluster)))

    hierarchy_down = base.cte("hierarchy_down", recursive=True)

    # Recursive case: Get both intermediate nodes AND their leaf records
    recursive = (
        select(
            hierarchy_down.c.parent,
            Contains.child.label("child"),
            (hierarchy_down.c.level + 1).label("level"),
            unnested_clusters.c.dataset.label("dataset"),
            unnested_clusters.c.source_pk.label("source_pk"),
        )
        .select_from(hierarchy_down)
        .join_from(
            hierarchy_down,
            Contains,
            Contains.parent == hierarchy_down.c.child,
        )
        .join_from(
            Contains,
            unnested_clusters,
            unnested_clusters.c.cluster_id == Contains.child,
            isouter=True,
        )
        .where(hierarchy_down.c.source_pk.is_(None))  # Only recurse on non-leaf nodes
    )

    # Add valid clusters filter to recursive part if provided
    if valid_clusters is not None:
        recursive = recursive.where(
            Contains.child.in_(select(valid_clusters.c.cluster))
        )

    return hierarchy_down.union_all(recursive)


def _get_target_resolutions(
    targets: list[SourceAddress], session: Session
) -> list[tuple[Resolutions, str]]:
    """Get target resolutions with source address."""
    target_resolutions = []
    for t in targets:
        target_resolution = _get_dataset_resolution(t, session)
        target_resolutions.append((target_resolution, t))

    return target_resolutions


def _build_match_query(
    source_pk: str,
    source_resolution_id: int,
    resolution_name: str,
    session: Session,
    threshold: float | None = None,
) -> Select:
    """Builds the SQL query that powers the match function."""
    # Get truth resolution
    truth_resolution = (
        session.query(Resolutions).filter(Resolutions.name == resolution_name).first()
    )
    if truth_resolution is None:
        raise MatchboxResolutionNotFoundError(resolution_name=resolution_name)

    # Get resolution lineage and resolve thresholds
    lineage_truths = truth_resolution.get_lineage()
    thresholds = _resolve_thresholds(
        lineage_truths=lineage_truths,
        resolution=truth_resolution,
        threshold=threshold,
    )

    # Get valid clusters across all resolutions
    valid_clusters = _union_valid_clusters(thresholds)

    # Build the query components
    unnested = _build_unnested_clusters()
    source_cluster = _find_source_cluster(unnested, source_resolution_id, source_pk)
    hierarchy_up = _build_hierarchy_up(source_cluster, valid_clusters)
    highest = _find_highest_parent(hierarchy_up)
    hierarchy_down = _build_hierarchy_down(highest, unnested, valid_clusters)

    # Get all matched IDs
    final_stmt = (
        select(
            hierarchy_down.c.parent.label("cluster"),
            hierarchy_down.c.dataset,
            hierarchy_down.c.source_pk,
        )
        .distinct()
        .select_from(hierarchy_down)
    )

    return final_stmt


def match(
    engine: Engine,
    source_pk: str,
    source: SourceAddress,
    targets: list[SourceAddress],
    resolution_name: str,
    threshold: float | None = None,
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
    with Session(engine) as session:
        # Get all matches for source_pk in all possible targets
        dataset_resolution = _get_dataset_resolution(source, session)

        match_stmt = _build_match_query(
            source_pk=source_pk,
            source_resolution_id=dataset_resolution.resolution_id,
            resolution_name=resolution_name,
            session=session,
            threshold=threshold,
        )

        matches = session.execute(match_stmt).all()

        # Return matches in target resolutions only
        target_resolutions = _get_target_resolutions(targets, session)

        cluster = None
        matches_by_dataset: dict[int, set] = {}
        for cluster_id, dataset_id, id_in_source in matches:
            if cluster is None:
                cluster = cluster_id
            if dataset_id not in matches_by_dataset:
                matches_by_dataset[dataset_id] = set()
            matches_by_dataset[dataset_id].add(id_in_source)

        result = []
        for target_resolution, target_address in target_resolutions:
            match_obj = Match(
                cluster=cluster,
                source=source,
                source_id=matches_by_dataset.get(
                    dataset_resolution.resolution_id, set()
                ),
                target=target_address,
                target_id=matches_by_dataset.get(
                    target_resolution.resolution_id, set()
                ),
            )
            result.append(match_obj)

        return result
