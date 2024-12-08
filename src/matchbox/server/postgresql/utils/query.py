import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import (
    Engine,
    and_,
    cast,
    func,
    literal,
    null,
    select,
    union,
)
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import CTE, Select

from matchbox.common.db import Match, Source, get_schema_table_names, sql_to_df
from matchbox.common.exceptions import (
    MatchboxDatasetError,
    MatchboxModelError,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Models,
    Probabilities,
    Sources,
)

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any

T = TypeVar("T")

logic_logger = logging.getLogger("mb_logic")


def hash_to_hex_decode(hash: bytes) -> bytes:
    """A workround for PostgreSQL so we can compile the query and use ConnectorX."""
    return func.decode(hash.hex(), "hex")


def key_to_sqlalchemy_label(key: str, source: Source) -> str:
    """Converts a key to the SQLAlchemy LABEL_STYLE_TABLENAME_PLUS_COL."""
    return f"{source.db_schema}_{source.db_table}_{key}"


def source_to_dataset_model(source: Source | str, session: Session) -> Models:
    """Converts a Source object to a Sources ORM object."""
    if isinstance(source, str):
        source_schema, source_table = get_schema_table_names(source, validate=True)
    else:
        source_schema, source_table = source.db_schema, source.db_table

    source_dataset = (
        session.query(Models)
        .join(Sources, Sources.model == Models.hash)
        .filter(
            Sources.schema == source_schema,
            Sources.table == source_table,
        )
        .first()
    )
    if source_dataset is None:
        raise MatchboxDatasetError(
            db_schema=source_schema,
            db_table=source_table,
        )

    return source_dataset


def _resolve_thresholds(
    lineage_truths: dict[str, float],
    model: Models,
    threshold: float | dict[str, float] | None,
    session: Session,
) -> dict[bytes, float]:
    """
    Resolves final thresholds for each model in the lineage based on user input.

    Args:
        lineage_truths: Dict from get_lineage_to_dataset with model hash -> cached truth
        model: The target model being used for clustering
        threshold: User-supplied threshold value or dict
        session: SQLAlchemy session

    Returns:
        Dict mapping model hash to their final threshold values
    """
    resolved_thresholds = {}

    for model_hash, default_truth in lineage_truths.items():
        # Dataset
        if default_truth is None:
            resolved_thresholds[model_hash] = None
            continue

        # Model
        if threshold is None:
            resolved_thresholds[model_hash] = default_truth
        elif isinstance(threshold, float):
            resolved_thresholds[model_hash] = (
                threshold if model_hash == model.hash.hex() else default_truth
            )
        elif isinstance(threshold, dict):
            model_obj = (
                session.query(Models)
                .filter(Models.hash == bytes.fromhex(model_hash))
                .first()
            )
            resolved_thresholds[model_hash] = threshold.get(
                model_obj.name, default_truth
            )
        else:
            raise ValueError(f"Invalid threshold type: {type(threshold)}")

    return resolved_thresholds


def _union_valid_clusters(lineage_thresholds: dict[bytes, float]) -> Select:
    """Creates a CTE of clusters that are valid for any model in the lineage.

    Each model may have a different threshold.
    """
    valid_clusters = None

    for model_hash, threshold in lineage_thresholds.items():
        if threshold is None:
            # This is a dataset - get all its clusters directly
            model_valid = select(Clusters.hash.label("cluster")).where(
                Clusters.dataset == hash_to_hex_decode(model_hash)
            )
        else:
            # This is a model - get clusters meeting threshold
            model_valid = select(Probabilities.cluster.label("cluster")).where(
                and_(
                    Probabilities.model == hash_to_hex_decode(model_hash),
                    Probabilities.probability >= threshold,
                )
            )

        if valid_clusters is None:
            valid_clusters = model_valid
        else:
            valid_clusters = union(valid_clusters, model_valid)

    if valid_clusters is None:
        # Handle empty lineage case
        return select(cast(null(), BYTEA).label("cluster")).where(False)

    return valid_clusters.cte("valid_clusters")


def _resolve_cluster_hierarchy(
    dataset_hash: bytes,
    model: Models,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Select:
    """
    Resolves the final cluster assignments for all records in a dataset.

    Args:
        dataset_hash: Hash of the dataset to query
        model: Model object representing the point of truth
        engine: Engine for database connection
        threshold: Optional threshold value or dict of model_name -> threshold

    Returns:
        SQLAlchemy Select statement that will resolve to (hash, id) pairs, where
        hash is the ultimate parent cluster hash and id is the original record ID
    """
    with Session(engine) as session:
        dataset_model = session.get(Models, dataset_hash)
        if dataset_model is None:
            raise MatchboxDatasetError("Dataset not found")

        try:
            lineage_truths = model.get_lineage_to_dataset(model=dataset_model)
        except ValueError as e:
            raise MatchboxModelError(f"Invalid model lineage: {str(e)}") from e

        thresholds = _resolve_thresholds(
            lineage_truths=lineage_truths,
            model=model,
            threshold=threshold,
            session=session,
        )

        # Get clusters valid across all models in lineage
        valid_clusters = _union_valid_clusters(thresholds)

        # Get base mapping of IDs to clusters
        mapping_0 = (
            select(
                Clusters.hash.label("cluster_hash"),
                func.unnest(Clusters.id).label("id"),
            )
            .where(
                and_(
                    Clusters.hash.in_(select(valid_clusters.c.cluster)),
                    Clusters.dataset == hash_to_hex_decode(dataset_hash),
                    Clusters.id.isnot(None),
                )
            )
            .cte("mapping_0")
        )

        # Build recursive hierarchy CTE
        hierarchy = (
            # Base case: direct parents
            select(
                mapping_0.c.cluster_hash.label("original_cluster"),
                mapping_0.c.cluster_hash.label("child"),
                Contains.parent.label("parent"),
                literal(1).label("level"),
            )
            .select_from(mapping_0)
            .join(Contains, Contains.child == mapping_0.c.cluster_hash)
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
                mapping_0.c.id,
                func.coalesce(
                    highest_parents.c.highest_parent, mapping_0.c.cluster_hash
                ).label("final_parent"),
            )
            .select_from(mapping_0)
            .join(
                highest_parents,
                highest_parents.c.original_cluster == mapping_0.c.cluster_hash,
                isouter=True,
            )
            .cte("final_mapping")
        )

        # Final select statement
        return select(final_mapping.c.final_parent.label("hash"), final_mapping.c.id)


def query(
    selector: dict[Source, list[str]],
    engine: Engine,
    return_type: Literal["pandas", "arrow", "polars"] = "pandas",
    model: str | None = None,
    threshold: float | dict[str, float] | None = None,
    limit: int = None,
) -> DataFrame | pa.Table | PolarsDataFrame:
    """
    Queries Matchbox and the Source warehouse to retrieve linked data.

    Takes the dictionaries of tables and fields outputted by selectors and
    queries database for them. If a model "point of truth" is supplied, will
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
        with the hash key of each row in Matchbox, in the requested return type
    """
    tables: list[pa.Table] = []

    if limit:
        limit_base = limit // len(selector)
        limit_remainder = limit % len(selector)

    with Session(engine) as session:
        # If a model was specified, validate and retrieve it
        truth_model = None
        if model is not None:
            truth_model = session.query(Models).filter(Models.name == model).first()
            if truth_model is None:
                raise MatchboxModelError(f"Model {model} not found")

        # Process each source dataset
        for source, fields in selector.items():
            dataset_model = source_to_dataset_model(source, session)
            hash_query = _resolve_cluster_hierarchy(
                dataset_hash=dataset_model.hash,
                model=truth_model if truth_model else dataset_model,
                threshold=threshold,
                engine=engine,
            )

            # Apply limit if specified
            if limit:
                remain = 1 if limit_remainder > 0 else 0
                if remain:
                    limit_remainder -= 1
                hash_query = hash_query.limit(limit_base + remain)

            # Get cluster assignments
            mb_hashes = sql_to_df(hash_query, engine, return_type="arrow")

            # Get source data
            raw_data = source.to_arrow(
                fields=set([source.db_pk] + fields), pks=mb_hashes["id"].to_pylist()
            )

            # Join and select columns
            joined_table = raw_data.join(
                right_table=mb_hashes,
                keys=key_to_sqlalchemy_label(source.db_pk, source),
                right_keys="id",
                join_type="inner",
            )

            keep_cols = ["hash"] + [key_to_sqlalchemy_label(f, source) for f in fields]
            match_cols = [col for col in joined_table.column_names if col in keep_cols]

            tables.append(joined_table.select(match_cols))

    # Combine results
    result = pa.concat_tables(tables, promote_options="default")

    # Return in requested format
    if return_type == "arrow":
        return result
    elif return_type == "pandas":
        return result.to_pandas(
            use_threads=True,
            split_blocks=True,
            self_destruct=True,
            types_mapper=ArrowDtype,
        )
    else:
        raise ValueError(f"return_type of {return_type} not valid")


def _build_unnested_clusters() -> CTE:
    """Create CTE that unnests cluster IDs for easier joining."""
    return (
        select(Clusters.hash, Clusters.dataset, func.unnest(Clusters.id).label("id"))
        .select_from(Clusters)
        .cte("unnested_clusters")
    )


def _find_source_cluster(
    unnested_clusters: CTE, source_dataset_hash: bytes, source_id: str
) -> Select:
    """Find the initial cluster containing the source ID."""
    return (
        select(unnested_clusters.c.hash)
        .select_from(unnested_clusters)
        .where(
            and_(
                unnested_clusters.c.dataset == hash_to_hex_decode(source_dataset_hash),
                unnested_clusters.c.id == source_id,
            )
        )
        .scalar_subquery()
    )


def _build_hierarchy_up(
    source_cluster: Select, valid_clusters: CTE | None = None
) -> CTE:
    """
    Build recursive CTE that finds all parent clusters.

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
    """
    Build recursive CTE that finds all child clusters and their IDs.

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
            unnested_clusters.c.id.label("id"),
        )
        .select_from(Contains)
        .join_from(
            Contains,
            unnested_clusters,
            unnested_clusters.c.hash == Contains.child,
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
            unnested_clusters.c.id.label("id"),
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
            unnested_clusters.c.hash == Contains.child,
            isouter=True,
        )
        .where(hierarchy_down.c.id.is_(None))  # Only recurse on non-leaf nodes
    )

    # Add valid clusters filter to recursive part if provided
    if valid_clusters is not None:
        recursive = recursive.where(
            Contains.child.in_(select(valid_clusters.c.cluster))
        )

    return hierarchy_down.union_all(recursive)


def match(
    source_id: str,
    source: str,
    target: str | list[str],
    model: str,
    engine: Engine,
    threshold: float | dict[str, float] | None = None,
) -> Match | list[Match]:
    """Matches an ID in a source dataset and returns the keys in the targets.

    To accomplish this, the function:

    * Reconstructs the model lineage from the specified model
    * Iterates through each target, and
        * Retrieves its cluster hash according to the model
        * Retrieves all other IDs in the cluster in the source dataset
        * Retrieves all other IDs in the cluster in the target dataset
    * Returns the results as Match objects, one per target
    """
    # Split source and target into schema/table
    targets = [target] if isinstance(target, str) else target
    target_pairs = [get_schema_table_names(t, validate=True) for t in targets]

    with Session(engine) as session:
        # Get source, target and truth models
        source_model = source_to_dataset_model(source, session)
        target_models = []
        for target in targets:
            target_models.append(source_to_dataset_model(target, session))
        truth_model = session.query(Models).filter(Models.name == model).first()
        if truth_model is None:
            raise MatchboxModelError(f"Model {model} not found")

        # Get model lineage and resolve thresholds
        lineage_truths = truth_model.get_lineage()
        thresholds = _resolve_thresholds(
            lineage_truths=lineage_truths,
            model=truth_model,
            threshold=threshold,
            session=session,
        )

        # Get valid clusters across all models
        valid_clusters = _union_valid_clusters(thresholds)

        # Build the query components
        unnested = _build_unnested_clusters()
        source_cluster = _find_source_cluster(unnested, source_model.hash, source_id)
        hierarchy_up = _build_hierarchy_up(source_cluster, valid_clusters)
        highest = _find_highest_parent(hierarchy_up)
        hierarchy_down = _build_hierarchy_down(highest, unnested, valid_clusters)

        # Get all matched IDs
        final_stmt = (
            select(
                hierarchy_down.c.parent.label("cluster"),
                hierarchy_down.c.dataset,
                hierarchy_down.c.id,
            )
            .distinct()
            .select_from(hierarchy_down)
        )
        matches = session.execute(final_stmt).all()

        # Group matches by dataset
        cluster = None
        matches_by_dataset = {}
        for cluster_hash, dataset_hash, id in matches:
            if cluster is None:
                cluster = cluster_hash
            if dataset_hash not in matches_by_dataset:
                matches_by_dataset[dataset_hash] = set()
            matches_by_dataset[dataset_hash].add(id)

        # Create Match objects for each target
        result = []
        for target_model in target_models:
            # Get source/target table names
            target_schema, target_table = next(
                (schema, table)
                for schema, table in target_pairs
                if session.get(Sources, target_model.hash).schema == schema
                and session.get(Sources, target_model.hash).table == table
            )
            target_name = f"{target_schema}.{target_table}"

            # Get source and target IDs
            source_ids = {
                id
                for _, dataset_hash, id in matches
                if dataset_hash == source_model.hash
            }
            target_ids = {
                id
                for _, dataset_hash, id in matches
                if dataset_hash == target_model.hash
            }

            match_obj = Match(
                cluster=cluster,
                source=source,
                source_id=source_ids,
                target=target_name,
                target_id=target_ids,
            )
            result.append(match_obj)

        return result[0] if isinstance(target, str) else result
