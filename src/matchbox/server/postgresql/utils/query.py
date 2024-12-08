import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from sqlalchemy import Engine, and_, cast, func, literal, null, select, union
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import Source, sql_to_df
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


def _get_valid_clusters_for_model(model_hash: bytes, threshold: float) -> Select:
    """Get clusters that meet the threshold for a specific model."""
    return select(Probabilities.cluster.label("cluster")).where(
        and_(
            Probabilities.model == hash_to_hex_decode(model_hash),
            Probabilities.probability >= threshold,
        )
    )


def _union_valid_clusters(lineage_thresholds: dict[bytes, float]) -> Select:
    """Creates a CTE of clusters that are valid for any model in the lineage.

    Each model may have a different threshold.
    """
    valid_clusters = None

    for model_hash, threshold in lineage_thresholds.items():
        model_valid = _get_valid_clusters_for_model(model_hash, threshold)

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
            # Get the dataset model
            dataset = (
                session.query(Models)
                .join(Sources, Sources.model == Models.hash)
                .filter(
                    Sources.schema == source.db_schema,
                    Sources.table == source.db_table,
                    Sources.id == source.db_pk,
                )
                .first()
            )

            if dataset is None:
                raise MatchboxDatasetError(
                    db_schema=source.db_schema, db_table=source.db_table
                )

            # Warn if non-indexed fields have been requested
            not_indexed = set(fields) - set(
                c.literal.name for c in source.db_columns if c.indexed
            )
            if not_indexed:
                warnings.warn(
                    "Found non-indexed fields. Do not use these fields in match jobs:"
                    f"{', '.join(sorted(not_indexed))}",
                    stacklevel=2,
                )

            hash_query = _resolve_cluster_hierarchy(
                dataset_hash=dataset.hash,
                model=truth_model if truth_model else dataset,
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
