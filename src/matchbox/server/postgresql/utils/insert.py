"""Utilities for inserting data into the PostgreSQL backend."""

from collections.abc import Iterator

import polars as pl
import pyarrow as pa
from sqlalchemy import func, join, literal, select
from sqlalchemy.dialects.postgresql import ARRAY, BIGINT, BYTEA, TEXT
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.sql import over

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    ModelResolutionPath,
    ResolutionType,
    SourceResolutionPath,
)
from matchbox.common.exceptions import (
    MatchboxResolutionExistingData,
    MatchboxResolutionInvalidData,
)
from matchbox.common.hash import IntMap, hash_arrow_table, hash_model_results
from matchbox.common.logging import logger
from matchbox.common.transform import Cluster, DisjointSet
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    PKSpace,
    Probabilities,
    Resolutions,
    Results,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    ingest_to_temporary_table,
    large_append,
)
from matchbox.server.postgresql.utils.query import get_parent_clusters_and_leaves


def _fetch_existing_clusters(data_hashes: pa.Table, column: str) -> pl.DataFrame:
    """Fetch existing clusters from database by joining with temporary table.

    Args:
        data_hashes: Arrow table with cluster hashes
        column: Name of the column containing the cluster hashes
    """
    with ingest_to_temporary_table(
        table_name="hashes",
        schema_name="mb",
        column_types={
            "cluster_hash": BYTEA(),
        },
        data=data_hashes.select([column]).rename_columns(["cluster_hash"]),
    ) as temp_table:
        existing_cluster_stmt = select(Clusters.cluster_id, Clusters.cluster_hash).join(
            temp_table, temp_table.c.cluster_hash == Clusters.cluster_hash
        )

        with MBDB.get_adbc_connection() as conn:
            existing_cluster_df: pl.DataFrame = sql_to_df(
                stmt=compile_sql(existing_cluster_stmt),
                connection=conn.dbapi_connection,
                return_type="polars",
            )
    return existing_cluster_df


def insert_hashes(
    path: SourceResolutionPath, data_hashes: pa.Table, batch_size: int
) -> None:
    """Indexes hash data for a source.

    Args:
        path: The path of the source resolution
        data_hashes: Arrow table containing hash data
        batch_size: Batch size for bulk operations

    Raises:
        MatchboxResolutionNotFoundError: If the specified resolution doesn't exist.
        MatchboxResolutionInvalidData: If data fingerprint conflicts with resolution.
        MatchboxResolutionExistingData: If data was already inserted for resolution.
    """
    log_prefix = f"Index hashes {path}"
    if data_hashes.num_rows == 0:
        logger.info("No hashes given.", prefix=log_prefix)
        return

    fingerprint = hash_arrow_table(data_hashes)

    with MBDB.get_session() as session:
        resolution = Resolutions.from_path(
            path=path, res_type=ResolutionType.SOURCE, session=session
        )
        # Check if the content hash is the same
        if resolution.fingerprint != fingerprint:
            raise MatchboxResolutionInvalidData

        # Determine if the resolution already has any keys
        existing_keys = session.execute(
            select(func.count())
            .select_from(
                join(
                    ClusterSourceKey,
                    SourceConfigs,
                    ClusterSourceKey.source_config_id == SourceConfigs.source_config_id,
                )
            )
            .where(SourceConfigs.resolution_id == resolution.resolution_id)
        ).scalar_one()

        if existing_keys > 0:
            raise MatchboxResolutionExistingData

        source_config_id = resolution.source_config.source_config_id

    with (
        ingest_to_temporary_table(
            table_name="incoming_hashes",
            schema_name="mb",
            column_types={"hash": BYTEA(), "keys": ARRAY(TEXT)},
            data=data_hashes.select(["hash", "keys"]),
            max_chunksize=batch_size,
        ) as incoming,
        MBDB.get_session() as session,
    ):
        try:
            distinct_hashes = (
                select(incoming.c.hash).distinct().subquery("distinct_hashes")
            )

            new_hashes = (
                select(distinct_hashes.c.hash.label("cluster_hash"))
                .select_from(
                    distinct_hashes.outerjoin(
                        Clusters, Clusters.cluster_hash == distinct_hashes.c.hash
                    )
                )
                .where(Clusters.cluster_id.is_(None))
                .subquery("new_hashes")
            )

            new_hashes_count = session.execute(
                select(func.count()).select_from(new_hashes)
            ).scalar_one()

            if new_hashes_count > 0:
                logger.info(
                    f"Will add {new_hashes_count:,} entries to Clusters table",
                    prefix=log_prefix,
                )
                base_cluster_id = PKSpace.reserve_block("clusters", new_hashes_count)

                numbered_clusters = (
                    select(
                        (
                            literal(base_cluster_id - 1, BIGINT)
                            + over(func.row_number())
                        ).label("cluster_id"),
                        new_hashes.c.cluster_hash,
                    )
                ).subquery("numbered_clusters")

                # Insert new clusters
                stmt_insert_clusters = (
                    pg_insert(Clusters)
                    .from_select(
                        ["cluster_id", "cluster_hash"],
                        select(
                            numbered_clusters.c.cluster_id,
                            numbered_clusters.c.cluster_hash,
                        ),
                    )
                    .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
                )
                session.execute(stmt_insert_clusters)
                session.flush()
            else:
                logger.info("No new clusters to add", prefix=log_prefix)

            cluster_map = (
                select(
                    Clusters.cluster_id,
                    Clusters.cluster_hash,
                ).select_from(
                    join(
                        Clusters,
                        distinct_hashes,
                        Clusters.cluster_hash == distinct_hashes.c.hash,
                    )
                )
            ).subquery("cluster_map")

            exploded = (
                select(
                    cluster_map.c.cluster_id,
                    func.unnest(incoming.c["keys"]).label("key"),
                )
                .select_from(
                    incoming.join(
                        cluster_map, cluster_map.c.cluster_hash == incoming.c.hash
                    )
                )
                .subquery("exploded")
            )

            # Count exploded rows
            new_keys_count = session.execute(
                select(func.count()).select_from(exploded)
            ).scalar_one()

            if new_keys_count > 0:
                logger.info(
                    f"Will add {new_keys_count:,} entries to ClusterSourceKey table",
                    prefix=log_prefix,
                )
                base_key_id = PKSpace.reserve_block("cluster_keys", new_keys_count)

                numbered_keys = (
                    select(
                        (
                            literal(base_key_id - 1, BIGINT) + over(func.row_number())
                        ).label("key_id"),
                        exploded.c.cluster_id,
                        literal(source_config_id, BIGINT).label("source_config_id"),
                        exploded.c.key,
                    )
                ).subquery("numbered_keys")

                # Unlike for clusters, we don't expect conflicts here:
                # - resolution should be locked
                # - earlier we checked that it doesn't have any data
                stmt_insert_keys = pg_insert(ClusterSourceKey).from_select(
                    ["key_id", "cluster_id", "source_config_id", "key"],
                    select(
                        numbered_keys.c.key_id,
                        numbered_keys.c.cluster_id,
                        numbered_keys.c.source_config_id,
                        numbered_keys.c.key,
                    ),
                )
                session.execute(stmt_insert_keys)
                session.commit()
            else:
                logger.info("No cluster keys to add", prefix=log_prefix)
        except Exception as e:
            # Log the error and rollback
            logger.warning(f"Error, rolling back: {e}", prefix=log_prefix)
            session.rollback()
            raise

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        ClusterSourceKey.__table__.fullname,
    )

    logger.info("Finished", prefix=log_prefix)


def _build_cluster_objects(
    nested_dict: dict[int, dict[str, list[dict]]],
    intmap: IntMap,
) -> dict[int, Cluster]:
    """Convert the nested dictionary to Cluster objects.

    Args:
        nested_dict: Dictionary from get_parent_clusters_and_leaves()
        intmap: IntMap object for creating new IDs safely

    Returns:
        Dict mapping cluster IDs to Cluster objects
    """
    cluster_lookup: dict[int, Cluster] = {}

    for cluster_id, data in nested_dict.items():
        # Create leaf clusters on-demand
        leaves = []
        for leaf_data in data["leaves"]:
            leaf_id = leaf_data["leaf_id"]
            if leaf_id not in cluster_lookup:
                cluster_lookup[leaf_id] = Cluster(
                    id=leaf_id, hash=leaf_data["leaf_hash"], intmap=intmap
                )
            leaves.append(cluster_lookup[leaf_id])

        # Create parent cluster
        cluster_lookup[cluster_id] = Cluster(
            id=cluster_id,
            hash=data["root_hash"],
            probability=data["probability"],
            leaves=leaves,
            intmap=intmap,
        )

    return cluster_lookup


def _results_to_cluster_pairs(
    cluster_lookup: dict[int, Cluster],
    results: pa.Table,
) -> Iterator[tuple[Cluster, Cluster, int]]:
    """Convert the results from a PyArrow table to an iterator of cluster pairs.

    Args:
        cluster_lookup (dict[int, Cluster]): A dictionary mapping cluster IDs to
            Cluster objects.
        results (pa.Table): The PyArrow table containing the results: left_id
            right_id, and probability.

    Returns:
        list[tuple[Cluster, Cluster, int]]: An iterator of tuples, each containing
            the left cluster, right cluster, and the probability, in descending
            order of probability.
    """
    for row in pl.from_arrow(results).sort("probability", descending=True).iter_rows():
        left_cluster: Cluster = cluster_lookup[row[0]]
        right_cluster: Cluster = cluster_lookup[row[1]]

        yield left_cluster, right_cluster, row[2]


def _build_cluster_hierarchy(
    cluster_lookup: dict[int, Cluster], probabilities: pa.Table
) -> dict[bytes, Cluster]:
    """Build cluster hierarchy using disjoint sets and probability thresholding.

    Args:
        cluster_lookup: Dictionary mapping cluster IDs to Cluster objects
        probabilities: Arrow table containing probability data

    Returns:
        Dictionary mapping cluster hashes to Cluster objects
    """
    logger.debug("Computing hierarchies")

    djs = DisjointSet[Cluster]()
    all_clusters: dict[bytes, Cluster] = {}
    seen_components: set[frozenset[Cluster]] = set()
    threshold: int = int(pa.compute.max(probabilities["probability"]).as_py())

    def _process_components(probability: int) -> None:
        """Process components at the current threshold."""
        components: set[frozenset[Cluster]] = {
            frozenset(component) for component in djs.get_components()
        }
        for component in components.difference(seen_components):
            cluster = Cluster.combine(
                clusters=component,
                probability=probability,
            )
            all_clusters[cluster.hash] = cluster

        return components

    for left_cluster, right_cluster, probability in _results_to_cluster_pairs(
        cluster_lookup, probabilities
    ):
        if probability < threshold:
            # Process the components at the previous threshold
            seen_components.update(_process_components(threshold))
            threshold = probability

        djs.union(left_cluster, right_cluster)

    # Process any remaining components
    _process_components(probability)

    return all_clusters


def _create_clusters_dataframe(all_clusters: dict[bytes, Cluster]) -> pl.DataFrame:
    """Create a DataFrame with cluster data and existing/new cluster information.

    Args:
        all_clusters: Dictionary mapping cluster hashes to Cluster objects

    Returns:
        Polars DataFrame with columns: cluster_id, cluster_hash, cluster_struct, new
    """
    # Convert all clusters to a DataFrame, converting Clusters to Polars structs
    cluster_data = []
    for cluster_hash, cluster in all_clusters.items():
        cluster_struct = {
            "id": cluster.id,
            "probability": cluster.probability,
            "leaves": [leaf.id for leaf in cluster.leaves] if cluster.leaves else [],
        }
        cluster_data.append(
            {"cluster_hash": cluster_hash, "cluster_struct": cluster_struct}
        )

    all_clusters_df = pl.DataFrame(
        cluster_data,
        schema={
            "cluster_hash": pl.Binary,
            "cluster_struct": pl.Struct(
                {"id": pl.Int64, "probability": pl.Int8, "leaves": pl.List(pl.Int64)}
            ),
        },
    )

    existing_cluster_df = _fetch_existing_clusters(
        all_clusters_df.to_arrow(), "cluster_hash"
    )

    # Use anti_join to find hashes that don't exist in the lookup
    new_clusters_df = all_clusters_df.join(
        existing_cluster_df, on="cluster_hash", how="anti"
    )

    # Assign new cluster IDs if needed
    next_cluster_id: int = 0
    if not new_clusters_df.is_empty():
        next_cluster_id = PKSpace.reserve_block("clusters", new_clusters_df.shape[0])

    new_clusters_df = new_clusters_df.with_columns(
        [
            (
                pl.arange(0, new_clusters_df.shape[0], dtype=pl.Int64) + next_cluster_id
            ).alias("cluster_id"),
            pl.lit(True).alias("new"),
        ]
    )

    # Add cluster data to existing and add new flag
    existing_with_data = all_clusters_df.join(
        existing_cluster_df, on="cluster_hash", how="inner"
    ).with_columns(pl.lit(False).alias("new"))

    # Concatenate existing and new clusters
    return pl.concat([existing_with_data, new_clusters_df]).select(
        "cluster_id", "cluster_hash", "cluster_struct", "new"
    )


def _results_to_insert_tables(
    resolution: Resolutions, probabilities: pa.Table
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Takes probabilities and returns three Arrow tables that can be inserted exactly.

    Returns:
        A tuple containing:

            * A Clusters update Arrow table
            * A Contains update Arrow table
            * A Probabilities update Arrow table
    """
    log_prefix = f"Model {resolution.name}"

    if probabilities.shape[0] == 0:
        clusters = pa.table(
            {"cluster_id": [], "cluster_hash": []},
            schema=pa.schema(
                [("cluster_id", pa.uint64()), ("cluster_hash", pa.large_binary())]
            ),
        )
        contains = pa.table(
            {"root": [], "leaf": []},
            schema=pa.schema([("root", pa.uint64()), ("leaf", pa.uint64())]),
        )
        probabilities = pa.table(
            {"resolution_id": [], "cluster_id": [], "probability": []},
            schema=pa.schema(
                [
                    ("resolution_id", pa.uint64()),
                    ("cluster_id", pa.uint64()),
                    ("probability", pa.uint8()),
                ]
            ),
        )
        return clusters, contains, probabilities

    logger.info("Wrangling data to insert tables", prefix=log_prefix)

    # Get a cluster lookup dictionary based on the resolution's parents
    im = IntMap()

    nested_data = get_parent_clusters_and_leaves(resolution=resolution)
    cluster_lookup: dict[int, Cluster] = _build_cluster_objects(nested_data, im)

    logger.debug("Computing hierarchies", prefix=log_prefix)
    all_clusters: dict[bytes, Cluster] = _build_cluster_hierarchy(
        cluster_lookup=cluster_lookup, probabilities=probabilities
    )
    del cluster_lookup

    logger.debug("Reconciling clusters against database", prefix=log_prefix)
    all_clusters_df = _create_clusters_dataframe(all_clusters)
    del all_clusters

    # Filter to new clusters for Clusters table
    new_clusters_df = all_clusters_df.filter(pl.col("new")).select(
        "cluster_id", "cluster_hash"
    )

    # Filter to new clusters and explode leaves for Contains table
    new_contains_df = (
        all_clusters_df.filter(pl.col("new"))
        .select("cluster_id", "cluster_struct")
        .rename({"cluster_id": "root"})
        .with_columns(pl.col("cluster_struct").struct.field("leaves").alias("leaf"))
        .drop("cluster_struct")
        .explode("leaf")
        .select("root", "leaf")
    )

    # Use all clusters and unnest probabilities for Probabilities table
    new_probabilities_df = (
        all_clusters_df.select("cluster_id", "cluster_struct")
        .with_columns(
            pl.col("cluster_struct").struct.field("probability").alias("probability")
        )
        .drop("cluster_struct")
        .with_columns(
            pl.lit(resolution.resolution_id, dtype=pl.Int64).alias("resolution_id")
        )
        .select("resolution_id", "cluster_id", "probability")
        .sort(["cluster_id", "probability"])
    )

    logger.info("Wrangling complete!", prefix=log_prefix)

    return (
        new_clusters_df.to_arrow(),
        new_contains_df.to_arrow(),
        new_probabilities_df.to_arrow(),
    )


def insert_results(
    path: ModelResolutionPath,
    results: pa.Table,
    batch_size: int,
) -> None:
    """Writes a results table to Matchbox.

    The PostgreSQL backend stores clusters in a hierarchical structure, where
    each component references its parent component at a higher threshold.

    This means two-item components are synonymous with their original pairwise
    probabilities.

    This allows easy querying of clusters at any threshold.

    Args:
        path: The path of the model resolution to upload results for
        results: A PyArrow results table with left_id, right_id, probability
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxResolutionNotFoundError: If the specified resolution doesn't exist.
        MatchboxResolutionInvalidData: If data fingerprint conflicts with resolution.
        MatchboxResolutionExistingData: If data was already inserted for resolution.
    """
    log_prefix = f"Model {path.name}"
    if results.num_rows == 0:
        logger.info("Empty results given.", prefix=log_prefix)
        return

    resolution = Resolutions.from_path(path=path, res_type=ResolutionType.MODEL)

    # Check if the content hash is the same
    fingerprint = hash_model_results(results)
    if resolution.fingerprint != fingerprint:
        raise MatchboxResolutionInvalidData

    with MBDB.get_session() as session:
        existing_results = session.execute(
            select(func.count())
            .select_from(Results)
            .where(Results.resolution_id == resolution.resolution_id)
        ).scalar_one()

        if existing_results > 0:
            raise MatchboxResolutionExistingData
    logger.info(
        f"Writing results data with batch size {batch_size:,}", prefix=log_prefix
    )

    clusters, contains, probabilities = _results_to_insert_tables(
        resolution=resolution, probabilities=results
    )

    with MBDB.get_adbc_connection() as adbc_connection:
        try:
            logger.info(
                f"Inserting {clusters.shape[0]:,} results objects", prefix=log_prefix
            )

            large_append(
                data=clusters,
                table_class=Clusters,
                adbc_connection=adbc_connection,
                max_chunksize=batch_size,
            )

            logger.info(
                f"Successfully inserted {clusters.shape[0]:,} rows into Clusters table",
                prefix=log_prefix,
            )

            large_append(
                data=contains,
                table_class=Contains,
                adbc_connection=adbc_connection,
                max_chunksize=batch_size,
            )

            logger.info(
                f"Successfully inserted {contains.shape[0]:,} rows into Contains table",
                prefix=log_prefix,
            )

            large_append(
                data=probabilities,
                table_class=Probabilities,
                adbc_connection=adbc_connection,
                max_chunksize=batch_size,
            )

            logger.info(
                f"Successfully inserted "
                f"{probabilities.shape[0]:,} objects into Probabilities table",
                prefix=log_prefix,
            )

            large_append(
                data=pl.from_arrow(results)
                .with_columns(
                    pl.lit(resolution.resolution_id)
                    .cast(pl.UInt64)
                    .alias("resolution_id")
                )
                .select("resolution_id", "left_id", "right_id", "probability")
                .to_arrow(),
                table_class=Results,
                adbc_connection=adbc_connection,
                max_chunksize=batch_size,
            )

            logger.info(
                f"Successfully inserted {results.shape[0]:,} rows into Results table",
                prefix=log_prefix,
            )

            adbc_connection.commit()

        except Exception as e:
            logger.error(
                f"Failed to insert data, rolling back: {str(e)}", prefix=log_prefix
            )
            adbc_connection.rollback()
            raise

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        Contains.__table__.fullname,
        Probabilities.__table__.fullname,
        Results.__table__.fullname,
    )

    logger.info("Insert operation complete!", prefix=log_prefix)
