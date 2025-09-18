"""Utilities for inserting data into the PostgreSQL backend."""

from collections.abc import Iterator

import polars as pl
import pyarrow as pa
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.exc import SQLAlchemyError

from matchbox.common.db import sql_to_df
from matchbox.common.graph import (
    ModelResolutionName,
    ResolutionType,
    SourceResolutionName,
)
from matchbox.common.hash import IntMap, hash_arrow_table
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
)
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    ingest_to_temporary_table,
    large_append,
)
from matchbox.server.postgresql.utils.query import get_parent_clusters_and_leaves


def insert_hashes(
    name: SourceResolutionName,
    data_hashes: pa.Table,
    batch_size: int,
) -> None:
    """Indexes hash data for a source within Matchbox.

    Args:
        name: The name of the source resolution
        data_hashes: Arrow table containing hash data
        batch_size: Batch size for bulk operations
    """
    log_prefix = f"Index hashes {name}"
    content_hash = hash_arrow_table(data_hashes)

    with MBDB.get_session() as session:
        resolution = Resolutions.from_name(
            name=name, res_type=ResolutionType.SOURCE, session=session
        )
        # Check if the content hash is the same
        if resolution.hash == content_hash:
            logger.info("Source data matches index. Finished", prefix=log_prefix)
            return

        resolution_id = resolution.resolution_id
        source_config_id = resolution.source_config.source_config_id

    # Don't insert new hashes, but new keys need existing hash IDs
    with MBDB.get_adbc_connection() as conn:
        existing_hash_lookup: pl.DataFrame = sql_to_df(
            stmt=compile_sql(select(Clusters.cluster_id, Clusters.cluster_hash)),
            connection=conn.dbapi_connection,
            return_type="polars",
        )

    data_hashes: pl.DataFrame = pl.from_arrow(data_hashes)
    if existing_hash_lookup.is_empty():
        new_hashes = data_hashes.select("hash").unique()
    else:
        new_hashes = (
            data_hashes.select("hash")
            .unique()
            .join(
                other=existing_hash_lookup,
                left_on="hash",
                right_on="cluster_hash",
                how="anti",
            )
        )

    if new_hashes.shape[0]:
        # Create new cluster records with sequential IDs
        next_cluster_id = PKSpace.reserve_block("clusters", len(new_hashes))
    else:
        # The value of next_cluster_id is irrelevant as cluster_records will be empty
        next_cluster_id = 0

    cluster_records = (
        new_hashes.with_row_index("cluster_id")
        .with_columns(
            [
                (pl.col("cluster_id") + next_cluster_id)
                .cast(pl.Int64)
                .alias("cluster_id")
            ]
        )
        .rename({"hash": "cluster_hash"})
    )

    # Create a combined lookup with both existing and new mappings
    lookup = pl.concat([existing_hash_lookup, cluster_records], how="vertical")

    # Add cluster_id values to data hashes
    hashes_with_ids = data_hashes.join(lookup, left_on="hash", right_on="cluster_hash")

    # Explode keys
    keys_records = (
        hashes_with_ids.select(["cluster_id", "keys"])
        .explode("keys")
        .rename({"keys": "key"})
    )

    if keys_records.shape[0] > 0:
        next_key_id = PKSpace.reserve_block("cluster_keys", len(keys_records))
    else:
        # The next_key_id is irrelevant if we don't write any keys records
        next_key_id = 0

    # Add required columns
    keys_records = keys_records.with_row_index("key_id").with_columns(
        [
            (pl.col("key_id") + next_key_id).alias("key_id"),
            pl.lit(source_config_id, dtype=pl.Int64).alias("source_config_id"),
        ]
    )

    # Insert new clusters and all source primary keys
    with MBDB.get_adbc_connection() as adbc_connection:
        try:
            # Bulk insert into Clusters table (only new clusters)
            if not cluster_records.is_empty():
                large_append(
                    data=cluster_records.to_arrow(),
                    table_class=Clusters,
                    adbc_connection=adbc_connection,
                    max_chunksize=batch_size,
                )
                logger.info(
                    f"Added {len(cluster_records):,} objects to Clusters table",
                    prefix=log_prefix,
                )

            # Bulk insert into ClusterSourceKey table (all links)
            if not keys_records.is_empty():
                large_append(
                    data=keys_records.to_arrow(),
                    table_class=ClusterSourceKey,
                    adbc_connection=adbc_connection,
                    max_chunksize=batch_size,
                )
                logger.info(
                    f"Added {len(keys_records):,} PKs to ClusterSourceKey table",
                    prefix=log_prefix,
                )

            adbc_connection.commit()

        except Exception as e:
            # Log the error and rollback
            logger.warning(f"Error, rolling back: {e}", prefix=log_prefix)
            adbc_connection.rollback()
            raise

    # Insert successful, safe to update the resolution's content hash
    with MBDB.get_session() as session:
        if not keys_records.is_empty():
            stmt = (
                update(Resolutions)
                .where(Resolutions.resolution_id == resolution_id)
                .values(hash=content_hash)
            )
            session.execute(stmt)
            session.commit()

    if cluster_records.is_empty() and keys_records.is_empty():
        logger.info("No new records to add", prefix=log_prefix)

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

    # Look up existing clusters in the database
    with ingest_to_temporary_table(
        table_name="hashes",
        schema_name="mb",
        column_types={
            "cluster_hash": BYTEA,
        },
        data=all_clusters_df.select("cluster_hash").to_arrow(),
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
    name: ModelResolutionName,
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
        name: The name of the model resolution to upload results for
        results: A PyArrow results table with left_id, right_id, probability
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxResolutionNotFoundError: If the specified model doesn't exist.
    """
    resolution = Resolutions.from_name(name=name, res_type=ResolutionType.MODEL)

    log_prefix = f"Model {resolution.name}"
    logger.info(
        f"Writing results data with batch size {batch_size:,}", prefix=log_prefix
    )

    # Check if the content hash is the same
    content_hash = hash_arrow_table(results, as_sorted_list=["left_id", "right_id"])
    if resolution.hash == content_hash:
        logger.info("Results already uploaded. Finished", prefix=log_prefix)
        return

    clusters, contains, probabilities = _results_to_insert_tables(
        resolution=resolution, probabilities=results
    )

    with MBDB.get_session() as session:
        try:
            # Clear existing probabilities and results for this resolution
            stmt = delete(Probabilities).where(
                Probabilities.resolution_id == resolution.resolution_id
            )
            session.execute(stmt)

            stmt = delete(Results).where(
                Results.resolution_id == resolution.resolution_id
            )
            session.execute(stmt)

            session.commit()
            logger.info("Removed old probabilities and results", prefix=log_prefix)

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(
                "Failed to clear old probabilities and results "
                f"or update content hash: {str(e)}",
                prefix=log_prefix,
            )
            raise

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

    # Insert successful, safe to update the resolution's content hash
    with MBDB.get_session() as session:
        stmt = (
            update(Resolutions)
            .where(Resolutions.resolution_id == resolution.resolution_id)
            .values(hash=content_hash)
        )
        session.execute(stmt)
        session.commit()

    logger.info("Insert operation complete!", prefix=log_prefix)
