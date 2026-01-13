"""Utilities for inserting data into the PostgreSQL backend."""

from collections.abc import Iterator

import polars as pl
import pyarrow as pa
from sqlalchemy import exists, func, join, literal, select, update
from sqlalchemy.dialects.postgresql import (
    ARRAY,
    BIGINT,
    BOOLEAN,
    BYTEA,
    SMALLINT,
    TEXT,
    insert,
)

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
    Probabilities,
    Resolutions,
    Results,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import (
    ingest_to_temporary_table,
)
from matchbox.server.postgresql.utils.query import get_parent_clusters_and_leaves

MODEL_CLUSTERS_SCHEMA = {
    "cluster_id": pl.Int64,
    "cluster_hash": pl.Binary,
    "leaves": pl.List(pl.Int64),
    "probability": pl.Int8,
}


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
            # Add clusters
            new_hashes = (
                select(incoming.c.hash)
                .distinct()
                .where(
                    ~exists(select(1).where(Clusters.cluster_hash == incoming.c.hash))
                )
            )

            stmt_insert_clusters = (
                insert(Clusters)
                .from_select(["cluster_hash"], new_hashes)
                .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
            )

            result = session.execute(stmt_insert_clusters)
            logger.info(
                f"Will add {result.rowcount:,} entries to Clusters table",
                prefix=log_prefix,
            )
            session.flush()

            # Add source keys
            exploded = select(
                Clusters.cluster_id,
                literal(source_config_id, BIGINT).label("source_config_id"),
                func.unnest(incoming.c["keys"]).label("key"),
            ).select_from(
                incoming.join(Clusters, Clusters.cluster_hash == incoming.c.hash)
            )

            stmt_insert_keys = insert(ClusterSourceKey).from_select(
                ["cluster_id", "source_config_id", "key"],
                exploded,
            )

            result = session.execute(stmt_insert_keys)
            logger.info(
                f"Will add {result.rowcount:,} entries to ClusterSourceKey table",
                prefix=log_prefix,
            )
            session.commit()

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
) -> pl.DataFrame:
    """Build cluster hierarchy using disjoint sets and probability thresholding.

    Args:
        cluster_lookup: Dictionary mapping cluster IDs to Cluster objects
        probabilities: Arrow table containing probability data

    Returns:
        DataFrame with MODEL_CLUSTERS_SCHEMA
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
    return pl.DataFrame(
        [
            {
                "cluster_hash": cluster.hash,
                "cluster_id": cluster.id,
                "probability": cluster.probability,
                "leaves": [leaf.id for leaf in cluster.leaves]
                if cluster.leaves
                else [],
            }
            for cluster in all_clusters.values()
        ],
        schema=MODEL_CLUSTERS_SCHEMA,
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

    # Get a cluster lookup dictionary based on the resolution's parents
    im = IntMap()
    nested_data = get_parent_clusters_and_leaves(resolution=resolution)
    cluster_lookup: dict[int, Cluster] = _build_cluster_objects(nested_data, im)

    logger.debug("Computing hierarchies", prefix=log_prefix)
    cluster_df = _build_cluster_hierarchy(
        cluster_lookup=cluster_lookup, probabilities=results
    )
    del cluster_lookup

    logger.debug("Ingesting clusters dataframe", prefix=log_prefix)
    with (
        MBDB.get_session() as session,
        ingest_to_temporary_table(
            table_name="model_results",
            schema_name="mb",
            column_types={
                "left_id": BIGINT(),
                "right_id": BIGINT(),
                "probability": SMALLINT(),
            },
            data=results,
            max_chunksize=batch_size,
        ) as new_results,
        ingest_to_temporary_table(
            table_name="model_clusters",
            schema_name="mb",
            column_types={
                "cluster_id": BIGINT(),
                "cluster_hash": BYTEA(),
                "leaves": ARRAY(BIGINT),
                "probability": SMALLINT(),
                "is_new": BOOLEAN(),
            },
            data=cluster_df.with_columns(pl.lit(True).alias("is_new")).to_arrow(),
            max_chunksize=batch_size,
        ) as new_clusters,
    ):
        try:
            # Labelling new clusters is technically unnecessary but theory is that
            # in our case it will be faster than a lot of ignored conflicts
            session.execute(
                update(new_clusters).values(
                    is_new=~exists(
                        select(1).where(
                            Clusters.cluster_hash == new_clusters.c.cluster_hash
                        )
                    )
                )
            )
            session.flush()
            # Clusters
            cluster_select = (
                select(new_clusters.c.cluster_hash)
                .distinct()
                .where(new_clusters.c.is_new)
            )
            cluster_results = session.execute(
                insert(Clusters)
                .from_select(["cluster_hash"], cluster_select)
                .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
            )
            logger.info(
                f"Will add {cluster_results.rowcount:,} entries to Clusters table",
                prefix=log_prefix,
            )
            session.flush()

            # Contains
            contains_select = (
                select(
                    Clusters.cluster_id.label("root"),
                    func.unnest(new_clusters.c.leaves).label("leaf"),
                )
                .select_from(
                    new_clusters.join(
                        Clusters, Clusters.cluster_hash == new_clusters.c.cluster_hash
                    )
                )
                .where(new_clusters.c.is_new)
            )
            contains_results = session.execute(
                insert(Contains)
                .from_select(["root", "leaf"], contains_select)
                .on_conflict_do_nothing(index_elements=[Contains.root, Contains.leaf])
            )
            logger.info(
                f"Will add {contains_results.rowcount:,} entries to Contains table",
                prefix=log_prefix,
            )

            # Probabilities
            prob_select = select(
                literal(resolution.resolution_id, BIGINT).label("resolution_id"),
                Clusters.cluster_id,
                new_clusters.c.probability,
            ).select_from(
                new_clusters.join(
                    Clusters, Clusters.cluster_hash == new_clusters.c.cluster_hash
                )
            )
            prob_res = session.execute(
                insert(Probabilities).from_select(
                    ["resolution_id", "cluster_id", "probability"], prob_select
                )
            )
            logger.info(
                f"Will add {prob_res.rowcount:,} entries to Probabilities table",
                prefix=log_prefix,
            )

            # Results
            results_select = select(
                literal(resolution.resolution_id, BIGINT).label("resolution_id"),
                new_results.c.left_id,
                new_results.c.right_id,
                new_results.c.probability,
            )
            results_res = session.execute(
                insert(Results).from_select(
                    ["resolution_id", "left_id", "right_id", "probability"],
                    results_select,
                )
            )
            logger.info(
                f"Will add {results_res.rowcount:,} entries to Results table",
                prefix=log_prefix,
            )
            session.commit()

        except Exception as e:
            logger.error(
                f"Failed to insert data, rolling back: {str(e)}", prefix=log_prefix
            )
            session.rollback()
            raise

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        Contains.__table__.fullname,
        Probabilities.__table__.fullname,
        Results.__table__.fullname,
    )

    logger.info("Insert operation complete!", prefix=log_prefix)
