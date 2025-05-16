"""Utilities for inserting data into the PostgreSQL backend."""

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ModelResolutionName
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.hash import hash_arrow_table, hash_values
from matchbox.common.logging import logger
from matchbox.common.sources import SourceConfig
from matchbox.common.transform import (
    attach_components_to_probabilities,
    to_hierarchical_clusters,
)
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    PKSpace,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql, large_ingest


class HashIDMap:
    """An object to help map between IDs and hashes.

    When given a set of IDs, returns their hashes. If any ID doesn't have a hash,
    it will error.

    When given a set of hashes, it will return their IDs. If any don't have IDs, it
    will create one and return it as part of the set.

    Args:
        start: The first integer to use for new IDs
        lookup (optional): A lookup table to use for existing hashes
    """

    def __init__(self, start: int | None = None, lookup: pa.Table | None = None):
        """Initialise the HashIDMap object."""
        self.next_int = start
        if not lookup:
            self.lookup = pa.Table.from_arrays(
                [
                    pa.array([], type=pa.uint64()),
                    pa.array([], type=pa.large_binary()),
                    pa.array([], type=pa.bool_()),
                ],
                names=["id", "hash", "new"],
            )
        else:
            new_column = pa.array([False] * lookup.shape[0], type=pa.bool_())
            self.lookup = pa.Table.from_arrays(
                [lookup["id"], lookup["hash"], new_column], names=["id", "hash", "new"]
            )

    def get_hashes(self, ids: pa.UInt64Array) -> pa.LargeBinaryArray:
        """Returns the hashes of the given IDs."""
        indices = pc.index_in(ids, self.lookup["id"])

        if pc.any(pc.is_null(indices)).as_py():
            m_mask = pc.is_null(indices)
            m_ids = pc.filter(ids, m_mask)

            raise ValueError(
                f"The following IDs were not found in lookup table: {m_ids.to_pylist()}"
            )

        return pc.take(self.lookup["hash"], indices)

    def generate_ids(self, hashes: pa.BinaryArray) -> pa.UInt64Array:
        """Returns the IDs of the given hashes, assigning new IDs for unknown hashes."""
        if self.next_int is None:
            raise RuntimeError("`next_int` was unset for HasIDMap")

        indices = pc.index_in(hashes, self.lookup["hash"])
        new_hashes = pc.unique(pc.filter(hashes, pc.is_null(indices)))

        if len(new_hashes) > 0:
            new_ids = pa.array(
                range(self.next_int, self.next_int + len(new_hashes)),
                type=pa.uint64(),
            )

            new_entries = pa.Table.from_arrays(
                [
                    new_ids,
                    new_hashes,
                    pa.array([True] * len(new_hashes), type=pa.bool_()),
                ],
                names=["id", "hash", "new"],
            )

            self.next_int += len(new_hashes)
            self.lookup = pa.concat_tables([self.lookup, new_entries])

            indices = pc.index_in(hashes, self.lookup["hash"])

        return pc.take(self.lookup["id"], indices)


def insert_source(
    source_config: SourceConfig, data_hashes: pa.Table, batch_size: int
) -> None:
    """Indexes a source within Matchbox."""
    log_prefix = f"Index {source_config.address.pretty}"
    content_hash = hash_arrow_table(data_hashes)

    with MBDB.get_session() as session:
        logger.info("Begin", prefix=log_prefix)

        # Check if resolution already exists
        existing_resolution = (
            session.query(Resolutions).filter_by(name=source_config.name).first()
        )

        if existing_resolution:
            resolution = existing_resolution
            # Check if the content hash is the same
            if resolution.hash == content_hash:
                logger.info("Source data matches index. Finished", prefix=log_prefix)
                return
        else:
            # Create new resolution
            resolution = Resolutions(
                name=source_config.name,
                hash=content_hash,
                type=ResolutionNodeType.SOURCE.value,
            )
            session.add(resolution)
            session.flush()

        # Store resolution ID for later use
        resolution_id: int = resolution.resolution_id

        # Check if source already exists
        existing_source = (
            session.query(SourceConfigs)
            .filter_by(resolution_id=resolution.resolution_id)
            .first()
        )

        if existing_source:
            logger.info("Deleting existing", prefix=log_prefix)
            session.delete(existing_source)
            session.flush()

        # Create new source with relationship to resolution
        source_obj = SourceConfigs.from_dto(resolution, source_config)
        session.add(source_obj)
        session.commit()

        logger.info(
            "Added to Resolutions, SourceConfigs, SourceFields", prefix=log_prefix
        )

        # Store source_config_id and max primary keys for later use
        source_config_id = source_obj.source_config_id

    # Don't insert new hashes, but new keys need existing hash IDs
    with MBDB.get_adbc_connection() as conn:
        existing_hash_lookup: pl.DataFrame = sql_to_df(
            stmt=compile_sql(select(Clusters.cluster_id, Clusters.cluster_hash)),
            connection=conn,
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
    try:
        # Bulk insert into Clusters table (only new clusters)
        if not cluster_records.is_empty():
            large_ingest(
                data=cluster_records.to_arrow(),
                table_class=Clusters,
                max_chunksize=batch_size,
            )
            logger.info(
                f"Added {len(cluster_records):,} objects to Clusters table",
                prefix=log_prefix,
            )

        # Bulk insert into ClusterSourceKey table (all links)
        if not keys_records.is_empty():
            large_ingest(
                data=keys_records.to_arrow(),
                table_class=ClusterSourceKey,
                max_chunksize=batch_size,
            )
            logger.info(
                f"Added {len(keys_records):,} primary keys to ClusterSourceKey table",
                prefix=log_prefix,
            )
    except IntegrityError as e:
        # Log the error and rollback
        logger.warning(f"Error, rolling back: {e}", prefix=log_prefix)
        conn.rollback()

    # Insert successful, safe to update the resolution's content hash
    with MBDB.get_session() as session:
        if not keys_records.is_empty():
            stmt = (
                update(Resolutions)
                .where(Resolutions.resolution_id == resolution_id)
                .values(hash=content_hash)
            )
            session.execute(stmt)

    if cluster_records.is_empty() and keys_records.is_empty():
        logger.info("No new records to add", prefix=log_prefix)

    logger.info("Finished", prefix=log_prefix)


def insert_model(
    name: ModelResolutionName,
    left: Resolutions,
    right: Resolutions,
    description: str,
) -> None:
    """Writes a model to Matchbox with a default truth value of 100.

    Args:
        name: Name of the new model
        left: Left parent of the model
        right: Right parent of the model. Same as left in a dedupe job
        description: Model description

    Raises:
        MatchboxResolutionNotFoundError: If the specified parent models don't exist.
    """
    log_prefix = f"Model {name}"
    logger.info("Registering", prefix=log_prefix)
    with MBDB.get_session() as session:
        # Check if resolution exists
        exists_stmt = select(Resolutions).where(Resolutions.name == name)
        exists_obj = session.scalar(exists_stmt)

        if exists_obj is not None:
            logger.info("Model resolution already exists", prefix=log_prefix)
            return
        else:
            new_res = Resolutions(
                type=ResolutionNodeType.MODEL.value,
                name=name,
                description=description,
                truth=100,
            )
            session.add(new_res)
            session.flush()

            def _create_closure_entries(parent_resolution: Resolutions) -> None:
                """Create closure entries for the new model.

                This is made up of mappings between nodes and any of their direct or
                indirect parents.
                """
                session.add(
                    ResolutionFrom(
                        parent=parent_resolution.resolution_id,
                        child=new_res.resolution_id,
                        level=1,
                        truth_cache=parent_resolution.truth,
                    )
                )

                ancestor_entries = (
                    session.query(ResolutionFrom)
                    .filter(ResolutionFrom.child == parent_resolution.resolution_id)
                    .all()
                )

                for entry in ancestor_entries:
                    session.add(
                        ResolutionFrom(
                            parent=entry.parent,
                            child=new_res.resolution_id,
                            level=entry.level + 1,
                            truth_cache=entry.truth_cache,
                        )
                    )

            # Create resolution lineage entries
            _create_closure_entries(parent_resolution=left)

            if right != left:
                _create_closure_entries(parent_resolution=right)
            status = "Inserted new"
            resolution_id = new_res.resolution_id

        session.commit()

    logger.info(f"{status} model with ID {resolution_id}", prefix=log_prefix)
    logger.info("Done!", prefix=log_prefix)


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
            {"parent": [], "child": []},
            schema=pa.schema([("parent", pa.uint64()), ("child", pa.uint64())]),
        )
        probabilities = pa.table(
            {"resolution": [], "cluster": [], "probability": []},
            schema=pa.schema(
                [
                    ("resolution", pa.uint64()),
                    ("cluster", pa.uint64()),
                    ("probability", pa.uint8()),
                ]
            ),
        )
        return clusters, contains, probabilities

    logger.info("Wrangling data to insert tables", prefix=log_prefix)

    # Create ID-Hash lookup for existing probabilities
    with MBDB.get_adbc_connection() as conn:
        lookup = sql_to_df(
            stmt=compile_sql(
                select(
                    Clusters.cluster_hash.label("hash"), Clusters.cluster_id.label("id")
                )
            ),
            connection=conn,
            return_type="arrow",
        )
    lookup = lookup.cast(pa.schema([("hash", pa.large_binary()), ("id", pa.uint64())]))

    hm = HashIDMap(lookup=lookup)

    # Join hashes, probabilities and components
    logger.debug("Attaching components to hashes", prefix=log_prefix)

    probs_with_ccs = attach_components_to_probabilities(
        pa.table(
            {
                "left_id": hm.get_hashes(probabilities["left_id"]),
                "right_id": hm.get_hashes(probabilities["right_id"]),
                "probability": probabilities["probability"],
            }
        )
    )

    # Calculate hierarchies
    logger.debug("Computing hierarchies", prefix=log_prefix)

    hierarchy = to_hierarchical_clusters(
        probabilities=probs_with_ccs,
        hash_func=hash_values,
        dtype=pa.large_binary,
    )

    # Determine number of new IDs to generate
    referenced_hashes = pc.unique(
        pa.concat_arrays(
            [hierarchy["parent"].combine_chunks(), hierarchy["parent"].combine_chunks()]
        )
    )
    indices_old_hashes = pc.index_in(referenced_hashes, lookup["hash"])
    num_new_hashes = pc.sum(pc.is_null(indices_old_hashes)).as_py()

    if num_new_hashes:
        hm.next_int = PKSpace.reserve_block(table="clusters", block_size=num_new_hashes)
    else:
        # No new hashes means no new cluster IDs, so next_int won't matter
        hm.next_int = 0

    # Create Probabilities Arrow table to insert, containing all generated probabilities
    logger.debug("Filtering to target table shapes", prefix=log_prefix)

    probabilities = pa.table(
        {
            "resolution": pa.array(
                [resolution.resolution_id] * hierarchy.shape[0],
                type=pa.uint64(),
            ),
            "cluster": hm.generate_ids(hierarchy["parent"]),
            "probability": hierarchy["probability"],
        }
    )

    # Probabilities will have duplicates because hierarchy tracks all parent-child edges
    probabilities = pl.from_arrow(probabilities).unique().to_arrow()

    # Create Clusters Arrow table to insert, containing only new clusters
    new_hashes = pc.filter(hm.lookup["hash"], hm.lookup["new"])
    clusters = pa.table(
        {
            "cluster_id": pc.filter(hm.lookup["id"], hm.lookup["new"]),
            "cluster_hash": new_hashes,
        }
    )

    # Create Contains Arrow table to insert, containing only new contains edges
    # Recall that clusters are defined by their parents, so all existing clusters
    # already have the same parent-child relationships as were calculated here
    hierarchy_new = hierarchy.filter(
        pa.compute.is_in(hierarchy["parent"], value_set=new_hashes)
    )
    hierarchy_new = pl.from_arrow(hierarchy_new).unique().to_arrow()

    contains = pa.table(
        {
            "parent": hm.generate_ids(hierarchy_new["parent"]),
            "child": hm.generate_ids(hierarchy_new["child"]),
        }
    )

    logger.info("Wrangling complete!", prefix=log_prefix)

    return clusters, contains, probabilities


def insert_results(
    resolution: Resolutions,
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
        resolution: Resolution of type model to associate results with
        results: A PyArrow results table with left_id, right_id, probability
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxResolutionNotFoundError: If the specified model doesn't exist.
    """
    log_prefix = f"Model {resolution.name}"
    logger.info(
        f"Writing results data with batch size {batch_size:,}", prefix=log_prefix
    )

    # Check if the content hash is the same
    content_hash = hash_arrow_table(results)
    if resolution.hash == content_hash:
        logger.info("Results already uploaded. Finished", prefix=log_prefix)
        return

    clusters, contains, probabilities = _results_to_insert_tables(
        resolution=resolution, probabilities=results
    )

    with MBDB.get_session() as session:
        try:
            # Clear existing probabilities for this resolution
            stmt = delete(Probabilities).where(
                Probabilities.resolution == resolution.resolution_id
            )
            session.execute(stmt)

            session.commit()
            logger.info("Removed old probabilities", prefix=log_prefix)

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(
                f"Failed to clear old probabilities or update content hash: {str(e)}",
                prefix=log_prefix,
            )
            raise

    try:
        logger.info(
            f"Inserting {clusters.shape[0]:,} results objects", prefix=log_prefix
        )

        large_ingest(
            data=clusters,
            table_class=Clusters,
            max_chunksize=batch_size,
        )

        logger.info(
            f"Successfully inserted {clusters.shape[0]:,} objects into Clusters table",
            prefix=log_prefix,
        )

        large_ingest(
            data=contains,
            table_class=Contains,
            max_chunksize=batch_size,
        )

        logger.info(
            f"Successfully inserted {contains.shape[0]:,} objects into Contains table",
            prefix=log_prefix,
        )

        large_ingest(
            data=probabilities,
            table_class=Probabilities,
            max_chunksize=batch_size,
        )

        logger.info(
            f"Successfully inserted "
            f"{probabilities.shape[0]:,} objects into Probabilities table",
            prefix=log_prefix,
        )

    except SQLAlchemyError as e:
        logger.error(f"Failed to insert data: {str(e)}", prefix=log_prefix)
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
