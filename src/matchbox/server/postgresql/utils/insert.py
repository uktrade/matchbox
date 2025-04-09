"""Utilities for inserting data into the PostgreSQL backend."""

import pyarrow as pa
import pyarrow.compute as pc
from sqlalchemy import Engine, delete, exists, select, union
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select

from matchbox.common.db import sql_to_df
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.hash import hash_data, hash_values
from matchbox.common.logging import logger
from matchbox.common.sources import Source
from matchbox.common.transform import (
    attach_components_to_probabilities,
    to_hierarchical_clusters,
)
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourcePK,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceColumns,
    Sources,
)
from matchbox.server.postgresql.utils.db import batch_ingest, compile_sql


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

    def __init__(self, start: int, lookup: pa.Table = None):
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

    def get_ids(self, hashes: pa.BinaryArray) -> pa.UInt64Array:
        """Returns the IDs of the given hashes, assigning new IDs for unknown hashes."""
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


def insert_dataset(source: Source, data_hashes: pa.Table, batch_size: int) -> None:
    """Indexes a dataset from your data warehouse within Matchbox."""
    log_prefix = f"Index {source.address.pretty}"
    resolution_hash = hash_data(str(source.address))
    engine = MBDB.get_engine()
    with Session(engine) as session:
        logger.info("Begin", prefix=log_prefix)

        # Check if resolution already exists
        existing_resolution = (
            session.query(Resolutions).filter_by(name=source.resolution_name).first()
        )

        if existing_resolution:
            resolution = existing_resolution
        else:
            # Create new resolution
            resolution = Resolutions(
                resolution_id=Resolutions.next_id(),
                name=source.resolution_name,
                resolution_hash=resolution_hash,
                type=ResolutionNodeType.DATASET.value,
            )
            session.add(resolution)

        # Check if source already exists
        existing_source = (
            session.query(Sources)
            .filter_by(resolution_id=resolution.resolution_id)
            .first()
        )

        if existing_source:
            logger.info("Deleting existing", prefix=log_prefix)
            session.delete(existing_source)
            session.flush()

        # Create new source with relationship to resolution
        source_obj = Sources(
            resolution_id=resolution.resolution_id,
            resolution_name=source.resolution_name,
            full_name=source.address.full_name,
            warehouse_hash=source.address.warehouse_hash,
            db_pk=source.db_pk,
        )

        # Add columns directly through the relationship
        for idx, column in enumerate(source.columns):
            source_column = SourceColumns(
                source_id=source_obj.source_id,
                column_index=idx,
                column_name=column.name,
                column_type=column.type,
            )
            source_obj.columns.append(source_column)

        session.add(source_obj)
        session.commit()

        logger.info("Added to Resolutions, Sources, SourceColumns", prefix=log_prefix)

        # Store source_id and max primary keys for later use
        source_id = source_obj.source_id
        next_cluster_id = Clusters.next_id()
        next_pk_id = ClusterSourcePK.next_id()

    # Don't insert new hashes, but new PKs need existing hash IDs
    with MBDB.get_adbc_connection() as conn:
        existing_hash_lookup = sql_to_df(
            stmt=compile_sql(select(Clusters.cluster_id, Clusters.cluster_hash)),
            connection=conn,
            return_type="arrow",
        )

    # Create a dictionary for faster lookups
    hash_to_id = {}
    if len(existing_hash_lookup) > 0:
        for i in range(len(existing_hash_lookup)):
            hash_bytes = existing_hash_lookup["cluster_hash"][i].as_py()
            cluster_id = existing_hash_lookup["cluster_id"][i].as_py()
            hash_to_id[hash_bytes] = cluster_id

    # Prepare records for both tables - new clusters and links
    cluster_records = []
    source_pk_records = []
    pk_id_counter = next_pk_id

    for clus in data_hashes.to_pylist():
        hash_bytes = clus["hash"]

        # Check if this hash already exists in the database
        if hash_bytes in hash_to_id:
            # Use existing cluster_id
            cluster_id = hash_to_id[hash_bytes]
        else:
            # Create a new cluster
            cluster_id = next_cluster_id + len(cluster_records)
            cluster_records.append((cluster_id, hash_bytes))

        # Add all source primary keys linking to this cluster
        for pk in clus["source_pk"]:
            source_pk_records.append(
                (
                    pk_id_counter,  # pk_id
                    cluster_id,  # cluster_id
                    source_id,  # source_id
                    pk,  # source_pk
                )
            )
            pk_id_counter += 1

    # Insert new clusters and all source primary keys
    try:
        with engine.connect() as conn:
            # Bulk insert into Clusters table (only new clusters)
            if cluster_records:
                batch_ingest(
                    records=cluster_records,
                    table=Clusters,
                    conn=conn,
                    batch_size=batch_size,
                )
                logger.info(
                    f"Added {len(cluster_records):,} objects to Clusters table",
                    prefix=log_prefix,
                )

            # Bulk insert into ClusterSourcePK table (all links)
            if source_pk_records:
                batch_ingest(
                    records=source_pk_records,
                    table=ClusterSourcePK,
                    conn=conn,
                    batch_size=batch_size,
                )
                logger.info(
                    f"Added {len(source_pk_records):,} primary keys to "
                    "ClusterSourcePK table",
                    prefix=log_prefix,
                )

            # Commit both inserts in a single transaction
            conn.commit()
    except IntegrityError as e:
        # Log the error and rollback
        logger.warning(f"Error, rolling back: {e}", prefix=log_prefix)
        conn.rollback()

    if not cluster_records and not source_pk_records:
        logger.info("No new records to add", prefix=log_prefix)

    logger.info("Finished", prefix=log_prefix)


def insert_model(
    model: str,
    left: Resolutions,
    right: Resolutions,
    description: str,
    engine: Engine,
) -> None:
    """Writes a model to Matchbox with a default truth value of 100.

    Args:
        model: Name of the new model
        left: Left parent of the model
        right: Right parent of the model. Same as left in a dedupe job
        description: Model description
        engine: SQLAlchemy engine instance

    Raises:
        MatchboxResolutionNotFoundError: If the specified parent models don't exist.
        MatchboxResolutionNotFoundError: If the specified model doesn't exist.
    """
    log_prefix = f"Model {model}"
    logger.info("Registering", prefix=log_prefix)
    with Session(engine) as session:
        resolution_hash = hash_values(
            left.resolution_hash,
            right.resolution_hash,
            bytes(model, encoding="utf-8"),
        )

        # Check if resolution exists
        exists_stmt = select(Resolutions).where(
            Resolutions.resolution_hash == resolution_hash
        )
        exists_obj = session.scalar(exists_stmt)
        exists = exists_obj is not None
        resolution_id = (
            Resolutions.next_id() if not exists else exists_obj.resolution_id
        )

        # Upsert new resolution
        stmt = (
            insert(Resolutions)
            .values(
                resolution_id=resolution_id,
                resolution_hash=resolution_hash,
                type=ResolutionNodeType.MODEL.value,
                name=model,
                description=description,
                truth=100,
            )
            .on_conflict_do_update(
                index_elements=["resolution_hash"],
                set_={"name": model, "description": description},
            )
        )

        session.execute(stmt)

        if not exists:

            def _create_closure_entries(parent_resolution: Resolutions) -> None:
                """Create closure entries for the new model.

                This is made up of mappings between nodes and any of their direct or
                indirect parents.
                """
                session.add(
                    ResolutionFrom(
                        parent=parent_resolution.resolution_id,
                        child=resolution_id,
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
                            child=resolution_id,
                            level=entry.level + 1,
                            truth_cache=entry.truth_cache,
                        )
                    )

            # Create resolution lineage entries
            _create_closure_entries(parent_resolution=left)

            if right != left:
                _create_closure_entries(parent_resolution=right)

        session.commit()

    status = "Inserted new" if not exists else "Updated existing"
    logger.info(f"{status} model with ID {resolution_id}", prefix=log_prefix)
    logger.info("Done!", prefix=log_prefix)


def _get_resolution_related_clusters(resolution_id: int) -> Select:
    """Get cluster hashes and IDs for a resolution, its parents, and siblings.

    * When a parent is a dataset, retrieves the data via the Sources table.
    * When a parent is a model, retrieves the data via the Probabilities table.

    This corresponds to all possible existing clusters that a resolution might ever be
    able to link together, or propose.

    Args:
        resolution_id: The ID of the resolution to query

    Returns:
        List of tuples containing (cluster_hash, cluster_id)
    """
    direct_resolution = select(Resolutions.resolution_id).where(
        Resolutions.resolution_id == resolution_id
    )

    parent_resolutions = select(ResolutionFrom.parent).where(
        ResolutionFrom.child == resolution_id
    )

    sibling_resolutions = (
        select(ResolutionFrom.child)
        .where(
            ResolutionFrom.parent.in_(
                select(ResolutionFrom.parent).where(
                    ResolutionFrom.child == resolution_id
                )
            )
        )
        .where(ResolutionFrom.child != resolution_id)
    )

    resolution_set = union(
        direct_resolution, parent_resolutions, sibling_resolutions
    ).cte("resolution_set")

    # Main query
    base_query = (
        select(Clusters.cluster_hash.label("hash"), Clusters.cluster_id.label("id"))
        .distinct()
        .select_from(Clusters)
        .join(Probabilities, Probabilities.cluster == Clusters.cluster_id, isouter=True)
    )

    # Subquery for source datasets
    source_datasets = (
        select(ClusterSourcePK.cluster_id)
        .join(Sources, Sources.resolution_id == ClusterSourcePK.source_id)
        .where(Sources.resolution_id.in_(select(resolution_set.c.resolution_id)))
    )

    # Subquery for model resolutions
    model_resolutions = select(resolution_set.c.resolution_id).where(
        ~exists()
        .select_from(Sources)
        .where(Sources.resolution_id == resolution_set.c.resolution_id)
    )

    # Combine conditions
    final_query = base_query.where(
        (Clusters.cluster_id.in_(source_datasets))
        | (Probabilities.resolution.in_(model_resolutions))
    )

    return final_query


def _results_to_insert_tables(
    resolution: Resolutions, probabilities: pa.Table, engine: Engine
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Takes probabilities and returns three Arrow tables that can be inserted exactly.

    Returns:
        A tuple containing:

            * A Clusters update Arrow table
            * A Contains update Arrow table
            * A Probabilities update Arrow table
    """
    log_prefix = f"Model {resolution.name}"
    logger.info("Wrangling data to insert tables", prefix=log_prefix)

    # Create ID-Hash lookup for existing probabilities
    with MBDB.get_adbc_connection() as conn:
        lookup = sql_to_df(
            stmt=compile_sql(
                _get_resolution_related_clusters(resolution.resolution_id)
            ),
            connection=conn,
            return_type="arrow",
        )
    lookup = lookup.cast(pa.schema([("hash", pa.large_binary()), ("id", pa.uint64())]))

    hm = HashIDMap(start=Clusters.next_id(), lookup=lookup)

    # Join hashes, probabilities and components
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
    hierarchy = to_hierarchical_clusters(
        probabilities=probs_with_ccs,
        hash_func=hash_values,
        dtype=pa.large_binary,
    )

    # Create Probabilities Arrow table to insert, containing all generated probabilities
    probabilities = pa.table(
        {
            "resolution": pa.array(
                [resolution.resolution_id] * hierarchy.shape[0],
                type=pa.uint64(),
            ),
            "cluster": hm.get_ids(hierarchy["parent"]),
            "probability": hierarchy["probability"],
        }
    )

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
    contains = pa.table(
        {
            "parent": hm.get_ids(hierarchy_new["parent"]),
            "child": hm.get_ids(hierarchy_new["child"]),
        }
    )

    logger.info("Wrangling complete!", prefix=log_prefix)

    return clusters, contains, probabilities


def insert_results(
    resolution: Resolutions,
    engine: Engine,
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
        engine: SQLAlchemy engine instance
        results: A PyArrow results table with left_id, right_id, probability
        batch_size: Number of records to insert in each batch

    Raises:
        MatchboxResolutionNotFoundError: If the specified model doesn't exist.
    """
    log_prefix = f"Model {resolution.name}"
    logger.info(
        f"Writing results data with batch size {batch_size:,}", prefix=log_prefix
    )

    clusters, contains, probabilities = _results_to_insert_tables(
        resolution=resolution, probabilities=results, engine=engine
    )

    with Session(engine) as session:
        try:
            # Clear existing probabilities for this resolution
            session.execute(
                delete(Probabilities).where(
                    Probabilities.resolution == resolution.resolution_id
                )
            )

            session.commit()
            logger.info("Removed old probabilities", prefix=log_prefix)

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(
                f"Failed to clear old probabilities: {str(e)}", prefix=log_prefix
            )
            raise

    with engine.connect() as conn:
        try:
            logger.info(
                f"Inserting {clusters.shape[0]:,} results objects", prefix=log_prefix
            )

            batch_ingest(
                records=[tuple(c.values()) for c in clusters.to_pylist()],
                table=Clusters,
                conn=conn,
                batch_size=batch_size,
            )

            logger.info(
                f"Successfully inserted {clusters.shape[0]:,} "
                "objects into Clusters table",
                prefix=log_prefix,
            )

            batch_ingest(
                records=[tuple(c.values()) for c in contains.to_pylist()],
                table=Contains,
                conn=conn,
                batch_size=batch_size,
            )

            logger.info(
                f"Successfully inserted {contains.shape[0]:,} "
                "objects into Contains table",
                prefix=log_prefix,
            )

            batch_ingest(
                records=[tuple(c.values()) for c in probabilities.to_pylist()],
                table=Probabilities,
                conn=conn,
                batch_size=batch_size,
            )

            logger.info(
                f"Successfully inserted "
                f"{probabilities.shape[0]:,} objects into Probabilities table",
                prefix=log_prefix,
            )

        except SQLAlchemyError as e:
            logger.error(f"Failed to insert data: {str(e)}", prefix=log_prefix)
            raise

    logger.info("Insert operation complete!", prefix=log_prefix)
