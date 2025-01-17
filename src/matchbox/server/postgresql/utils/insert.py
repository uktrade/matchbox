import logging

import pyarrow as pa
import pyarrow.compute as pc
from sqlalchemy import Engine, delete, exists, select, union
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select

from matchbox.client.results import Results
from matchbox.common.db import sql_to_df
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.hash import hash_values
from matchbox.common.sources import Source
from matchbox.common.transform import (
    attach_components_to_probabilities,
    to_hierarchical_clusters,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Sources,
)
from matchbox.server.postgresql.utils.db import batch_ingest

logic_logger = logging.getLogger("mb_logic")


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
        self.next_int = start
        if not lookup:
            self.lookup = pa.Table.from_arrays(
                [
                    pa.array([], type=pa.uint64()),
                    pa.array([], type=pa.binary()),
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


def insert_dataset(source: Source, engine: Engine, batch_size: int) -> None:
    """Indexes a dataset from your data warehouse within Matchbox."""

    db_logger = logging.getLogger("sqlalchemy.engine")
    db_logger.setLevel(logging.WARNING)

    ##################
    # Insert dataset #
    ##################

    resolution_hash = source.content_address

    resolution_data = {
        "resolution_hash": resolution_hash,
        "type": ResolutionNodeType.DATASET.value,
    }

    source_data = {
        "alias": source.alias,
        "full_name": source.name_address.full_name,
        "warehouse_hash": source.name_address.warehouse_hash,
        "id": source.db_pk,
        "indices": {
            "literal": [c.literal.base64 for c in source.columns],
            "alias": [c.alias.base64 for c in source.columns],
        },
    }

    clusters = source.to_hashlist()

    with engine.connect() as conn:
        logic_logger.info(f"Adding {source}")

        # Generate existing max primary key values
        next_cluster_id = Clusters.next_id()
        resolution_data["resolution_id"] = Resolutions.next_id()
        source_data["resolution_id"] = resolution_data["resolution_id"]

        # Upsert into Resolutions table
        resolution_stmt = insert(Resolutions).values([resolution_data])
        resolution_stmt = resolution_stmt.on_conflict_do_update(
            index_elements=["resolution_hash"],
            set_={
                "name": resolution_stmt.excluded.name,
                "type": resolution_stmt.excluded.type,
            },
        )
        conn.execute(resolution_stmt)

        logic_logger.info(f"{source} added to Resolutions table")

        # Upsert into Sources table
        sources_stmt = insert(Sources).values([source_data])
        sources_stmt = sources_stmt.on_conflict_do_update(
            index_elements=["resolution_id"],
            set_={
                "schema": sources_stmt.excluded.schema,
                "table": sources_stmt.excluded.table,
                "id": sources_stmt.excluded.id,
            },
        )
        conn.execute(sources_stmt)

        conn.commit()

        logic_logger.info(f"{source} added to Sources table")

        # Upsert into Clusters table
        batch_ingest(
            records=[
                (
                    next_cluster_id + i,
                    clus["hash"],
                    source_data["resolution_id"],
                    clus["source_pk"],
                )
                for i, clus in enumerate(clusters)
            ],
            table=Clusters,
            conn=conn,
            batch_size=batch_size,
        )

        conn.commit()

        logic_logger.info(f"{source} added {len(clusters)} objects to Clusters table")

    logic_logger.info(f"Finished {source}")


def insert_model(
    model: str,
    left: Resolutions,
    right: Resolutions,
    description: str,
    engine: Engine,
) -> None:
    """
    Writes a model to Matchbox with a default truth value of 1.0.

    Args:
        model: Name of the new model
        left: Left parent of the model
        right: Right parent of the model. Same as left in a dedupe job
        description: Model description
        engine: SQLAlchemy engine instance

    Raises:
        ServerResolutionError if the specified parent models don't exist.

    Raises:
        ServerResolutionError if the specified model doesn't exist.
    """
    logic_logger.info(f"[{model}] Registering model")
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
                truth=1.0,
            )
            .on_conflict_do_update(
                index_elements=["resolution_hash"],
                set_={"name": model, "description": description},
            )
        )

        session.execute(stmt)

        if not exists:

            def _create_closure_entries(parent_resolution: Resolutions) -> None:
                """Create closure entries for the new model, i.e. mappings between
                nodes and any of their direct or indirect parents"""
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
    logic_logger.info(f"[{model}] {status} model with ID {resolution_id}")
    logic_logger.info(f"[{model}] Done!")


def _map_ids(
    array: pa.Array,
    lookup: pa.Table,
    source: str,
    target: str,
) -> pa.Array:
    """Maps values in an array via a lookup.

    Args:
        array: Array of values to map
        lookup: Table of values to map to
        source: Name of the column to map from
        target: Name of the column to map to

    Returns:
        Array of mapped values
    """
    indices = pc.index_in(array, lookup[source])
    return pc.take(lookup[target], indices)


def _get_resolution_related_clusters(resolution_id: int) -> Select:
    """
    Get cluster hashes and IDs for a resolution, its parents, and siblings.

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
    source_datasets = select(resolution_set.c.resolution_id).join(
        Sources, Sources.resolution_id == resolution_set.c.resolution_id
    )

    # Subquery for model resolutions
    model_resolutions = select(resolution_set.c.resolution_id).where(
        ~exists()
        .select_from(Sources)
        .where(Sources.resolution_id == resolution_set.c.resolution_id)
    )

    # Combine conditions
    final_query = base_query.where(
        (Clusters.dataset.in_(source_datasets))
        | (Probabilities.resolution.in_(model_resolutions))
    )

    return final_query


def _results_to_insert_tables(
    resolution: Resolutions, results: Results, engine: Engine
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Takes Results and returns three Arrow tables that can be inserted exactly.

    Returns:
        A tuple containing:
            * A Clusters update Arrow table
            * A Contains update Arrow table
            * A Probabilities update Arrow table
    """
    logic_logger.info(f"[{resolution.name}] Wrangling data to insert tables")

    # Create ID-Hash lookup for existing probabilities
    lookup = sql_to_df(
        stmt=_get_resolution_related_clusters(resolution.resolution_id),
        engine=engine,
        return_type="arrow",
    )
    lookup = lookup.cast(pa.schema([("hash", pa.binary()), ("id", pa.uint64())]))

    hm = HashIDMap(start=Clusters.next_id(), lookup=lookup)

    # Join hashes, probabilities and components
    probs_with_ccs = attach_components_to_probabilities(
        pa.table(
            {
                "left": hm.get_hashes(results.probabilities["left_id"]),
                "right": hm.get_hashes(results.probabilities["right_id"]),
                "probability": results.probabilities["probability"],
            }
        )
    )

    # Calculate hierarchies
    hierarchy = to_hierarchical_clusters(
        probabilities=probs_with_ccs,
        hash_func=hash_values,
        dtype=pa.binary,
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
            "dataset": pa.nulls(len(new_hashes), type=pa.uint64()),
            "source_pk": pa.nulls(len(new_hashes), type=pa.list_(pa.string())),
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

    logic_logger.info(f"[{resolution.name}] Wrangling complete!")

    return clusters, contains, probabilities


def insert_results(
    resolution: Resolutions,
    engine: Engine,
    results: Results,
    batch_size: int,
) -> None:
    """
    Writes a Results object to Matchbox.

    The PostgreSQL backend stores clusters in a hierarchical structure, where
    each component references its parent component at a higher threshold.

    This means two-item components are synonymous with their original pairwise
    probabilities.

    This allows easy querying of clusters at any threshold.

    Args:
        resolution: Resolution of type model to associate results with
        engine: SQLAlchemy engine instance
        results: A results object
        batch_size: Number of records to insert in each batch

    Raises:
        ServerResolutionError if the specified model doesn't exist.
    """
    logic_logger.info(
        f"[{resolution.name}] Writing results data with batch size {batch_size:,}"
    )

    clusters, contains, probabilities = _results_to_insert_tables(
        resolution=resolution, results=results, engine=engine
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
            logic_logger.info(f"[{resolution.name}] Removed old probabilities")

        except SQLAlchemyError as e:
            session.rollback()
            logic_logger.error(
                f"[{resolution.name}] Failed to clear old probabilities: {str(e)}"
            )
            raise

    with engine.connect() as conn:
        try:
            logic_logger.info(
                f"[{resolution.name}] Inserting {clusters.shape[0]:,} results objects"
            )

            batch_ingest(
                records=[tuple(c.values()) for c in clusters.to_pylist()],
                table=Clusters,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {clusters.shape[0]} "
                "objects into Clusters table"
            )

            batch_ingest(
                records=[tuple(c.values()) for c in contains.to_pylist()],
                table=Contains,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {contains.shape[0]} "
                "objects into Contains table"
            )

            batch_ingest(
                records=[tuple(c.values()) for c in probabilities.to_pylist()],
                table=Probabilities,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted "
                f"{probabilities.shape[0]} objects into Probabilities table"
            )

        except SQLAlchemyError as e:
            logic_logger.error(f"[{resolution.name}] Failed to insert data: {str(e)}")
            raise

    logic_logger.info(f"[{resolution.name}] Insert operation complete!")
