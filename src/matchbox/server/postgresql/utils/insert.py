import logging
from itertools import count
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
from sqlalchemy import Engine, bindparam, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from matchbox.client.results import Results
from matchbox.common.db import Source
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.hash import dataset_to_hashlist, hash_values
from matchbox.common.transform import (
    attach_components_to_probabilities,
    drop_duplicates,
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


def insert_dataset(dataset: Source, engine: Engine, batch_size: int) -> None:
    """Indexes a dataset from your data warehouse within Matchbox."""

    db_logger = logging.getLogger("sqlalchemy.engine")
    db_logger.setLevel(logging.WARNING)

    ##################
    # Insert dataset #
    ##################

    resolution_hash = dataset.to_hash()

    resolution_data = {
        "resolution_hash": resolution_hash,
        "type": ResolutionNodeType.DATASET.value,
        "name": f"{dataset.db_schema}.{dataset.db_table}",
    }

    source_data = {
        "alias": dataset.alias,
        "schema": dataset.db_schema,
        "table": dataset.db_table,
        "id": dataset.db_pk,
        "indices": {
            "literal": [c.literal.base64 for c in dataset.db_columns if c.indexed],
            "alias": [c.alias.base64 for c in dataset.db_columns if c.indexed],
        },
    }

    clusters = dataset_to_hashlist(dataset=dataset)

    with engine.connect() as conn:
        logic_logger.info(f"Adding {dataset}")

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

        logic_logger.info(f"{dataset} added to Resolutions table")

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

        logic_logger.info(f"{dataset} added to Sources table")

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

        logic_logger.info(f"{dataset} added {len(clusters)} objects to Clusters table")

    logic_logger.info(f"Finished {dataset}")


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
        MatchboxResolutionError if the specified parent models don't exist.

    Raises:
        MatchboxResolutionError if the specified model doesn't exist.
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
    replace: Iterator | None = None,
) -> pa.Array:
    """Maps values in an array to a lookup, replacing nulls with an iterator.

    Args:
        array: Array of values to map
        lookup: Table of values to map to
        source: Name of the column to map from
        target: Name of the column to map to
        replace (optional): Iterator that generates values to replace nulls

    Raises:
        ValueError if nulls are found with no replacement iterator

    Returns:
        Array of mapped values
    """
    indices = pc.index_in(array, lookup[source])
    ids = pc.take(lookup[target], indices)
    nulls = pc.is_null(ids)

    if replace is None and pc.any(nulls).as_py():
        raise ValueError(
            "Null values found in mapping but no replacement iterator provided"
        )

    if replace:
        replacements = pa.array(
            [next(replace) if is_null else None for is_null in nulls.to_pylist()],
            type=ids.type,
        )
        return pc.coalesce(ids, replacements)

    return ids


def _create_id_lookup(*id_arrs: *tuple[pa.Array, ...], engine: Engine):
    """Creates an ID-Hash lookup for existing Clusters.

    This shares logic with MatchboxDBAdapter.cluster_id_to_hash, and should use
    it when this is moved to an API layer.
    """
    unique_ids = pc.unique(pa.concat_arrays([pc.unique(a) for a in id_arrs]))

    with Session(engine) as session:
        matched = (
            session.query(Clusters)
            .filter(
                Clusters.cluster_id.in_(
                    bindparam(
                        "ins_ids",
                        unique_ids.to_pylist(),
                        expanding=True,
                    )
                )
            )
            .all()
        )

    return pa.table(
        {
            "id": pa.array([c.cluster_id for c in matched], type=pa.uint64()),
            "hash": pa.array([c.cluster_hash for c in matched], type=pa.large_binary()),
        }
    )


def _create_hash_lookup(*hash_arrs: *tuple[pa.Array, ...], engine: Engine):
    """Creates a Hash-ID lookup for existing Clusters.

    This shares logic with MatchboxDBAdapter.cluster_hash_to_id, and should use
    it when this is moved to an API layer.
    """
    unique_hashes = pc.unique(pa.concat_arrays([pc.unique(a) for a in hash_arrs]))

    with Session(engine) as session:
        matched = (
            session.query(Clusters)
            .filter(
                Clusters.cluster_hash.in_(
                    bindparam(
                        "ins_hashes",
                        unique_hashes.to_pylist(),
                        expanding=True,
                    )
                )
            )
            .all()
        )

    return pa.table(
        {
            "id": pa.array([c.cluster_id for c in matched], type=pa.uint64()),
            "hash": pa.array([c.cluster_hash for c in matched], type=pa.large_binary()),
        }
    )


def _get_existing_hash_mask(hashes: pa.Array, engine: Engine) -> pa.Array:
    """Given a an array of hashes, return a mask array of the ones in Clusters.

    Array contains true if the value exists.
    """
    with Session(engine) as session:
        existing = (
            session.query(Clusters.cluster_hash)
            .filter(
                Clusters.cluster_hash.in_(
                    bindparam(
                        "ins_hashes",
                        hashes.to_pylist(),
                        expanding=True,
                    )
                )
            )
            .all()
        )

    existing_hashes = {row[0] for row in existing}

    mask = pa.array(
        [hash_val in existing_hashes for hash_val in hashes], type=pa.bool_()
    )

    return mask


def _results_to_insert_tables(
    resolution: Resolutions, results: Results, engine: Engine
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Takes Results and returns three Arrow tables that can be inserted exactly.

    Returns:
        A tuple containing:
            * The Clusters table
                * cluster_id, pa.uint64()
                * cluster_hash, pa.large_binary()
                * dataset, pa.uint64() (all Null)
                * source_pk, pa.list_(type=pa.string())) (all Null)
            * The Contains table
                * parent, pa.uint64()
                * child, pa.uint64()
            * The Probabilities table
                * resolution, pa.uint64()
                * cluster, pa.uint64()
                * probabilities, pa.uint8()
    """
    logic_logger.info(f"[{resolution.name}] Wrangling data to insert tables")

    # Create initial ID-Hash lookup for probabilities
    lookup = _create_id_lookup(
        results.probabilities["left_id"],
        results.probabilities["right_id"],
        engine=engine,
    )

    # Look up hashes
    left = _map_ids(
        array=results.probabilities["left_id"],
        lookup=lookup,
        source="id",
        target="hash",
    )
    right = _map_ids(
        array=results.probabilities["right_id"],
        lookup=lookup,
        source="id",
        target="hash",
    )

    # Join hashes, probabilities and components
    probs_with_ccs = attach_components_to_probabilities(
        pa.table(
            {
                "left": left,
                "right": right,
                "probability": results.probabilities["probability"],
            }
        )
    )

    # Calculate hierarchies
    hierarchy = to_hierarchical_clusters(
        probabilities=probs_with_ccs,
        hash_func=hash_values,
        dtype=pa.large_binary,
    )

    # New hashes may already exist. Augment the lookup
    new_lookup = _create_hash_lookup(
        hierarchy["parent"], hierarchy["child"], engine=engine
    )
    lookup = drop_duplicates(pa.concat_tables([lookup, new_lookup]))

    cluster_id_generator = count(Clusters.next_id())

    parent_ids = _map_ids(
        array=hierarchy["parent"],
        lookup=lookup,
        source="hash",
        target="id",
        replace=cluster_id_generator,
    )
    child_ids = _map_ids(
        array=hierarchy["child"],
        lookup=lookup,
        source="hash",
        target="id",
        replace=cluster_id_generator,
    )

    hierarchy = pa.table(
        {
            "parent_id": parent_ids,
            "parent_hash": hierarchy["parent"],
            "child_id": child_ids,
            "probability": hierarchy["probability"],
        }
    )
    hierarchy_deduped = drop_duplicates(
        hierarchy.select(["parent_id", "parent_hash", "probability"]),
        on=["parent_id", "parent_hash", "probability"],
        keep="first",
    )

    # Wrangle to output formats

    clusters = pa.table(
        {
            "cluster_id": hierarchy_deduped["parent_id"],
            "cluster_hash": hierarchy_deduped["parent_hash"],
            "dataset": pa.nulls(hierarchy_deduped.shape[0], type=pa.uint64()),
            "source_pk": pa.nulls(
                hierarchy_deduped.shape[0], type=pa.list_(pa.string())
            ),
        }
    )
    contains = hierarchy.select(["parent_id", "child_id"])
    probabilities = pa.table(
        {
            "resolution": pa.array(
                [resolution.resolution_id] * hierarchy_deduped.shape[0],
                type=pa.uint64(),
            ),
            "cluster": hierarchy_deduped["parent_id"],
            "probability": hierarchy_deduped["probability"],
        }
    )

    # Final masking to ensure we only insert new Clusters
    # pg_bulk_ingest can't handle upsert on non-primary key

    mask = pc.invert(pc.is_in(clusters["cluster_hash"], lookup["hash"]))
    clusters = clusters.filter(mask)

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
        MatchboxResolutionError if the specified model doesn't exist.
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
                f"[{resolution.name}] Inserting {clusters.shape[0]:,} results "
                "objects"
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
