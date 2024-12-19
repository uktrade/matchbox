import logging
from collections import defaultdict
from itertools import count

import numpy as np
import pandas as pd
from sqlalchemy import Engine, bindparam, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from matchbox.common.db import Source
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.hash import dataset_to_hashlist, list_to_value_ordered_hash
from matchbox.common.results import ClusterResults, ProbabilityResults, Results
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
        "hash": resolution_hash,
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
            index_elements=["hash"],
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
        resolution_hash = list_to_value_ordered_hash(
            [left.hash, right.hash, bytes(model, encoding="utf-8")]
        )

        # Check if resolution exists
        exists_stmt = select(Resolutions).where(Resolutions.hash == resolution_hash)
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
                hash=resolution_hash,
                type=ResolutionNodeType.MODEL.value,
                name=model,
                description=description,
                truth=1.0,
            )
            .on_conflict_do_update(
                index_elements=["hash"],
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


def _cluster_results_to_hierarchical(
    probabilities: ProbabilityResults,
    clusters: ClusterResults,
) -> pd.DataFrame:
    """
    Converts results to a hierarchical structure by building up from base components.

    Args:
        probabilities: Original pairwise probabilities containing base components
        clusters: Connected components at each threshold

    Returns:
        Pandas DataFrame of (parent, child, threshold) tuples representing the hierarchy
    """
    prob_df = probabilities.dataframe
    cluster_df = clusters.dataframe

    # Sort thresholds in descending order
    thresholds = sorted(cluster_df["threshold"].unique(), reverse=True)

    hierarchy: list[tuple[int, int, float]] = []
    ultimate_parents: dict[int, set[int]] = defaultdict(set)

    # Process each threshold level
    for threshold in thresholds:
        threshold_float = float(threshold)

        # Add new pairwise relationships at this threshold
        current_probs = prob_df[prob_df["probability"] == threshold_float]

        for _, row in current_probs.iterrows():
            parent = row["id"]
            left_id = row["left_id"]
            right_id = row["right_id"]

            hierarchy.extend(
                [
                    (parent, left_id, threshold_float),
                    (parent, right_id, threshold_float),
                ]
            )

            ultimate_parents[left_id].add(parent)
            ultimate_parents[right_id].add(parent)

        # Process clusters at this threshold
        current_clusters = cluster_df[cluster_df["threshold"] == threshold_float]

        # Group by parent to process components together
        for parent, group in current_clusters.groupby("parent"):
            children = set(group["child"])
            if len(children) <= 2:
                continue  # Skip pairs already handled by pairwise probabilities

            current_ultimate_parents: set[int] = set()
            for child in children:
                current_ultimate_parents.update(ultimate_parents[child])

            for up in current_ultimate_parents:
                hierarchy.append((parent, up, threshold_float))

            for child in children:
                ultimate_parents[child] = {parent}

    # Sort hierarchy by threshold (descending), then parent, then child
    return (
        pd.DataFrame(hierarchy, columns=["parent", "child", "threshold"])
        .sort_values(["threshold", "parent", "child"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _update_lookup_with_column(
    df: pd.DataFrame,
    column_name: str,
    lookup: dict[bytes, int | None],
    engine: Engine,
):
    """Updates the hash to int lookup based on the supplied data and column.

    Mimics backend functionality. Would expect this function to disappear when
    feature branch merges -- only here because data moves differently right now.
    """
    hashes = df[column_name].unique().tolist()
    initial_dict = {hash: None for hash in hashes}

    with Session(engine) as session:
        data_inner_join = (
            session.query(Clusters)
            .filter(
                Clusters.hash.in_(
                    bindparam(
                        "ins_ids",
                        hashes,
                        expanding=True,
                    )
                )
            )
            .all()
        )

    return lookup.update(
        initial_dict | {item.hash: item.cluster_id for item in data_inner_join}
    )


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

    # Get the lookup of existing database values and generate new ones
    hierarchy = _cluster_results_to_hierarchical(
        probabilities=results.probabilities, clusters=results.clusters
    )
    hashes = np.unique(
        np.concatenate([hierarchy["parent"].unique(), hierarchy["child"].unique()])
    ).tolist()
    lookup: dict[bytes, int | None] = {hash: None for hash in hashes}

    with Session(engine) as session:
        data_inner_join = (
            session.query(Clusters)
            .filter(
                Clusters.hash.in_(
                    bindparam(
                        "ins_ids",
                        hashes,
                        expanding=True,
                    )
                )
            )
            .all()
        )

        gen_cluster_id = count(Clusters.next_id())

    lookup.update({item.hash: item.cluster_id for item in data_inner_join})
    lookup = {k: next(gen_cluster_id) if v is None else v for k, v in lookup.items()}

    hierarchy["parent_id"] = (
        hierarchy["parent"].apply(lambda i: lookup[i]).astype("int32[pyarrow]")
    )
    hierarchy["child_id"] = (
        hierarchy["child"].apply(lambda i: lookup[i]).astype("int32[pyarrow]")
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
            total_records = results.clusters.dataframe.shape[0]
            logic_logger.info(
                f"[{resolution.name}] Inserting {total_records:,} results objects"
            )

            cluster_records: list[tuple[int, bytes, None, None]] = list(
                zip(
                    hierarchy["parent_id"],
                    hierarchy["parent"],
                    [None] * hierarchy.shape[0],
                    [None] * hierarchy.shape[0],
                    strict=True,
                )
            )
            contains_records: list[tuple[int, int]] = list(
                zip(
                    hierarchy["parent_id"],
                    hierarchy["child_id"],
                    strict=True,
                )
            )
            probability_records: list[tuple[int, int, float]] = list(
                zip(
                    [resolution.resolution_id] * hierarchy.shape[0],
                    hierarchy["parent_id"],
                    hierarchy["threshold"],
                    strict=True,
                )
            )

            batch_ingest(
                records=cluster_records,
                table=Clusters,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {len(cluster_records)} "
                "objects into Clusters table"
            )

            batch_ingest(
                records=contains_records,
                table=Contains,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {len(contains_records)} "
                "objects into Contains table"
            )

            batch_ingest(
                records=probability_records,
                table=Probabilities,
                conn=conn,
                batch_size=batch_size,
            )

            logic_logger.info(
                f"[{resolution.name}] Successfully inserted {len(probability_records)} "
                "objects into Probabilities table"
            )

        except SQLAlchemyError as e:
            logic_logger.error(f"[{resolution.name}] Failed to insert data: {str(e)}")
            raise

    logic_logger.info(f"[{resolution.name}] Insert operation complete!")
