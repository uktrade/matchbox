"""Utilities for inserting data into the PostgreSQL backend."""

import pyarrow as pa
from sqlalchemy import exists, func, join, literal, select
from sqlalchemy.dialects.postgresql import (
    ARRAY,
    BIGINT,
    BYTEA,
    REAL,
    TEXT,
    insert,
)
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import TableClause
from sqlalchemy.sql.selectable import Subquery

from matchbox.common.dtos import (
    ModelStepPath,
    ResolverStepPath,
    SourceStepPath,
    StepType,
)
from matchbox.common.exceptions import (
    MatchboxStepExistingData,
    MatchboxStepInvalidData,
)
from matchbox.common.hash import hash_arrow_table, hash_clusters, hash_model_results
from matchbox.common.logging import logger
from matchbox.common.transform import hash_cluster_leaves
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    ModelEdges,
    ResolverClusters,
    SourceConfigs,
    Steps,
)
from matchbox.server.postgresql.utils.db import ingest_to_temporary_table


def insert_hashes(path: SourceStepPath, data_hashes: pa.Table, batch_size: int) -> None:
    """Indexes hash data for a source.

    Args:
        path: The path of the source step
        data_hashes: Arrow table containing hash data
        batch_size: Batch size for bulk operations

    Raises:
        MatchboxStepNotFoundError: If the specified step doesn't exist.
        MatchboxStepInvalidData: If data fingerprint conflicts with the step.
        MatchboxStepExistingData: If data was already inserted for the step.
    """
    log_prefix = f"Index hashes {path}"
    if data_hashes.num_rows == 0:
        logger.info("No hashes given.", prefix=log_prefix)
        return

    fingerprint = hash_arrow_table(data_hashes)

    with MBDB.get_session() as session:
        step = Steps.from_path(path=path, res_type=StepType.SOURCE, session=session)
        # Check if the content hash is the same
        if step.fingerprint != fingerprint:
            raise MatchboxStepInvalidData

        # Determine if the step already has any keys.
        existing_keys = session.execute(
            select(func.count())
            .select_from(
                join(
                    ClusterSourceKey,
                    SourceConfigs,
                    ClusterSourceKey.source_config_id == SourceConfigs.source_config_id,
                )
            )
            .where(SourceConfigs.step_id == step.step_id)
        ).scalar_one()

        if existing_keys > 0:
            raise MatchboxStepExistingData

        source_config_id = step.source_config.source_config_id

    with (
        ingest_to_temporary_table(
            table_name="incoming_hashes",
            schema_name=MBDB.settings.postgres.db_schema,
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
                .returning(Clusters.cluster_id)
            )

            result: int = session.execute(
                select(func.count()).select_from(
                    stmt_insert_clusters.cte("inserted_clusters")
                )
            ).scalar_one()

            logger.info(
                f"Will add {result:,} entries to Clusters table",
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

            stmt_insert_keys = (
                insert(ClusterSourceKey)
                .from_select(
                    ["cluster_id", "source_config_id", "key"],
                    exploded,
                )
                .returning(ClusterSourceKey.key_id)
            )

            result: int = session.execute(
                select(func.count()).select_from(stmt_insert_keys.cte("inserted_keys"))
            ).scalar_one()

            logger.info(
                f"Will add {result:,} entries to ClusterSourceKey table",
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


def insert_model_edges(
    path: ModelStepPath,
    results: pa.Table,
    batch_size: int,
) -> None:
    """Writes model edges to Matchbox."""
    log_prefix = f"Model {path.name}"
    if results.num_rows == 0:
        logger.info("Empty model edges given.", prefix=log_prefix)
        return

    step = Steps.from_path(path=path, res_type=StepType.MODEL)

    # Check if the content hash is the same
    fingerprint = hash_model_results(results)
    if step.fingerprint != fingerprint:
        raise MatchboxStepInvalidData

    with MBDB.get_session() as session:
        existing_edges = session.execute(
            select(func.count())
            .select_from(ModelEdges)
            .where(ModelEdges.step_id == step.step_id)
        ).scalar_one()

        if existing_edges > 0:
            raise MatchboxStepExistingData

    logger.info(
        f"Writing model edges with batch size {batch_size:,}", prefix=log_prefix
    )

    with (
        MBDB.get_session() as session,
        ingest_to_temporary_table(
            table_name="incoming_model_edges",
            schema_name=MBDB.settings.postgres.db_schema,
            column_types={
                "left_id": BIGINT(),
                "right_id": BIGINT(),
                "score": REAL(),
            },
            data=results,
            max_chunksize=batch_size,
        ) as incoming_edges,
    ):
        try:
            edges_select = select(
                literal(step.step_id, BIGINT).label("step_id"),
                incoming_edges.c.left_id,
                incoming_edges.c.right_id,
                incoming_edges.c.score,
            )
            stmt_insert_edges = (
                insert(ModelEdges)
                .from_select(
                    ["step_id", "left_id", "right_id", "score"],
                    edges_select,
                )
                .returning(ModelEdges.result_id)
            )
            result: int = session.execute(
                select(func.count()).select_from(
                    stmt_insert_edges.cte("inserted_edges")
                )
            ).scalar_one()

            logger.info(
                f"Will add {result:,} entries to ModelEdges table",
                prefix=log_prefix,
            )
            session.commit()

        except Exception as e:
            logger.error(
                f"Failed to insert model edges, rolling back: {str(e)}",
                prefix=log_prefix,
            )
            session.rollback()
            raise

    MBDB.vacuum_analyze(ModelEdges.__table__.fullname)
    logger.info("Model edge insert complete!", prefix=log_prefix)


def _build_expanded_leaves_subquery(
    incoming_cluster_assignments: TableClause,
) -> Subquery:
    """Expand child assignments to leaf-level cluster IDs per parent cluster."""
    return (
        select(
            incoming_cluster_assignments.c.parent_id,
            func.coalesce(
                Contains.leaf,
                incoming_cluster_assignments.c.child_id,
            ).label("leaf_id"),
        )
        .distinct()
        .select_from(
            # Clusters with no children in Contains resolve to themselves
            incoming_cluster_assignments.outerjoin(
                Contains,
                Contains.root == incoming_cluster_assignments.c.child_id,
            )
        )
        .subquery("expanded_leaves")
    )


def _compute_resolver_hashes(
    incoming_cluster_assignments: TableClause,
    session: Session,
) -> pa.Table:
    """Expand cluster assignments to leaves and compute cluster hashes.

    Each uploaded assignment maps a parent_id to a child_id. A child cluster
    may itself be a parent cluster (with children in Contains), or a leaf cluster
    with no children. This function:

    1. Expands each child cluster to its leaf-level cluster IDs via an outer
        join on Contains. Clusters with no children resolve to themselves.
    2. Groups the leaf hashes per parent cluster.
    3. Computes a single composite hash per parent cluster in Python.

    Uses an **inner join** to Clusters when fetching hashes, so unknown
    leaf IDs are silently dropped here. They will surface later as FK
    violations when inserting into Contains.

    Args:
        incoming_cluster_assignments: Temporary table with
            (parent_id, child_id).
        session: Active SQLAlchemy session (must share the temp table's connection).

    Returns:
        Arrow table with columns (parent_id: int64, cluster_hash: binary).
    """
    # Expand each child_id to its constituent leaves
    expanded_leaves = _build_expanded_leaves_subquery(incoming_cluster_assignments)

    # Group leaf hashes per parent_id
    rows = session.execute(
        select(
            expanded_leaves.c.parent_id,
            func.array_agg(Clusters.cluster_hash).label("leaf_hashes"),
        )
        .select_from(
            expanded_leaves.join(
                Clusters,
                Clusters.cluster_id == expanded_leaves.c.leaf_id,
            )
        )
        .group_by(expanded_leaves.c.parent_id)
    ).all()

    # Compute a single composite hash per parent cluster from leaf hashes
    return pa.table(
        {
            "parent_id": [int(r[0]) for r in rows],
            "cluster_hash": [
                hash_cluster_leaves([bytes(h) for h in r[1]]) for r in rows
            ],
        }
    )


def insert_clusters(
    path: ResolverStepPath,
    cluster_assignments: pa.Table,
    batch_size: int,
) -> None:
    """Write resolver cluster assignments.

    The function proceeds in three phases:

    1. Validate: check fingerprint, ensure no prior data, short-circuit
        if the upload is empty
    2. Compute hashes: ingest assignments to a temp table, expand each
        child cluster to its leaves, and derive a cluster hash per parent
        cluster (Python round-trip via _compute_resolver_hashes)
    3. Insert everything: with both temp tables live in one session,
        materialise new Clusters rows, then insert Contains and
        ResolverClusters membership rows

    Args:
        path: The resolver step path to upload cluster assignments for
        cluster_assignments: Arrow table conforming to SCHEMA_CLUSTERS, having
            (parent_id, child_id) columns
        batch_size: Batch size for temporary table ingestion

    Raises:
        MatchboxStepNotFoundError: If the step doesn't exist
        MatchboxStepInvalidData: If the fingerprint doesn't match
        MatchboxStepExistingData: If clusters already exist for this resolver
    """
    log_prefix = f"Resolver {path.name}"
    fingerprint = hash_clusters(cluster_assignments)
    cluster_assignment_data = cluster_assignments.select(["parent_id", "child_id"])

    # 1) Validate

    with MBDB.get_session() as session:
        step = Steps.from_path(path=path, res_type=StepType.RESOLVER, session=session)

        if step.fingerprint != fingerprint:
            raise MatchboxStepInvalidData

        existing = session.execute(
            select(func.count())
            .select_from(ResolverClusters)
            .where(ResolverClusters.step_id == step.step_id)
        ).scalar_one()

        if existing > 0:
            raise MatchboxStepExistingData

        if cluster_assignment_data.num_rows == 0:
            return

        step_id = step.step_id

    # 2) Compute hashes
    #
    # The assignments temp table stays open for both phases: we query it to
    # compute hashes, then reference it again for the Contains insert. The
    # hashes temp table is opened inside once the Arrow data is ready
    with (
        ingest_to_temporary_table(
            table_name="resolver_assignments",
            schema_name=MBDB.settings.postgres.db_schema,
            column_types={
                "parent_id": BIGINT(),
                "child_id": BIGINT(),
            },
            data=cluster_assignment_data,
            max_chunksize=batch_size,
        ) as incoming_cluster_assignments,
        MBDB.get_session() as session,
    ):
        # Expand child clusters → leaves → hash per parent cluster
        hash_data = _compute_resolver_hashes(incoming_cluster_assignments, session)

        # Leaf expansion subquery
        expanded_leaves = _build_expanded_leaves_subquery(incoming_cluster_assignments)

        # 3) Insert everything
        with ingest_to_temporary_table(
            table_name="resolver_hashes",
            schema_name=MBDB.settings.postgres.db_schema,
            column_types={"parent_id": BIGINT(), "cluster_hash": BYTEA()},
            data=hash_data,
            max_chunksize=batch_size,
        ) as incoming_hashes:
            try:
                # Clusters
                # Insert any hashes we haven't seen before
                session.execute(
                    insert(Clusters)
                    .from_select(
                        ["cluster_hash"],
                        select(incoming_hashes.c.cluster_hash)
                        .distinct()
                        .where(
                            ~exists(
                                select(1).where(
                                    Clusters.cluster_hash
                                    == incoming_hashes.c.cluster_hash
                                )
                            )
                        ),
                    )
                    .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
                )
                session.flush()

                # Map each parent_id to its canonical Clusters.cluster_id
                # by joining hashes back to the now-populated Clusters table
                cluster_map = (
                    select(
                        incoming_hashes.c.parent_id,
                        Clusters.cluster_id,
                    )
                    .select_from(
                        incoming_hashes.join(
                            Clusters,
                            Clusters.cluster_hash == incoming_hashes.c.cluster_hash,
                        )
                    )
                    .subquery("cluster_map")
                )

                # Contains
                # Record which leaves belong to each new resolver cluster
                session.execute(
                    insert(Contains)
                    .from_select(
                        ["root", "leaf"],
                        select(
                            cluster_map.c.cluster_id,
                            expanded_leaves.c.leaf_id,
                        )
                        .select_from(
                            expanded_leaves.join(
                                cluster_map,
                                expanded_leaves.c.parent_id == cluster_map.c.parent_id,
                            )
                        )
                        .distinct(),
                    )
                    .on_conflict_do_nothing(
                        index_elements=[Contains.root, Contains.leaf]
                    )
                )

                # ResolverClusters
                # Associate all proposed clusters with this resolver step.
                session.execute(
                    insert(ResolverClusters)
                    .from_select(
                        ["step_id", "cluster_id"],
                        select(
                            literal(step_id, BIGINT).label("step_id"),
                            cluster_map.c.cluster_id,
                        ).distinct(),
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            ResolverClusters.step_id,
                            ResolverClusters.cluster_id,
                        ]
                    )
                )
                session.commit()

            except Exception:
                session.rollback()
                raise

    logger.info("Resolver cluster insert complete!", prefix=log_prefix)

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        Contains.__table__.fullname,
        ResolverClusters.__table__.fullname,
    )
