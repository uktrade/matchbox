"""Utilities for inserting data into the PostgreSQL backend."""

import pyarrow as pa
from sqlalchemy import exists, func, join, literal, select
from sqlalchemy.dialects.postgresql import (
    ARRAY,
    BIGINT,
    BYTEA,
    SMALLINT,
    TEXT,
    aggregate_order_by,
    insert,
)

from matchbox.common.arrow import SCHEMA_RESOLVER_MAPPING
from matchbox.common.dtos import (
    ModelResolutionPath,
    ResolutionType,
    ResolverResolutionPath,
    SourceResolutionPath,
)
from matchbox.common.exceptions import (
    MatchboxResolutionExistingData,
    MatchboxResolutionInvalidData,
)
from matchbox.common.hash import hash_arrow_table, hash_model_results
from matchbox.common.logging import logger
from matchbox.common.transform import hash_cluster_leaves
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    ModelEdges,
    ResolutionClusters,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import ingest_to_temporary_table


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
        if resolution.fingerprint != fingerprint:
            raise MatchboxResolutionInvalidData

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
            logger.warning(f"Error, rolling back: {e}", prefix=log_prefix)
            session.rollback()
            raise

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        ClusterSourceKey.__table__.fullname,
    )
    logger.info("Finished", prefix=log_prefix)


def insert_model_edges(
    path: ModelResolutionPath,
    results: pa.Table,
    batch_size: int,
) -> None:
    """Writes model edges to Matchbox."""
    log_prefix = f"Model {path.name}"
    if results.num_rows == 0:
        logger.info("Empty model edges given.", prefix=log_prefix)
        return

    resolution = Resolutions.from_path(path=path, res_type=ResolutionType.MODEL)

    fingerprint = hash_model_results(results)
    if resolution.fingerprint != fingerprint:
        raise MatchboxResolutionInvalidData

    with MBDB.get_session() as session:
        existing_edges = session.execute(
            select(func.count())
            .select_from(ModelEdges)
            .where(ModelEdges.resolution_id == resolution.resolution_id)
        ).scalar_one()

        if existing_edges > 0:
            raise MatchboxResolutionExistingData

    logger.info(
        f"Writing model edges with batch size {batch_size:,}", prefix=log_prefix
    )

    with (
        MBDB.get_session() as session,
        ingest_to_temporary_table(
            table_name="incoming_model_edges",
            schema_name="mb",
            column_types={
                "left_id": BIGINT(),
                "right_id": BIGINT(),
                "probability": SMALLINT(),
            },
            data=results,
            max_chunksize=batch_size,
        ) as incoming_edges,
    ):
        try:
            edges_select = select(
                literal(resolution.resolution_id, BIGINT).label("resolution_id"),
                incoming_edges.c.left_id,
                incoming_edges.c.right_id,
                incoming_edges.c.probability,
            )
            inserted = session.execute(
                insert(ModelEdges).from_select(
                    ["resolution_id", "left_id", "right_id", "probability"],
                    edges_select,
                )
            )
            logger.info(
                f"Will add {inserted.rowcount:,} entries to ModelEdges table",
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


def insert_resolver_clusters(
    path: ResolverResolutionPath,
    assignments: pa.Table,
    batch_size: int,
) -> pa.Table:
    """Write resolver assignments and return client-to-cluster ID mapping.

    Steps:
    1. Validate fingerprint/existing-data constraints and short-circuit empty uploads.
    2. Ingest uploaded assignments into a temporary table.
    3. Expand uploaded nodes to leaves in SQL and validate all leaves exist.
    4. Compute per-client cluster hashes in Python from SQL-aggregated leaf hashes.
    5. Ingest hashes, materialise canonical clusters, and build client->cluster map.
    6. Insert ``contains`` and ``resolution_clusters`` rows from that map.
    7. Return the deterministic mapping table.
    """
    log_prefix = f"Resolver {path.name}"
    fingerprint = hash_arrow_table(assignments)
    assignment_data = assignments.select(["client_cluster_id", "node_id"])

    # 1) Validate resolver upload constraints and short-circuit empty payloads.
    with MBDB.get_session() as session:
        resolution = Resolutions.from_path(
            path=path, res_type=ResolutionType.RESOLVER, session=session
        )

        if resolution.fingerprint != fingerprint:
            raise MatchboxResolutionInvalidData

        existing = session.execute(
            select(func.count())
            .select_from(ResolutionClusters)
            .where(ResolutionClusters.resolution_id == resolution.resolution_id)
        ).scalar_one()

        if existing > 0:
            raise MatchboxResolutionExistingData

        if assignment_data.num_rows == 0:
            return pa.Table.from_pydict(
                {"client_cluster_id": [], "cluster_id": []},
                schema=SCHEMA_RESOLVER_MAPPING,
            )
        resolution_id = resolution.resolution_id

    # 2) Ingest uploaded assignments.
    with ingest_to_temporary_table(
        table_name="resolver_assignments",
        schema_name="mb",
        column_types={"client_cluster_id": BIGINT(), "node_id": BIGINT()},
        data=assignment_data,
        max_chunksize=batch_size,
    ) as incoming_assignments:
        # 3) Expand uploaded nodes to leaves in SQL.
        expanded_leaves = (
            select(
                incoming_assignments.c.client_cluster_id.label("client_cluster_id"),
                func.coalesce(Contains.leaf, incoming_assignments.c.node_id).label(
                    "leaf_id"
                ),
            )
            .distinct()
            .select_from(
                incoming_assignments.outerjoin(
                    Contains,
                    Contains.root == incoming_assignments.c.node_id,
                )
            )
            .subquery("expanded_resolver_leaves")
        )

        with MBDB.get_session() as session:
            # 3a) Resolve leaf hashes and missing leaf IDs in one grouped query.
            grouped_hash_rows = session.execute(
                select(
                    expanded_leaves.c.client_cluster_id,
                    func.array_agg(
                        aggregate_order_by(
                            Clusters.cluster_hash,
                            Clusters.cluster_hash,
                        )
                    )
                    .filter(Clusters.cluster_hash.is_not(None))
                    .label("leaf_hashes"),
                    func.array_agg(
                        aggregate_order_by(
                            expanded_leaves.c.leaf_id,
                            expanded_leaves.c.leaf_id,
                        )
                    )
                    .filter(Clusters.cluster_id.is_(None))
                    .label("missing_leaf_ids"),
                )
                .select_from(
                    expanded_leaves.outerjoin(
                        Clusters,
                        Clusters.cluster_id == expanded_leaves.c.leaf_id,
                    )
                )
                .group_by(expanded_leaves.c.client_cluster_id)
                .order_by(expanded_leaves.c.client_cluster_id)
            ).all()

        # 3b) Validate leaf IDs are all known.
        missing_leaf_ids: set[int] = set()
        for _, _, missing_for_client in grouped_hash_rows:
            if not missing_for_client:
                continue
            if any(leaf_id is None for leaf_id in missing_for_client):
                raise MatchboxResolutionInvalidData(
                    "Resolver upload references null cluster IDs."
                )
            missing_leaf_ids.update(int(leaf_id) for leaf_id in missing_for_client)

        if missing_leaf_ids:
            raise MatchboxResolutionInvalidData(
                "Resolver upload references unknown cluster IDs: "
                f"{sorted(missing_leaf_ids)}"
            )

        # 4) Compute deterministic cluster hashes per client in Python.
        client_cluster_ids: list[int] = []
        cluster_hashes: list[bytes] = []
        for client_cluster_id, leaf_hashes, _ in grouped_hash_rows:
            if not leaf_hashes:
                raise MatchboxResolutionInvalidData(
                    "Resolver upload references no resolvable leaves."
                )
            client_cluster_ids.append(int(client_cluster_id))
            cluster_hashes.append(
                hash_cluster_leaves([bytes(leaf_hash) for leaf_hash in leaf_hashes])
            )
        hash_data = pa.table(
            {
                "client_cluster_id": client_cluster_ids,
                "cluster_hash": cluster_hashes,
            }
        )

        # 5) Ingest hashes, materialise clusters, and build a client->cluster map.
        with (
            ingest_to_temporary_table(
                table_name="resolver_hashes",
                schema_name="mb",
                column_types={"client_cluster_id": BIGINT(), "cluster_hash": BYTEA()},
                data=hash_data,
                max_chunksize=batch_size,
            ) as incoming_hashes,
            MBDB.get_session() as session,
        ):
            try:
                new_hashes = (
                    select(incoming_hashes.c.cluster_hash)
                    .distinct()
                    .where(
                        ~exists(
                            select(1).where(
                                Clusters.cluster_hash == incoming_hashes.c.cluster_hash
                            )
                        )
                    )
                )
                session.execute(
                    insert(Clusters)
                    .from_select(["cluster_hash"], new_hashes)
                    .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
                )
                session.flush()

                cluster_map = (
                    select(
                        incoming_hashes.c.client_cluster_id,
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

                # 6) Persist memberships and resolution-level cluster mapping.
                contains_rows_select = (
                    select(
                        cluster_map.c.cluster_id.label("root"),
                        expanded_leaves.c.leaf_id.label("leaf"),
                    )
                    .select_from(
                        expanded_leaves.join(
                            cluster_map,
                            expanded_leaves.c.client_cluster_id
                            == cluster_map.c.client_cluster_id,
                        )
                    )
                    .distinct()
                )
                session.execute(
                    insert(Contains)
                    .from_select(["root", "leaf"], contains_rows_select)
                    .on_conflict_do_nothing(
                        index_elements=[Contains.root, Contains.leaf]
                    )
                )

                resolution_rows = select(
                    literal(resolution_id, BIGINT).label("resolution_id"),
                    cluster_map.c.cluster_id,
                ).distinct()
                session.execute(
                    insert(ResolutionClusters)
                    .from_select(["resolution_id", "cluster_id"], resolution_rows)
                    .on_conflict_do_nothing(
                        index_elements=[
                            ResolutionClusters.resolution_id,
                            ResolutionClusters.cluster_id,
                        ]
                    )
                )

                # 7) Return mapping rows in deterministic order.
                mapping_rows = session.execute(
                    select(
                        cluster_map.c.client_cluster_id,
                        cluster_map.c.cluster_id,
                    ).order_by(cluster_map.c.client_cluster_id)
                ).all()

                session.commit()
            except Exception:
                session.rollback()
                raise

    logger.info("Resolver cluster insert complete!", prefix=log_prefix)

    MBDB.vacuum_analyze(
        Clusters.__table__.fullname,
        Contains.__table__.fullname,
        ResolutionClusters.__table__.fullname,
    )

    if not mapping_rows:
        return pa.Table.from_pydict(
            {"client_cluster_id": [], "cluster_id": []},
            schema=SCHEMA_RESOLVER_MAPPING,
        )

    return pa.table(
        {
            "client_cluster_id": [
                int(client_cluster_id) for client_cluster_id, _ in mapping_rows
            ],
            "cluster_id": [int(cluster_id) for _, cluster_id in mapping_rows],
        },
        schema=SCHEMA_RESOLVER_MAPPING,
    )
