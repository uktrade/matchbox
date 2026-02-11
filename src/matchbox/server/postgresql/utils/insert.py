"""Utilities for inserting data into the PostgreSQL backend."""

import polars as pl
import pyarrow as pa
from sqlalchemy import exists, func, join, literal, select
from sqlalchemy.dialects.postgresql import (
    ARRAY,
    BIGINT,
    BYTEA,
    SMALLINT,
    TEXT,
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
) -> pa.Table:
    """Writes resolver assignments and returns client-to-cluster ID mapping."""
    log_prefix = f"Resolver {path.name}"
    fingerprint = hash_arrow_table(assignments)
    assignment_df = pl.from_arrow(assignments)

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

        if assignment_df.height == 0:
            return pa.Table.from_pydict(
                {"client_cluster_id": [], "cluster_id": []},
                schema=SCHEMA_RESOLVER_MAPPING,
            )

        node_ids: list[int] = (
            assignment_df.select("node_id")
            .unique()
            .to_series()
            .cast(pl.Int64)
            .to_list()
        )
        contains_rows = session.execute(
            select(Contains.root, Contains.leaf).where(Contains.root.in_(node_ids))
        ).all()

        root_to_leaves: dict[int, set[int]] = {}
        for root_id, leaf_id in contains_rows:
            root_to_leaves.setdefault(root_id, set()).add(leaf_id)

        expanded_rows: list[tuple[int, int]] = []
        for client_cluster_id, node_id in assignment_df.iter_rows():
            node = int(node_id)
            leaves = root_to_leaves.get(node, {node})
            for leaf_id in leaves:
                expanded_rows.append((int(client_cluster_id), int(leaf_id)))

        expanded_df = (
            pl.DataFrame(
                expanded_rows,
                schema={"client_cluster_id": pl.Int64, "leaf_id": pl.Int64},
                orient="row",
            )
            .group_by("client_cluster_id", maintain_order=True)
            .agg(pl.col("leaf_id").unique().sort())
        )

        all_leaves: list[int] = (
            expanded_df.select(pl.col("leaf_id").explode())
            .unique()
            .to_series()
            .to_list()
        )
        leaf_hash_rows = session.execute(
            select(Clusters.cluster_id, Clusters.cluster_hash).where(
                Clusters.cluster_id.in_(all_leaves)
            )
        ).all()
        leaf_hashes: dict[int, bytes] = {
            int(cluster_id): cluster_hash for cluster_id, cluster_hash in leaf_hash_rows
        }

        missing_leaves = set(all_leaves) - set(leaf_hashes)
        if missing_leaves:
            raise MatchboxResolutionInvalidData(
                "Resolver upload references unknown cluster IDs: "
                f"{sorted(missing_leaves)}"
            )

        mapping_rows: list[dict[str, int]] = []

        for row in expanded_df.iter_rows(named=True):
            client_cluster_id = int(row["client_cluster_id"])
            leaves = [int(leaf_id) for leaf_id in row["leaf_id"]]
            cluster_hash = hash_cluster_leaves(
                [leaf_hashes[leaf_id] for leaf_id in leaves]
            )

            inserted_root = session.execute(
                insert(Clusters)
                .values(cluster_hash=cluster_hash)
                .on_conflict_do_nothing(index_elements=[Clusters.cluster_hash])
                .returning(Clusters.cluster_id)
            ).scalar_one_or_none()

            if inserted_root is None:
                root_id = int(
                    session.execute(
                        select(Clusters.cluster_id).where(
                            Clusters.cluster_hash == cluster_hash
                        )
                    ).scalar_one()
                )
            else:
                root_id = int(inserted_root)

            session.execute(
                insert(Contains)
                .values([{"root": root_id, "leaf": leaf_id} for leaf_id in leaves])
                .on_conflict_do_nothing(index_elements=[Contains.root, Contains.leaf])
            )

            session.execute(
                insert(ResolutionClusters)
                .values(resolution_id=resolution.resolution_id, cluster_id=root_id)
                .on_conflict_do_nothing(
                    index_elements=[
                        ResolutionClusters.resolution_id,
                        ResolutionClusters.cluster_id,
                    ]
                )
            )

            mapping_rows.append(
                {"client_cluster_id": client_cluster_id, "cluster_id": root_id}
            )

        session.commit()

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

    return (
        pl.DataFrame(mapping_rows)
        .sort("client_cluster_id")
        .to_arrow()
        .cast(SCHEMA_RESOLVER_MAPPING)
    )
