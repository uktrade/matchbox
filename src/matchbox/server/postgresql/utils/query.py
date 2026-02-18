"""Utilities for querying and matching in the PostgreSQL backend."""

import pyarrow as pa
from sqlalchemy import and_, func, literal_column, select
from sqlalchemy.sql.selectable import Select, Subquery

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    Match,
    ResolutionType,
    ResolverResolutionPath,
    SourceResolutionPath,
    UploadStage,
)
from matchbox.common.exceptions import MatchboxResolutionNotQueriable
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ClusterSourceKey,
    Contains,
    ResolutionClusters,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql


def _resolver_assignment_subquery(resolution_id: int) -> Subquery:
    """Build leaf->root assignment subquery for a resolver."""
    return (
        select(
            Contains.leaf.label("leaf_id"),
            Contains.root.label("root_id"),
        )
        .select_from(Contains)
        .join(
            ResolutionClusters,
            and_(
                ResolutionClusters.cluster_id == Contains.root,
                ResolutionClusters.resolution_id == resolution_id,
            ),
        )
        .subquery("resolver_assignments")
    )


def query(
    source: SourceResolutionPath,
    point_of_truth: ResolverResolutionPath | None = None,
    return_leaf_id: bool = False,
    limit: int | None = None,
) -> pa.Table:
    """Query Matchbox to retrieve linked data for a source."""
    with MBDB.get_session() as session:
        source_resolution: Resolutions = Resolutions.from_path(
            path=source,
            session=session,
        )
        source_config: SourceConfigs = source_resolution.source_config

        if point_of_truth is None:
            query_stmt: Select = (
                select(
                    ClusterSourceKey.cluster_id.label("root_id"),
                    ClusterSourceKey.cluster_id.label("leaf_id"),
                    ClusterSourceKey.key,
                )
                .where(
                    ClusterSourceKey.source_config_id == source_config.source_config_id
                )
                .distinct()
            )
        else:
            resolver_resolution = Resolutions.from_path(
                path=point_of_truth,
                session=session,
            )
            if resolver_resolution.type != ResolutionType.RESOLVER:
                raise MatchboxResolutionNotQueriable
            if resolver_resolution.upload_stage != UploadStage.COMPLETE:
                raise MatchboxResolutionNotQueriable

            assignments = _resolver_assignment_subquery(
                resolver_resolution.resolution_id
            )
            query_stmt = (
                select(
                    func.coalesce(
                        assignments.c.root_id,
                        ClusterSourceKey.cluster_id,
                    ).label("root_id"),
                    ClusterSourceKey.cluster_id.label("leaf_id"),
                    ClusterSourceKey.key,
                )
                .select_from(ClusterSourceKey)
                .join(
                    assignments,
                    assignments.c.leaf_id == ClusterSourceKey.cluster_id,
                    isouter=True,
                )
                .where(
                    ClusterSourceKey.source_config_id == source_config.source_config_id
                )
                .distinct()
            )

    query_stmt = query_stmt.order_by(
        literal_column("root_id"),
        literal_column("leaf_id"),
        ClusterSourceKey.key,
    )

    if limit is not None:
        query_stmt = query_stmt.limit(limit)

    with MBDB.get_adbc_connection() as conn:
        stmt = compile_sql(query_stmt)
        logger.debug(f"Query SQL: \n {stmt}")
        id_results = sql_to_df(
            stmt=stmt,
            connection=conn,
            return_type="arrow",
        ).rename_columns({"root_id": "id"})

    selection = ["id", "key"]
    if return_leaf_id:
        selection.append("leaf_id")

    return id_results.select(selection)


def match(
    key: str,
    source: SourceResolutionPath,
    targets: list[SourceResolutionPath],
    point_of_truth: ResolverResolutionPath,
) -> list[Match]:
    """Match a source key against targets under a resolver point-of-truth."""
    with MBDB.get_session() as session:
        source_config: SourceConfigs = Resolutions.from_path(
            path=source,
            session=session,
        ).source_config
        resolver_resolution: Resolutions = Resolutions.from_path(
            path=point_of_truth,
            session=session,
        )

        if resolver_resolution.type != ResolutionType.RESOLVER:
            raise MatchboxResolutionNotQueriable
        if resolver_resolution.upload_stage != UploadStage.COMPLETE:
            raise MatchboxResolutionNotQueriable

        target_configs: list[SourceConfigs] = [
            Resolutions.from_path(path=target, session=session).source_config
            for target in targets
        ]

        assignments = _resolver_assignment_subquery(resolver_resolution.resolution_id)
        target_cluster_query = (
            select(func.coalesce(assignments.c.root_id, ClusterSourceKey.cluster_id))
            .select_from(ClusterSourceKey)
            .join(
                assignments,
                assignments.c.leaf_id == ClusterSourceKey.cluster_id,
                isouter=True,
            )
            .where(
                and_(
                    ClusterSourceKey.source_config_id == source_config.source_config_id,
                    ClusterSourceKey.key == key,
                )
            )
            .limit(1)
        )

        cluster = session.execute(target_cluster_query).scalar_one_or_none()
        if cluster is None:
            return [
                Match(
                    cluster=None,
                    source=source,
                    source_id=set(),
                    target=target,
                    target_id=set(),
                )
                for target in targets
            ]

        source_and_target_ids = [
            source_config.source_config_id,
            *(tc.source_config_id for tc in target_configs),
        ]

        assignments_alias = assignments.alias("resolver_assignments_match")
        matched_rows = session.execute(
            select(ClusterSourceKey.source_config_id, ClusterSourceKey.key)
            .select_from(ClusterSourceKey)
            .join(
                assignments_alias,
                assignments_alias.c.leaf_id == ClusterSourceKey.cluster_id,
                isouter=True,
            )
            .where(
                and_(
                    ClusterSourceKey.source_config_id.in_(source_and_target_ids),
                    func.coalesce(
                        assignments_alias.c.root_id,
                        ClusterSourceKey.cluster_id,
                    )
                    == cluster,
                )
            )
            .distinct()
        ).all()

        matches_by_source_id: dict[int, set[str]] = {}
        for source_config_id_result, key_in_source in matched_rows:
            matches_by_source_id.setdefault(source_config_id_result, set()).add(
                key_in_source
            )

        source_ids = matches_by_source_id.get(source_config.source_config_id, set())
        result: list[Match] = []
        for target, target_config in zip(targets, target_configs, strict=False):
            result.append(
                Match(
                    cluster=int(cluster),
                    source=source,
                    source_id=source_ids,
                    target=target,
                    target_id=matches_by_source_id.get(
                        target_config.source_config_id,
                        set(),
                    ),
                )
            )

        return result
