"""Utilities for querying and matching in the PostgreSQL backend."""

from collections import defaultdict

import pyarrow as pa
from sqlalchemy import literal_column, select
from sqlalchemy.orm import Session

from matchbox.common.adapters.sql.query import (
    build_matching_leaves_cte,
    build_target_cluster_cte,
    build_unified_query,
)
from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    Match,
    ResolverStepPath,
    SourceStepPath,
    StepType,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxStepNotQueriable,
    MatchboxStepTypeError,
)
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ClusterSourceKey,
    SourceConfigs,
    Steps,
)
from matchbox.server.postgresql.utils.db import compile_sql


def require_complete_resolver(
    session: Session,
    path: ResolverStepPath,
) -> Steps:
    """Resolve and validate a resolver path for query-time operations."""
    resolver_step = Steps.from_path(path=path, session=session)
    if resolver_step.type != StepType.RESOLVER:
        raise MatchboxStepTypeError(
            step_name=str(path),
            step_type=resolver_step.type,
            expected_step_types=[StepType.RESOLVER],
        )
    if resolver_step.upload_stage != UploadStage.COMPLETE:
        raise MatchboxStepNotQueriable
    return resolver_step


def query(
    source: SourceStepPath,
    resolver: ResolverStepPath | None = None,
    return_leaf_id: bool = False,
    limit: int | None = None,
) -> pa.Table:
    """Query Matchbox to retrieve linked data for a source."""
    with MBDB.get_session() as session:
        source_step: Steps = Steps.from_path(
            path=source,
            session=session,
        )
        source_config: SourceConfigs = source_step.source_config

        # Use the provided resolver to resolve the multi-source query from, or fall
        # back to  the source step for a simple single-source query
        if resolver is None:
            step = source_step
        else:
            step = require_complete_resolver(session, resolver)

        if step.upload_stage != UploadStage.COMPLETE:
            raise MatchboxStepNotQueriable

        lineage = step.get_lineage(sources=[source_config], queryable_only=True)
        query_stmt = build_unified_query(
            lineage=lineage,
            level="key",
            include_source_config_id=False,
        )

    # Order outside the session so the sort is applied to the final statement
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
        # Rename root_id → id to match the public-facing schema
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
    source: SourceStepPath,
    targets: list[SourceStepPath],
    resolver: ResolverStepPath,
) -> list[Match]:
    """Match a source key against targets via a resolver."""
    with MBDB.get_session() as session:
        source_config: SourceConfigs = Steps.from_path(
            path=source,
            session=session,
        ).source_config
        resolver_step = require_complete_resolver(session, resolver)

        # Resolve source configs for all targets to enable ID lookup and result assembly
        target_configs: list[SourceConfigs] = [
            Steps.from_path(path=target, session=session).source_config
            for target in targets
        ]
        source_and_target_ids: list[int] = [
            source_config.source_config_id,
            *(tc.source_config_id for tc in target_configs),
        ]

        # Resolve which cluster this key belongs to according to the resolver.
        # Lineage is unfiltered (no sources) and shared by both CTEs below
        lineage = resolver_step.get_lineage(queryable_only=True)

        target_cluster_cte = build_target_cluster_cte(
            key=key,
            source_config_id=source_config.source_config_id,
            lineage=lineage,
        )

        matching_leaves_cte = build_matching_leaves_cte(
            source_and_target_ids=source_and_target_ids,
            lineage=lineage,
            target_cluster_cte=target_cluster_cte,
        )

        matched_rows = session.execute(
            select(
                matching_leaves_cte.c.cluster_id,
                matching_leaves_cte.c.source_config_id,
                matching_leaves_cte.c.key,
            )
        ).all()

        # Accumulate matching keys by source config ID for fast lookup below
        cluster: int | None = None
        matches_by_source_id: defaultdict[int, set[str]] = defaultdict(set)
        for cluster_id, source_config_id_result, key_in_source in matched_rows:
            if cluster is None:
                cluster = cluster_id
            matches_by_source_id[source_config_id_result].add(key_in_source)

        # Build one Match object per target, defaulting to an empty set when no
        # keys were found for that target config
        return [
            Match(
                cluster=cluster,
                source=source,
                source_id=matches_by_source_id[source_config.source_config_id],
                target=target,
                target_id=matches_by_source_id[target_config.source_config_id],
            )
            for target, target_config in zip(targets, target_configs, strict=False)
        ]
