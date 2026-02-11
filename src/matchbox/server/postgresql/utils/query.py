"""Utilities for querying and matching in the PostgreSQL backend."""

import polars as pl
import pyarrow as pa
from sqlalchemy import and_, func, literal_column, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select, Subquery

from matchbox.common.arrow import SCHEMA_QUERY, SCHEMA_QUERY_WITH_LEAVES
from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    FusionStrategy,
    Match,
    ModelResolutionName,
    ResolutionType,
    ResolverResolutionPath,
    SourceResolutionPath,
    UploadStage,
)
from matchbox.common.exceptions import MatchboxResolutionNotQueriable
from matchbox.common.fusion import fuse_components
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ClusterSourceKey,
    Contains,
    ModelEdges,
    ResolutionClusters,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql


def _empty_edges_df() -> pl.DataFrame:
    return pl.DataFrame(schema={"left_id": pl.UInt64, "right_id": pl.UInt64})


def _empty_assignments_df() -> pl.DataFrame:
    return pl.DataFrame(schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64})


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


def _source_keys_df(
    session: Session,
    source_config_id: int,
    *,
    include_source_config: bool,
) -> pl.DataFrame:
    """Load source keys and source-cluster IDs into a dataframe."""
    columns = [ClusterSourceKey.cluster_id.label("leaf_id"), ClusterSourceKey.key]
    if include_source_config:
        columns.insert(0, ClusterSourceKey.source_config_id)

    rows = session.execute(
        select(*columns).where(ClusterSourceKey.source_config_id == source_config_id)
    ).all()

    if not rows:
        schema: dict[str, pl.DataType] = {
            "leaf_id": pl.Int64,
            "key": pl.String,
        }
        if include_source_config:
            schema = {
                "source_config_id": pl.Int64,
                **schema,
            }
        return pl.DataFrame(schema=schema)

    if include_source_config:
        return pl.DataFrame(
            rows,
            schema={
                "source_config_id": pl.Int64,
                "leaf_id": pl.Int64,
                "key": pl.String,
            },
            orient="row",
        )

    return pl.DataFrame(
        rows,
        schema={
            "leaf_id": pl.Int64,
            "key": pl.String,
        },
        orient="row",
    )


def _source_keys_for_configs_df(
    session: Session,
    source_config_ids: list[int],
) -> pl.DataFrame:
    """Load source keys for multiple source configs."""
    if not source_config_ids:
        return pl.DataFrame(
            schema={
                "source_config_id": pl.Int64,
                "leaf_id": pl.Int64,
                "key": pl.String,
            }
        )

    rows = session.execute(
        select(
            ClusterSourceKey.source_config_id,
            ClusterSourceKey.cluster_id.label("leaf_id"),
            ClusterSourceKey.key,
        ).where(ClusterSourceKey.source_config_id.in_(source_config_ids))
    ).all()

    if not rows:
        return pl.DataFrame(
            schema={
                "source_config_id": pl.Int64,
                "leaf_id": pl.Int64,
                "key": pl.String,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "source_config_id": pl.Int64,
            "leaf_id": pl.Int64,
            "key": pl.String,
        },
        orient="row",
    )


def _assignment_lookup(assignments: pl.DataFrame) -> pl.DataFrame:
    """Build leaf->cluster lookup from fused assignments."""
    if assignments.height == 0:
        return pl.DataFrame(schema={"leaf_id": pl.Int64, "id": pl.Int64})

    return assignments.select(
        pl.col("node_id").cast(pl.Int64).alias("leaf_id"),
        pl.col("cluster_id").cast(pl.Int64).alias("id"),
    ).unique()


def _apply_assignments(
    source_rows: pl.DataFrame,
    assignment_lookup: pl.DataFrame,
) -> pl.DataFrame:
    """Attach root IDs to source rows using override assignments."""
    if source_rows.height == 0:
        return source_rows.with_columns(pl.lit(None, pl.Int64).alias("id"))

    if assignment_lookup.height == 0:
        return source_rows.with_columns(pl.col("leaf_id").cast(pl.Int64).alias("id"))

    return source_rows.join(assignment_lookup, on="leaf_id", how="left").with_columns(
        pl.coalesce([pl.col("id"), pl.col("leaf_id").cast(pl.Int64)]).alias("id")
    )


def _load_direct_inputs(
    session: Session,
    resolver_resolution: Resolutions,
) -> dict[str, Resolutions]:
    """Load direct resolver inputs by name, preserving resolver config order."""
    input_names = list(resolver_resolution.resolver_config.inputs)
    input_resolutions = session.execute(
        select(Resolutions).where(
            and_(
                Resolutions.run_id == resolver_resolution.run_id,
                Resolutions.name.in_(input_names),
            )
        )
    ).scalars()

    by_name = {res.name: res for res in input_resolutions}

    missing = [name for name in input_names if name not in by_name]
    if missing:
        raise MatchboxResolutionNotQueriable(
            f"Resolver inputs were not found for {resolver_resolution.name}: {missing}"
        )

    return by_name


def _load_direct_threshold_cache(
    session: Session,
    resolver_resolution: Resolutions,
) -> dict[str, int | None]:
    """Return cached direct thresholds from the closure table."""
    rows = session.execute(
        select(Resolutions.name, ResolutionFrom.truth_cache)
        .select_from(ResolutionFrom)
        .join(Resolutions, ResolutionFrom.parent == Resolutions.resolution_id)
        .where(
            and_(
                ResolutionFrom.child == resolver_resolution.resolution_id,
                ResolutionFrom.level == 1,
            )
        )
    ).all()

    return {name: truth_cache for name, truth_cache in rows}


def _require_backend_threshold(value: object, *, label: str) -> int:
    """Validate backend threshold payloads are integer percentages in [0, 100]."""
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= 100):
        raise MatchboxResolutionNotQueriable(
            f"Invalid threshold for {label}: {value!r}. "
            "Thresholds must be ints in [0, 100]."
        )
    return value


def _resolve_effective_model_thresholds(
    session: Session,
    resolver_resolution: Resolutions,
    threshold_overrides: dict[ModelResolutionName, int],
    direct_inputs: dict[str, Resolutions],
) -> dict[str, int]:
    """Validate and resolve direct-model thresholds for query-time overrides."""
    input_names = list(resolver_resolution.resolver_config.inputs)

    direct_model_names = {
        name
        for name in input_names
        if direct_inputs[name].type == ResolutionType.MODEL.value
    }
    direct_resolver_names = {
        name
        for name in input_names
        if direct_inputs[name].type == ResolutionType.RESOLVER.value
    }

    invalid_keys = sorted(set(threshold_overrides) - direct_model_names)
    if invalid_keys:
        resolver_keys = [name for name in invalid_keys if name in direct_resolver_names]
        non_direct_keys = [
            name for name in invalid_keys if name not in direct_resolver_names
        ]

        issues: list[str] = []
        if resolver_keys:
            issues.append(
                "resolver inputs cannot be overridden: " + ", ".join(resolver_keys)
            )
        if non_direct_keys:
            issues.append(
                "unknown or non-direct model inputs: " + ", ".join(non_direct_keys)
            )

        raise MatchboxResolutionNotQueriable(
            "threshold_overrides can only target direct model inputs; "
            + "; ".join(issues)
        )

    defaults = resolver_resolution.resolver_config.thresholds
    direct_cache = _load_direct_threshold_cache(session, resolver_resolution)

    effective: dict[str, int] = {}
    for model_name in direct_model_names:
        default = direct_cache.get(model_name)
        if default is None:
            default = defaults.get(model_name, 0)
        effective[model_name] = _require_backend_threshold(
            default,
            label=f"default model input '{model_name}'",
        )

    for model_name, threshold in threshold_overrides.items():
        effective[str(model_name)] = _require_backend_threshold(
            threshold,
            label=f"override model input '{model_name}'",
        )

    return effective


def _load_model_edges_for_threshold(
    session: Session,
    resolution_id: int,
    threshold: int,
) -> pl.DataFrame:
    """Load model edges for a model resolution at a threshold."""
    rows = session.execute(
        select(ModelEdges.left_id, ModelEdges.right_id).where(
            and_(
                ModelEdges.resolution_id == resolution_id,
                ModelEdges.probability >= threshold,
            )
        )
    ).all()

    if not rows:
        return _empty_edges_df()

    return pl.DataFrame(
        rows,
        schema={"left_id": pl.UInt64, "right_id": pl.UInt64},
        orient="row",
    )


def _load_resolver_input_assignments(
    session: Session,
    resolution_id: int,
) -> pl.DataFrame:
    """Load assignments for a direct resolver input (including root membership)."""
    cluster_ids = session.execute(
        select(ResolutionClusters.cluster_id).where(
            ResolutionClusters.resolution_id == resolution_id
        )
    ).scalars()
    cluster_ids_list = [int(cluster_id) for cluster_id in cluster_ids]

    if not cluster_ids_list:
        return _empty_assignments_df()

    rows = session.execute(
        select(Contains.root.label("cluster_id"), Contains.leaf.label("node_id")).where(
            Contains.root.in_(cluster_ids_list)
        )
    ).all()

    assignment_rows = [
        {"cluster_id": int(cluster_id), "node_id": int(node_id)}
        for cluster_id, node_id in rows
    ]
    assignment_rows.extend(
        {
            "cluster_id": cluster_id,
            "node_id": cluster_id,
        }
        for cluster_id in cluster_ids_list
    )

    return (
        pl.DataFrame(assignment_rows)
        .cast({"cluster_id": pl.UInt64, "node_id": pl.UInt64})
        .unique()
    )


def _build_override_assignments(
    session: Session,
    resolver_resolution: Resolutions,
    threshold_overrides: dict[ModelResolutionName, int],
) -> pl.DataFrame:
    """Recompute assignments in memory for query-time threshold overrides."""
    direct_inputs = _load_direct_inputs(session, resolver_resolution)
    effective_thresholds = _resolve_effective_model_thresholds(
        session=session,
        resolver_resolution=resolver_resolution,
        threshold_overrides=threshold_overrides,
        direct_inputs=direct_inputs,
    )

    model_edges: list[pl.DataFrame] = []
    resolver_assignments: list[pl.DataFrame] = []

    for input_name in resolver_resolution.resolver_config.inputs:
        input_resolution = direct_inputs[input_name]

        if input_resolution.type == ResolutionType.MODEL.value:
            model_edges.append(
                _load_model_edges_for_threshold(
                    session=session,
                    resolution_id=input_resolution.resolution_id,
                    threshold=effective_thresholds[input_name],
                )
            )
        elif input_resolution.type == ResolutionType.RESOLVER.value:
            resolver_assignments.append(
                _load_resolver_input_assignments(
                    session=session,
                    resolution_id=input_resolution.resolution_id,
                )
            )

    return fuse_components(
        strategy=FusionStrategy(resolver_resolution.resolver_config.strategy),
        model_edges=model_edges,
        resolver_assignments=resolver_assignments,
    )


def _build_override_query_results(
    session: Session,
    source_config_id: int,
    assignments: pl.DataFrame,
    *,
    return_leaf_id: bool,
    limit: int | None,
) -> pa.Table:
    """Materialise query output using transient override assignments."""
    source_rows = _source_keys_df(
        session=session,
        source_config_id=source_config_id,
        include_source_config=False,
    )
    if source_rows.height == 0:
        schema = SCHEMA_QUERY_WITH_LEAVES if return_leaf_id else SCHEMA_QUERY
        empty = {name: [] for name in schema.names}
        return pa.Table.from_pydict(empty, schema=schema)

    assignment_lookup = _assignment_lookup(assignments)
    resolved_rows = _apply_assignments(
        source_rows=source_rows,
        assignment_lookup=assignment_lookup,
    )

    selection = ["id", "key"]
    if return_leaf_id:
        selection.append("leaf_id")

    resolved_rows = (
        resolved_rows.select(selection)
        .unique()
        .sort(["id", "leaf_id", "key"] if return_leaf_id else ["id", "key"])
    )

    if limit is not None:
        resolved_rows = resolved_rows.head(limit)

    schema = SCHEMA_QUERY_WITH_LEAVES if return_leaf_id else SCHEMA_QUERY
    return resolved_rows.to_arrow().cast(schema)


def _query_without_overrides(
    source_config_id: int,
    point_of_truth_resolution_id: int | None,
    *,
    return_leaf_id: bool,
    limit: int | None,
) -> pa.Table:
    """Execute the cached resolver query path."""
    if point_of_truth_resolution_id is None:
        query_stmt: Select = (
            select(
                ClusterSourceKey.cluster_id.label("root_id"),
                ClusterSourceKey.cluster_id.label("leaf_id"),
                ClusterSourceKey.key,
            )
            .where(ClusterSourceKey.source_config_id == source_config_id)
            .distinct()
        )
    else:
        assignments = _resolver_assignment_subquery(point_of_truth_resolution_id)

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
            .where(ClusterSourceKey.source_config_id == source_config_id)
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


def query(
    source: SourceResolutionPath,
    point_of_truth: ResolverResolutionPath | None = None,
    threshold_overrides: dict[ModelResolutionName, int] | None = None,
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
            if threshold_overrides:
                raise MatchboxResolutionNotQueriable(
                    "threshold_overrides require a resolver point_of_truth."
                )
            return _query_without_overrides(
                source_config_id=source_config.source_config_id,
                point_of_truth_resolution_id=None,
                return_leaf_id=return_leaf_id,
                limit=limit,
            )

        resolver_resolution = Resolutions.from_path(
            path=point_of_truth,
            session=session,
        )
        if resolver_resolution.type != ResolutionType.RESOLVER:
            raise MatchboxResolutionNotQueriable
        if resolver_resolution.upload_stage != UploadStage.COMPLETE:
            raise MatchboxResolutionNotQueriable

        if not threshold_overrides:
            return _query_without_overrides(
                source_config_id=source_config.source_config_id,
                point_of_truth_resolution_id=resolver_resolution.resolution_id,
                return_leaf_id=return_leaf_id,
                limit=limit,
            )

        assignments = _build_override_assignments(
            session=session,
            resolver_resolution=resolver_resolution,
            threshold_overrides=threshold_overrides,
        )

        return _build_override_query_results(
            session=session,
            source_config_id=source_config.source_config_id,
            assignments=assignments,
            return_leaf_id=return_leaf_id,
            limit=limit,
        )


def _match_without_overrides(
    session: Session,
    key: str,
    source_config: SourceConfigs,
    target_configs: list[SourceConfigs],
    resolver_resolution_id: int,
    source_path: SourceResolutionPath,
    targets: list[SourceResolutionPath],
) -> list[Match]:
    """Execute cached resolver matching path."""
    assignments = _resolver_assignment_subquery(resolver_resolution_id)
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
                source=source_path,
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
                source=source_path,
                source_id=source_ids,
                target=target,
                target_id=matches_by_source_id.get(
                    target_config.source_config_id,
                    set(),
                ),
            )
        )

    return result


def _match_with_overrides(
    session: Session,
    key: str,
    source_config: SourceConfigs,
    target_configs: list[SourceConfigs],
    resolver_resolution: Resolutions,
    source_path: SourceResolutionPath,
    targets: list[SourceResolutionPath],
    threshold_overrides: dict[ModelResolutionName, int],
) -> list[Match]:
    """Execute matching using transient query-time override assignments."""
    assignments = _build_override_assignments(
        session=session,
        resolver_resolution=resolver_resolution,
        threshold_overrides=threshold_overrides,
    )
    assignment_lookup = _assignment_lookup(assignments)

    source_rows = _source_keys_df(
        session=session,
        source_config_id=source_config.source_config_id,
        include_source_config=False,
    ).filter(pl.col("key") == key)

    if source_rows.height == 0:
        return [
            Match(
                cluster=None,
                source=source_path,
                source_id=set(),
                target=target,
                target_id=set(),
            )
            for target in targets
        ]

    source_resolved = _apply_assignments(source_rows, assignment_lookup)
    cluster_ids = source_resolved.select("id").unique().to_series().to_list()

    if not cluster_ids:
        return [
            Match(
                cluster=None,
                source=source_path,
                source_id=set(),
                target=target,
                target_id=set(),
            )
            for target in targets
        ]

    # Mirror the historical SQL path with deterministic tie-breaking.
    target_cluster = int(sorted(int(cluster_id) for cluster_id in cluster_ids)[0])

    source_and_target_ids = [
        source_config.source_config_id,
        *(tc.source_config_id for tc in target_configs),
    ]

    source_and_target_rows = _source_keys_for_configs_df(
        session=session,
        source_config_ids=source_and_target_ids,
    )
    resolved_rows = _apply_assignments(
        source_and_target_rows,
        assignment_lookup,
    ).filter(pl.col("id") == target_cluster)

    matches_by_source_id: dict[int, set[str]] = {}
    if resolved_rows.height > 0:
        grouped = resolved_rows.group_by("source_config_id").agg(pl.col("key").unique())
        for row in grouped.iter_rows(named=True):
            matches_by_source_id[int(row["source_config_id"])] = {
                str(value) for value in row["key"]
            }

    source_ids = matches_by_source_id.get(source_config.source_config_id, set())

    result: list[Match] = []
    for target, target_config in zip(targets, target_configs, strict=False):
        result.append(
            Match(
                cluster=target_cluster,
                source=source_path,
                source_id=source_ids,
                target=target,
                target_id=matches_by_source_id.get(
                    target_config.source_config_id,
                    set(),
                ),
            )
        )

    return result


def match(
    key: str,
    source: SourceResolutionPath,
    targets: list[SourceResolutionPath],
    point_of_truth: ResolverResolutionPath,
    threshold_overrides: dict[ModelResolutionName, int] | None = None,
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

        if threshold_overrides:
            return _match_with_overrides(
                session=session,
                key=key,
                source_config=source_config,
                target_configs=target_configs,
                resolver_resolution=resolver_resolution,
                source_path=source,
                targets=targets,
                threshold_overrides=threshold_overrides,
            )

        return _match_without_overrides(
            session=session,
            key=key,
            source_config=source_config,
            target_configs=target_configs,
            resolver_resolution_id=resolver_resolution.resolution_id,
            source_path=source,
            targets=targets,
        )
