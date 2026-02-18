"""Query and match API routes for the Matchbox server."""

from typing import Annotated, Any

import polars as pl
import pyarrow as pa
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import ValidationError

from matchbox.common.arrow import (
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
    table_to_buffer,
)
from matchbox.common.dtos import (
    CollectionName,
    ErrorResponse,
    Match,
    ModelResolutionPath,
    ResolutionType,
    ResolverResolutionName,
    ResolverResolutionPath,
    RunID,
    SourceResolutionName,
    SourceResolutionPath,
)
from matchbox.common.exceptions import MatchboxResolutionNotQueriable
from matchbox.common.resolvers import (
    ResolverMethod,
    build_override_lookup,
    collect_used_ids,
    compute_override_assignments,
    get_resolver_class,
    project_baseline_rows,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    RequireCollectionRead,
)

router = APIRouter(tags=["retrieval"])

_BASELINE_SCHEMA = {"id": pl.UInt64, "leaf_id": pl.UInt64, "key": pl.String}


def _load_override_inputs(
    backend: BackendDependency,
    point_of_truth: ResolverResolutionPath,
) -> tuple[
    type[ResolverMethod],
    dict[str, pl.DataFrame],
    dict[str, pl.DataFrame],
]:
    run = backend.get_run(
        collection=point_of_truth.collection,
        run_id=point_of_truth.run,
    )
    resolution = run.resolutions.get(point_of_truth.name)
    if resolution is None:
        raise MatchboxResolutionNotQueriable(
            f"Resolver '{point_of_truth.name}' does not exist in run."
        )
    if resolution.resolution_type != ResolutionType.RESOLVER:
        raise MatchboxResolutionNotQueriable(
            "resolver_overrides require a resolver point_of_truth."
        )

    resolver_class = get_resolver_class(resolution.config.resolver_class)
    model_edges: dict[str, pl.DataFrame] = {}
    resolver_assignments: dict[str, pl.DataFrame] = {}
    for input_name in resolution.config.inputs:
        input_resolution = run.resolutions.get(input_name)
        if input_resolution is None:
            raise MatchboxResolutionNotQueriable(
                f"Resolver input '{input_name}' does not exist in run."
            )

        if input_resolution.resolution_type == ResolutionType.MODEL:
            model_edges[input_name] = pl.from_arrow(
                backend.get_model_data(
                    ModelResolutionPath(
                        collection=point_of_truth.collection,
                        run=point_of_truth.run,
                        name=input_name,
                    )
                )
            )
            continue

        if input_resolution.resolution_type == ResolutionType.RESOLVER:
            resolver_assignments[input_name] = pl.from_arrow(
                backend.get_resolver_data(
                    ResolverResolutionPath(
                        collection=point_of_truth.collection,
                        run=point_of_truth.run,
                        name=input_name,
                    )
                )
            )
            continue

        raise MatchboxResolutionNotQueriable(
            f"Resolver input '{input_name}' has unsupported type "
            f"'{input_resolution.resolution_type}'."
        )

    return resolver_class, model_edges, resolver_assignments


def _baseline_rows(
    backend: BackendDependency,
    source: SourceResolutionPath,
    point_of_truth: ResolverResolutionPath,
) -> pl.DataFrame:
    rows = pl.from_arrow(
        backend.query(
            source=source,
            point_of_truth=point_of_truth,
            return_leaf_id=True,
        )
    ).select("id", "leaf_id", "key")
    if rows.height == 0:
        return pl.DataFrame(schema=_BASELINE_SCHEMA)
    return rows.cast(_BASELINE_SCHEMA)


def _override_lookup_for_sources(
    backend: BackendDependency,
    point_of_truth: ResolverResolutionPath,
    resolver_overrides: dict[str, Any],
    baseline_rows_by_source: dict[SourceResolutionPath, pl.DataFrame],
) -> pl.DataFrame:
    resolver_class, model_edges, resolver_assignments = _load_override_inputs(
        backend=backend,
        point_of_truth=point_of_truth,
    )
    assignments = compute_override_assignments(
        resolver_class=resolver_class,
        resolver_overrides=resolver_overrides,
        model_edges=model_edges,
        resolver_assignments=resolver_assignments,
    )
    return build_override_lookup(
        assignments=assignments,
        used_ids=collect_used_ids(baseline_rows_by_source.values()),
    )


def _project_query_rows(
    baseline_rows: pl.DataFrame,
    override_lookup: pl.DataFrame,
    *,
    return_leaf_id: bool,
    limit: int | None,
) -> pa.Table:
    projected = project_baseline_rows(
        baseline_rows=baseline_rows,
        override_lookup=override_lookup,
    )

    columns = ["id", "key"]
    sort_columns = ["id", "key"]
    if return_leaf_id:
        columns.append("leaf_id")
        sort_columns = ["id", "leaf_id", "key"]

    result = projected.select(columns).unique().sort(sort_columns)
    if limit is not None:
        result = result.head(limit)

    schema = SCHEMA_QUERY_WITH_LEAVES if return_leaf_id else SCHEMA_QUERY
    return result.to_arrow().cast(schema)


def _project_match_rows(
    source: SourceResolutionPath,
    targets: list[SourceResolutionPath],
    key: str,
    baseline_rows_by_source: dict[SourceResolutionPath, pl.DataFrame],
    override_lookup: pl.DataFrame,
) -> list[Match]:
    source_rows = project_baseline_rows(
        baseline_rows=baseline_rows_by_source[source],
        override_lookup=override_lookup,
    )
    source_cluster = source_rows.filter(pl.col("key") == key).select("id").head(1)
    if source_cluster.height == 0:
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

    cluster = int(source_cluster.item())
    source_ids = set(
        source_rows.filter(pl.col("id") == cluster).get_column("key").to_list()
    )

    results: list[Match] = []
    for target in targets:
        target_rows = project_baseline_rows(
            baseline_rows=baseline_rows_by_source[target],
            override_lookup=override_lookup,
        )
        target_ids = set(
            target_rows.filter(pl.col("id") == cluster).get_column("key").to_list()
        )
        results.append(
            Match(
                cluster=cluster,
                source=source,
                source_id=source_ids,
                target=target,
                target_id=target_ids,
            )
        )
    return results


@router.get(
    "/query",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
)
def query(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    source: SourceResolutionName,
    return_leaf_id: bool,
    resolution: ResolverResolutionName | None = None,
    limit: int | None = None,
) -> ParquetResponse:
    """Query Matchbox for matches based on a source resolution name."""
    res = backend.query(
        source=SourceResolutionPath(collection=collection, run=run_id, name=source),
        point_of_truth=(
            ResolverResolutionPath(collection=collection, run=run_id, name=resolution)
            if resolution
            else None
        ),
        return_leaf_id=return_leaf_id,
        limit=limit,
    )
    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@router.post(
    "/query",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
)
def query_with_overrides(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    source: SourceResolutionName,
    return_leaf_id: bool,
    resolution: ResolverResolutionName,
    resolver_overrides: Annotated[dict[str, Any], Body(embed=True)],
    limit: int | None = None,
) -> ParquetResponse:
    """Query Matchbox by computing resolver overrides in API runtime."""
    source_path = SourceResolutionPath(collection=collection, run=run_id, name=source)
    point_of_truth = ResolverResolutionPath(
        collection=collection,
        run=run_id,
        name=resolution,
    )
    baseline = _baseline_rows(
        backend=backend,
        source=source_path,
        point_of_truth=point_of_truth,
    )
    try:
        override_lookup = _override_lookup_for_sources(
            backend=backend,
            point_of_truth=point_of_truth,
            resolver_overrides=resolver_overrides,
            baseline_rows_by_source={source_path: baseline},
        )
        res = _project_query_rows(
            baseline_rows=baseline,
            override_lookup=override_lookup,
            return_leaf_id=return_leaf_id,
            limit=limit,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@router.get(
    "/match",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
)
def match(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    targets: Annotated[list[SourceResolutionName], Query()],
    source: SourceResolutionName,
    key: str,
    resolution: ResolverResolutionName,
) -> list[Match]:
    """Match a source key against a list of target source resolutions."""
    return backend.match(
        key=key,
        source=SourceResolutionPath(collection=collection, run=run_id, name=source),
        targets=[
            SourceResolutionPath(collection=collection, run=run_id, name=target)
            for target in targets
        ],
        point_of_truth=ResolverResolutionPath(
            collection=collection,
            run=run_id,
            name=resolution,
        ),
    )


@router.post(
    "/match",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    dependencies=[Depends(RequireCollectionRead)],
)
def match_with_overrides(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    targets: Annotated[list[SourceResolutionName], Query()],
    source: SourceResolutionName,
    key: str,
    resolution: ResolverResolutionName,
    resolver_overrides: Annotated[dict[str, Any], Body(embed=True)],
) -> list[Match]:
    """Match source and target keys using resolver override computation."""
    source_path = SourceResolutionPath(collection=collection, run=run_id, name=source)
    target_paths = [
        SourceResolutionPath(collection=collection, run=run_id, name=target)
        for target in targets
    ]
    point_of_truth = ResolverResolutionPath(
        collection=collection,
        run=run_id,
        name=resolution,
    )
    baseline_rows_by_source = {
        source_path: _baseline_rows(
            backend=backend,
            source=source_path,
            point_of_truth=point_of_truth,
        )
    }
    for target in target_paths:
        baseline_rows_by_source[target] = _baseline_rows(
            backend=backend,
            source=target,
            point_of_truth=point_of_truth,
        )

    try:
        override_lookup = _override_lookup_for_sources(
            backend=backend,
            point_of_truth=point_of_truth,
            resolver_overrides=resolver_overrides,
            baseline_rows_by_source=baseline_rows_by_source,
        )
        return _project_match_rows(
            source=source_path,
            targets=target_paths,
            key=key,
            baseline_rows_by_source=baseline_rows_by_source,
            override_lookup=override_lookup,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
