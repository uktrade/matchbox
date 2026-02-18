"""Shared runtime utilities for executing resolver methods."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl

from matchbox.common.dtos import ResolutionName
from matchbox.common.resolvers.base import (
    ResolverMethod,
    ResolverSettings,
)

_MODEL_EDGE_SCHEMA = {
    "left_id": pl.UInt64,
    "right_id": pl.UInt64,
    "probability": pl.UInt8,
}
_ASSIGNMENT_SCHEMA = {
    "cluster_id": pl.UInt64,
    "node_id": pl.UInt64,
}
_BASELINE_SCHEMA = {
    "id": pl.UInt64,
    "leaf_id": pl.UInt64,
    "key": pl.String,
}
_OVERRIDE_LOOKUP_SCHEMA = {
    "leaf_id": pl.UInt64,
    "override_id": pl.UInt64,
}


def _normalise_model_edges(edges: pl.DataFrame) -> pl.DataFrame:
    """Normalise model-edge inputs to the canonical runtime schema."""
    if edges.height == 0:
        return pl.DataFrame(schema=_MODEL_EDGE_SCHEMA)

    return edges.select("left_id", "right_id", "probability").cast(_MODEL_EDGE_SCHEMA)


def _normalise_assignments(assignments: pl.DataFrame) -> pl.DataFrame:
    """Normalise assignment tables to the canonical runtime schema."""
    if assignments.height == 0:
        return pl.DataFrame(schema=_ASSIGNMENT_SCHEMA)

    return (
        assignments.select("cluster_id", "node_id")
        .cast(_ASSIGNMENT_SCHEMA)
        .unique()
        .sort(["cluster_id", "node_id"])
    )


def run_resolver_method(
    resolver_class: type[ResolverMethod],
    settings_payload: ResolverSettings | Mapping[str, Any],
    model_edges: Mapping[ResolutionName, pl.DataFrame],
    resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
) -> pl.DataFrame:
    """Validate settings, execute resolver method, and normalise output."""
    method = resolver_class(settings=settings_payload)
    normalised_model_edges = {
        resolution_name: _normalise_model_edges(edges)
        for resolution_name, edges in model_edges.items()
    }
    normalised_resolver_assignments = {
        resolution_name: _normalise_assignments(assignments)
        for resolution_name, assignments in resolver_assignments.items()
    }
    computed = method.compute_clusters(
        model_edges=normalised_model_edges,
        resolver_assignments=normalised_resolver_assignments,
    )
    return _normalise_assignments(computed)


def compute_override_assignments(
    resolver_class: type[ResolverMethod],
    resolver_overrides: ResolverSettings | Mapping[str, Any],
    model_edges: Mapping[ResolutionName, pl.DataFrame],
    resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
) -> pl.DataFrame:
    """Compute canonical resolver assignments for a resolver override payload."""
    return run_resolver_method(
        resolver_class=resolver_class,
        settings_payload=resolver_overrides,
        model_edges=model_edges,
        resolver_assignments=resolver_assignments,
    )


def collect_used_ids(rows_list: Iterable[pl.DataFrame]) -> set[int]:
    """Collect used cluster IDs from baseline query row sets."""
    used_ids: set[int] = set()
    for rows in rows_list:
        if rows.height == 0 or "id" not in rows.columns:
            continue
        used_ids.update(
            int(value) for value in rows.get_column("id").unique().to_list()
        )
    return used_ids


def build_override_lookup(
    assignments: pl.DataFrame,
    used_ids: set[int],
) -> pl.DataFrame:
    """Build leaf-to-override lookup using deterministic ID reallocation."""
    normalised = _normalise_assignments(assignments)
    if normalised.height == 0:
        return pl.DataFrame(schema=_OVERRIDE_LOOKUP_SCHEMA)

    cluster_ids = sorted(
        int(cluster_id)
        for cluster_id in normalised.get_column("cluster_id").unique().to_list()
    )
    next_id = max(used_ids, default=0)
    reassigned: dict[int, int] = {}
    for cluster_id in cluster_ids:
        next_id += 1
        reassigned[cluster_id] = next_id

    cluster_map = pl.DataFrame(
        {
            "cluster_id": list(reassigned.keys()),
            "override_id": list(reassigned.values()),
        },
        schema={"cluster_id": pl.UInt64, "override_id": pl.UInt64},
    )

    return (
        normalised.join(cluster_map, on="cluster_id", how="left")
        .select(pl.col("node_id").alias("leaf_id"), "override_id")
        .drop_nulls("override_id")
        .unique()
        .sort(["leaf_id", "override_id"])
        .cast(_OVERRIDE_LOOKUP_SCHEMA)
    )


def project_baseline_rows(
    baseline_rows: pl.DataFrame,
    override_lookup: pl.DataFrame,
) -> pl.DataFrame:
    """Project override IDs over baseline IDs with hierarchy-preserving fallback."""
    if baseline_rows.height == 0:
        return pl.DataFrame(schema=_BASELINE_SCHEMA)

    baseline = baseline_rows.select("id", "leaf_id", "key").cast(_BASELINE_SCHEMA)
    lookup = (
        pl.DataFrame(schema=_OVERRIDE_LOOKUP_SCHEMA)
        if override_lookup.height == 0
        else override_lookup.select("leaf_id", "override_id").cast(
            _OVERRIDE_LOOKUP_SCHEMA
        )
    )
    if lookup.height == 0:
        return baseline

    return (
        baseline.join(lookup, on="leaf_id", how="left")
        .with_columns(pl.coalesce([pl.col("override_id"), pl.col("id")]).alias("id"))
        .drop("override_id")
        .cast(_BASELINE_SCHEMA)
    )
