from collections.abc import Mapping

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from matchbox.common.dtos import ResolutionName
from matchbox.common.resolvers import (
    Components,
    ComponentsSettings,
    ResolverMethod,
    ResolverSettings,
    add_resolver_class,
    build_override_lookup,
    collect_used_ids,
    compute_override_assignments,
    get_resolver_class,
    project_baseline_rows,
    run_resolver_method,
)


def test_components_settings_normalises_float_thresholds() -> None:
    settings = ComponentsSettings(thresholds={"model_a": 0.63})
    assert settings.thresholds == {"model_a": 63}


def test_components_compute_clusters_uses_thresholds() -> None:
    method = Components(
        settings=ComponentsSettings(thresholds={"model_a": 60}),
    )
    model_edges = {
        "model_a": pl.DataFrame(
            {
                "left_id": [1, 2],
                "right_id": [2, 3],
                "probability": [80, 40],
            },
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "probability": pl.UInt8,
            },
        )
    }
    resolver_assignments = {
        "resolver_a": pl.DataFrame(
            {
                "cluster_id": [10, 10],
                "node_id": [3, 4],
            },
            schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64},
        )
    }

    clusters = method.compute_clusters(
        model_edges=model_edges,
        resolver_assignments=resolver_assignments,
    )

    expected = pl.DataFrame(
        {
            "cluster_id": [1, 1, 2, 2],
            "node_id": [1, 2, 3, 4],
        },
        schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64},
    )
    assert_frame_equal(clusters, expected)


class _DummyResolverSettings(ResolverSettings):
    pass


class _DummyResolverMethod(ResolverMethod):
    settings: _DummyResolverSettings

    def compute_clusters(
        self,
        model_edges: Mapping[ResolutionName, pl.DataFrame],
        resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
    ) -> pl.DataFrame:
        return pl.DataFrame(schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64})


def test_resolver_registry_allows_custom_class() -> None:
    add_resolver_class(_DummyResolverMethod)
    assert get_resolver_class("_DummyResolverMethod") is _DummyResolverMethod


def test_run_resolver_method_validates_settings_payload() -> None:
    with pytest.raises(ValueError, match="ComponentsSettings"):
        run_resolver_method(
            resolver_class=Components,
            settings_payload=_DummyResolverSettings(),
            model_edges={},
            resolver_assignments={},
        )


def test_run_resolver_method_normalises_assignment_output() -> None:
    assignments = run_resolver_method(
        resolver_class=Components,
        settings_payload=ComponentsSettings(thresholds={}),
        model_edges={},
        resolver_assignments={
            "resolver_a": pl.DataFrame(
                {
                    "cluster_id": [10, 10, 10],
                    "node_id": [4, 3, 3],
                },
                schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64},
            )
        },
    )

    expected = pl.DataFrame(
        {
            "cluster_id": [1, 1],
            "node_id": [3, 4],
        },
        schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64},
    )
    assert_frame_equal(assignments, expected)


def test_compute_override_assignments_returns_canonical_assignments() -> None:
    assignments = compute_override_assignments(
        resolver_class=Components,
        resolver_overrides=ComponentsSettings(thresholds={"model_a": 80}),
        model_edges={
            "model_a": pl.DataFrame(
                {
                    "left_id": [1],
                    "right_id": [2],
                    "probability": [90],
                },
                schema={
                    "left_id": pl.UInt64,
                    "right_id": pl.UInt64,
                    "probability": pl.UInt8,
                },
            )
        },
        resolver_assignments={},
    )

    expected = pl.DataFrame(
        {"cluster_id": [1, 1], "node_id": [1, 2]},
        schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64},
    )
    assert_frame_equal(assignments, expected)


def test_collect_used_ids_returns_unique_cluster_ids() -> None:
    used_ids = collect_used_ids(
        [
            pl.DataFrame({"id": [100, 100, 200]}, schema={"id": pl.UInt64}),
            pl.DataFrame({"id": [200, 300]}, schema={"id": pl.UInt64}),
        ]
    )
    assert used_ids == {100, 200, 300}


def test_build_override_lookup_reallocates_ids_above_used_ids() -> None:
    lookup = build_override_lookup(
        assignments=pl.DataFrame(
            {
                "cluster_id": [10, 10, 20, 20],
                "node_id": [1, 2, 3, 4],
            },
            schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64},
        ),
        used_ids={100, 200},
    )

    expected = pl.DataFrame(
        {
            "leaf_id": [1, 2, 3, 4],
            "override_id": [201, 201, 202, 202],
        },
        schema={"leaf_id": pl.UInt64, "override_id": pl.UInt64},
    )
    assert_frame_equal(lookup, expected)


def test_project_baseline_rows_coalesces_with_fallback_ids() -> None:
    baseline = pl.DataFrame(
        {
            "id": [100, 100, 200, 200],
            "leaf_id": [1, 2, 3, 4],
            "key": ["A", "B", "C", "D"],
        },
        schema={"id": pl.UInt64, "leaf_id": pl.UInt64, "key": pl.String},
    )
    override_lookup = pl.DataFrame(
        {"leaf_id": [1, 2], "override_id": [201, 201]},
        schema={"leaf_id": pl.UInt64, "override_id": pl.UInt64},
    )

    projected = project_baseline_rows(baseline, override_lookup)
    expected = pl.DataFrame(
        {
            "id": [201, 201, 200, 200],
            "leaf_id": [1, 2, 3, 4],
            "key": ["A", "B", "C", "D"],
        },
        schema={"id": pl.UInt64, "leaf_id": pl.UInt64, "key": pl.String},
    )
    assert_frame_equal(projected, expected)


def test_project_baseline_rows_ab_cd_fallback_regression() -> None:
    baseline = pl.DataFrame(
        {
            "id": [10, 10, 20, 20],
            "leaf_id": [1, 2, 3, 4],
            "key": ["A", "B", "C", "D"],
        },
        schema={"id": pl.UInt64, "leaf_id": pl.UInt64, "key": pl.String},
    )
    override_lookup = pl.DataFrame(
        {"leaf_id": [1, 2], "override_id": [30, 30]},
        schema={"leaf_id": pl.UInt64, "override_id": pl.UInt64},
    )

    projected = project_baseline_rows(baseline, override_lookup)
    assert projected.select("id").to_series().to_list() == [30, 30, 20, 20]


def test_project_baseline_rows_singleton_fallback_regression() -> None:
    baseline = pl.DataFrame(
        {
            "id": [100, 100, 500],
            "leaf_id": [1, 2, 5],
            "key": ["A", "B", "E"],
        },
        schema={"id": pl.UInt64, "leaf_id": pl.UInt64, "key": pl.String},
    )
    override_lookup = pl.DataFrame(
        {"leaf_id": [1, 2], "override_id": [700, 700]},
        schema={"leaf_id": pl.UInt64, "override_id": pl.UInt64},
    )

    projected = project_baseline_rows(baseline, override_lookup)
    assert projected.select("id").to_series().to_list() == [700, 700, 500]
