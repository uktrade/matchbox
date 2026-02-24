from collections.abc import Mapping

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from matchbox.client.resolvers import (
    Components,
    ComponentsSettings,
    ResolverMethod,
    ResolverSettings,
    add_resolver_class,
    get_resolver_class,
)
from matchbox.client.resolvers.models import run_resolver_method
from matchbox.common.dtos import ResolutionName


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
