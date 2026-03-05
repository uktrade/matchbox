"""Tests for the Components resolver methodology."""

import polars as pl
from polars.testing import assert_frame_equal

from matchbox.client.resolvers import Components, ComponentsSettings
from matchbox.common.arrow import SCHEMA_CLUSTERS


def test_components_compute_clusters_uses_thresholds() -> None:
    """Test thresholds are honoured by the Components.compute_clusters."""
    method = Components(settings=ComponentsSettings(thresholds={"model_a": 0.6}))
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

    clusters = method.compute_clusters(model_edges=model_edges)

    expected = pl.DataFrame(
        {"parent_id": [1, 1], "child_id": [1, 2]},
        schema={"parent_id": pl.UInt64, "child_id": pl.UInt64},
    )
    assert_frame_equal(clusters, expected)


def test_components_compute_clusters_returns_empty_for_no_edges() -> None:
    """Test Components.compute_clusters can work with no data."""
    clusters = Components(settings=ComponentsSettings()).compute_clusters(
        model_edges={}
    )
    assert clusters.height == 0
    assert clusters.schema == pl.Schema(SCHEMA_CLUSTERS)


def test_components_compute_clusters_merges_multiple_models() -> None:
    """Test Components.compute_clusters can work with multiple models."""
    method = Components(
        settings=ComponentsSettings(thresholds={"model_a": 0.0, "model_b": 0.0})
    )
    model_edges = {
        "model_a": pl.DataFrame(
            {"left_id": [1], "right_id": [2], "probability": [90]},
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "probability": pl.UInt8,
            },
        ),
        "model_b": pl.DataFrame(
            {"left_id": [3], "right_id": [4], "probability": [80]},
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "probability": pl.UInt8,
            },
        ),
    }

    clusters = method.compute_clusters(model_edges=model_edges)

    expected = pl.DataFrame(
        {"parent_id": [1, 1, 2, 2], "child_id": [1, 2, 3, 4]},
        schema=pl.Schema(SCHEMA_CLUSTERS),
    )
    assert_frame_equal(clusters, expected)
