"""Test deterministic behavior of dedupers."""

from typing import Any, Callable

import polars as pl
import pytest

from matchbox import make_model
from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.dedupers.naive import NaiveDeduper, NaiveSettings
from matchbox.client.results import Results
from matchbox.common.factories.entities import FeatureConfig
from matchbox.common.factories.sources import (
    SourceTestkit,
    SourceTestkitParameters,
    linked_sources_factory,
)

DeduperConfigurator = Callable[[SourceTestkit], dict[str, Any]]

# Methodology configuration adapters


def configure_naive_deduper(testkit: SourceTestkit) -> dict[str, Any]:
    """Configure settings for NaiveDeduper.

    Args:
        testkit: SourceTestkit object from linked_sources_factory

    Returns:
        A dictionary with validated settings for NaiveDeduper
    """
    # Extract field names excluding key and id
    fields = [
        c.name
        for c in testkit.source_config.index_fields
        if c.name not in ("key", "id")
    ]

    settings_dict = {
        "id": "id",
        "unique_fields": fields,
    }

    NaiveSettings.model_validate(settings_dict)

    return settings_dict


DEDUPERS = [
    pytest.param(NaiveDeduper, configure_naive_deduper, id="Naive"),
    # Add more deduper classes and configuration functions here
]


# Test cases


@pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
def test_no_deduplication(Deduper: Deduper, configure_deduper: DeduperConfigurator):
    """Test deduplication where there aren't actually any duplicates."""
    # Create a source with exact duplicates
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
        ),
    )

    source_parameters = SourceTestkitParameters(
        name="source_exact",
        features=features,
        n_true_entities=10,
        repetition=0,  # Each entity appears once
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    source = linked.sources["source_exact"]

    # Configure and run the deduper
    deduper = make_model(
        name="exact_deduper",
        description="Deduplication of exact duplicates",
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left_data=pl.from_arrow(source.query).drop("key"),
        left_resolution="source_exact",
    )
    results: Results = deduper.run()

    # Validate results against ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=source.entities,
        right_clusters=None,
        sources=["source_exact"],
        threshold=0,
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
def test_exact_duplicate_deduplication(
    Deduper: Deduper, configure_deduper: DeduperConfigurator
):
    """Test deduplication with exact duplicates."""
    # Create a source with exact duplicates
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
        ),
    )

    source_parameters = SourceTestkitParameters(
        name="source_exact",
        features=features,
        n_true_entities=10,
        repetition=2,  # Each entity appears 3 times (base + 2 repetitions)
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    source = linked.sources["source_exact"]

    # Configure and run the deduper
    deduper = make_model(
        name="exact_deduper",
        description="Deduplication of exact duplicates",
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left_data=pl.from_arrow(source.query).drop("key"),
        left_resolution="source_exact",
    )
    results: Results = deduper.run()

    # Validate results against ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=source.entities,
        right_clusters=None,
        sources=["source_exact"],
        threshold=0,
    )

    assert identical, f"Expected perfect results but got: {report}"
