"""Test deterministic behavior of dedupers."""

from typing import Any, Callable

import pytest

from matchbox import make_model
from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.dedupers.naive import NaiveDeduper, NaiveSettings
from matchbox.client.results import Results
from matchbox.common.factories.entities import FeatureConfig, SuffixRule
from matchbox.common.factories.sources import (
    SourceConfig,
    SourceTestkit,
    linked_sources_factory,
)
from matchbox.common.sources import Source

DeduperConfigurator = Callable[[Source], dict[str, Any]]

# Methodology configuration adapters


def configure_naive_deduper(source: SourceTestkit) -> dict[str, Any]:
    """Configure settings for NaiveDeduper.

    Args:
        source: Source object from linked_sources_factory

    Returns:
        A dictionary with validated settings for NaiveDeduper
    """
    # Extract column names excluding pk and id
    fields = [c.name for c in source.source.columns if c.name not in ("pk", "id")]

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

    source_config = SourceConfig(
        full_name="source_exact",
        features=features,
        n_true_entities=10,
        repetition=0,  # Each entity appears once
    )

    linked = linked_sources_factory(source_configs=(source_config,), seed=42)
    source = linked.sources["source_exact"]

    # Configure and run the deduper
    deduper = make_model(
        model_name="exact_deduper",
        description="Deduplication of exact duplicates",
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left_data=source.query.to_pandas().drop("pk", axis=1),
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
        verbose=True,
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

    source_config = SourceConfig(
        full_name="source_exact",
        features=features,
        n_true_entities=10,
        repetition=2,  # Each entity appears 3 times (base + 2 repetitions)
    )

    linked = linked_sources_factory(source_configs=(source_config,), seed=42)
    source = linked.sources["source_exact"]

    # Configure and run the deduper
    deduper = make_model(
        model_name="exact_deduper",
        description="Deduplication of exact duplicates",
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left_data=source.query.to_pandas().drop("pk", axis=1),
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
        verbose=True,
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
def test_variation_deduplication(
    Deduper: Deduper, configure_deduper: DeduperConfigurator
):
    """Test deduplication with name variations.

    Without cleaning, identifying duplicates with probability 1.0 is impossible.
    """
    # Create a source with name variations
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
            variations=[
                SuffixRule(suffix=" Inc"),
                SuffixRule(suffix=" Ltd"),
            ],
            drop_base=False,
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
        ),
    )

    source_config = SourceConfig(
        full_name="source_var",
        features=features,
        n_true_entities=10,
        repetition=0,  # No repetition needed since we have variations
    )

    linked = linked_sources_factory(source_configs=(source_config,), seed=42)
    source = linked.sources["source_var"]

    # Configure and run the deduper
    deduper = make_model(
        model_name="variation_deduper",
        description="Deduplication with name variations",
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left_data=source.query.to_pandas().drop("pk", axis=1),
        left_resolution="source_var",
    )
    results: Results = deduper.run()

    # Validate results against ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=source.entities,
        right_clusters=None,
        sources=["source_var"],
        threshold=0,
        verbose=True,
    )

    assert not identical
    # Check that there are missing links (partial results)
    assert len(report["partial"]) > 0 or len(report["missing"]) > 0
