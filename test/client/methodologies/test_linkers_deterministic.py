"""Test deterministic behavior of linkers."""

from typing import Any, Callable

import pytest
from splink import SettingsCreator
from splink import blocking_rule_library as brl
from splink import comparison_library as cl
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.comparison_creator import ComparisonCreator

from matchbox import make_model
from matchbox.client.models.linkers.base import Linker
from matchbox.client.models.linkers.deterministic import (
    DeterministicLinker,
    DeterministicSettings,
)
from matchbox.client.models.linkers.splinklinker import SplinkLinker, SplinkSettings
from matchbox.client.models.linkers.weighteddeterministic import (
    WeightedDeterministicLinker,
    WeightedDeterministicSettings,
)
from matchbox.client.results import Results
from matchbox.common.factories.entities import FeatureConfig
from matchbox.common.factories.sources import (
    SourceConfig,
    SourceTestkit,
    linked_sources_factory,
    source_factory,
)

LinkerConfigurator = Callable[[SourceTestkit, SourceTestkit], dict[str, Any]]

# Methodology configuration adapters


def configure_deterministic_linker(
    left_source: SourceTestkit, right_source: SourceTestkit
) -> dict[str, Any]:
    """Configure settings for DeterministicLinker.

    Args:
        left_source: Left source object from linked_sources_factory
        right_source: Right source object from linked_sources_factory

    Returns:
        A dictionary with validated settings for DeterministicLinker
    """
    # Extract column names excluding pk and id
    left_fields = [
        c.name for c in left_source.source.columns if c.name not in ("pk", "id")
    ]
    right_fields = [
        c.name for c in right_source.source.columns if c.name not in ("pk", "id")
    ]

    # Build comparison string
    comparisons = []
    for l_field, r_field in zip(left_fields, right_fields, strict=True):
        comparisons.append(f"l.{l_field} = r.{r_field}")

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "comparisons": " and ".join(
            comparisons,
        ),
    }

    # Validate the settings dictionary
    DeterministicSettings.model_validate(settings_dict)

    return settings_dict


def configure_weighted_deterministic_linker(
    left_source: SourceTestkit, right_source: SourceTestkit
) -> dict[str, Any]:
    """Configure settings for WeightedDeterministicLinker.

    Args:
        left_source: Left source object from linked_sources_factory
        right_source: Right source object from linked_sources_factory

    Returns:
        A dictionary with validated settings for WeightedDeterministicLinker
    """
    # Extract column names excluding pk and id
    left_fields = [
        c.name for c in left_source.source.columns if c.name not in ("pk", "id")
    ]
    right_fields = [
        c.name for c in right_source.source.columns if c.name not in ("pk", "id")
    ]

    # Build weighted comparisons with equal weights
    weighted_comparisons = []
    for l_field, r_field in zip(left_fields, right_fields, strict=True):
        weighted_comparisons.append(
            {"comparison": f"l.{l_field} = r.{r_field}", "weight": 1}
        )

    # Create settings dictionary
    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "weighted_comparisons": weighted_comparisons,
        "threshold": 1,  # Require all comparisons to match
    }

    # Validate the settings dictionary
    WeightedDeterministicSettings.model_validate(settings_dict)

    return settings_dict


def configure_splink_linker(
    left_source: SourceTestkit, right_source: SourceTestkit
) -> dict[str, Any]:
    """Configure settings for SplinkLinker.

    Args:
        left_source: Left source object from linked_sources_factory
        right_source: Right source object from linked_sources_factory

    Returns:
        A dictionary with validated settings for SplinkLinker
    """
    # Extract column names excluding pk and id
    left_fields = [
        c.name for c in left_source.source.columns if c.name not in ("pk", "id")
    ]
    right_fields = [
        c.name for c in right_source.source.columns if c.name not in ("pk", "id")
    ]

    deterministic_matching_rules: list[str] = []
    blocking_rules_to_generate_predictions: list[BlockingRuleCreator] = []
    comparisons: list[ComparisonCreator] = []

    for l_field, r_field in zip(left_fields, right_fields, strict=True):
        # Splink requires exact name matches
        assert l_field == r_field

        deterministic_matching_rules.append(f"l.{l_field} = r.{r_field}")
        blocking_rules_to_generate_predictions.append(brl.block_on(l_field))
        comparisons.append(cl.ExactMatch(l_field).configure(m_probabilities=[1, 0]))

    linker_training_functions = [
        {
            "function": "estimate_probability_two_random_records_match",
            "arguments": {
                "deterministic_matching_rules": deterministic_matching_rules,
                "recall": 1,
            },
        },
        {
            "function": "estimate_u_using_random_sampling",
            "arguments": {"max_pairs": 1e4},
        },
    ]

    # The m parameter is 1 because we're testing in a deterministic system, and
    # many of these tests only have one field, so we can't use expectation
    # maximisation to estimate. For testing raw functionality, fine to use 1
    linker_settings = SettingsCreator(
        link_type="link_only",
        retain_matching_columns=False,
        retain_intermediate_calculation_columns=False,
        blocking_rules_to_generate_predictions=blocking_rules_to_generate_predictions,
        comparisons=comparisons,
    )

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "linker_training_functions": linker_training_functions,
        "linker_settings": linker_settings,
        "threshold": None,
    }

    # Validate the settings dictionary
    SplinkSettings.model_validate(settings_dict)

    return settings_dict


LINKERS = [
    pytest.param(
        DeterministicLinker, configure_deterministic_linker, id="Deterministic"
    ),
    pytest.param(
        WeightedDeterministicLinker,
        configure_weighted_deterministic_linker,
        id="WeightedDeterministic",
    ),
    pytest.param(SplinkLinker, configure_splink_linker, id="Splink"),
    # Add more linker classes and configuration functions here
]

# Test cases


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
def test_exact_match_linking(Linker: Linker, configure_linker: LinkerConfigurator):
    """Test linking with exact matches between sources."""
    # Create sources with the same entities
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

    configs = (
        SourceConfig(
            full_name="source_left",
            features=features,
            n_true_entities=10,
        ),
        SourceConfig(
            full_name="source_right",
            features=features,
            n_true_entities=10,  # Same number of entities
        ),
    )

    linked = linked_sources_factory(source_configs=configs, seed=42)
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    # Configure and run the linker
    linker = make_model(
        model_name="exact_match_linker",
        description="Linking with exact matches",
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left_data=left_source.query.to_pandas().drop("pk", axis=1),
        left_resolution="source_left",
        right_data=right_source.query.to_pandas().drop("pk", axis=1),
        right_resolution="source_right",
    )
    results: Results = linker.run()

    # Validate results against ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=left_source.entities,
        right_clusters=right_source.entities,
        sources=["source_left", "source_right"],
        threshold=0,
        verbose=True,
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
def test_exact_match_with_duplicates_linking(
    Linker: Linker, configure_linker: LinkerConfigurator
):
    """Test linking with exact matches between sources, where data is duplicated."""
    # Create sources with the same entities
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

    configs = (
        SourceConfig(
            full_name="source_left",
            features=features,
            n_true_entities=10,
            repetition=1,  # Each entity appears once
        ),
        SourceConfig(
            full_name="source_right",
            features=features,
            n_true_entities=10,  # Same number of entities
            repetition=3,  # Each entity appears 3 times
        ),
    )

    linked = linked_sources_factory(source_configs=configs, seed=42)
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    # Configure and run the linker
    linker = make_model(
        model_name="exact_match_linker",
        description="Linking with exact matches",
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left_data=left_source.query.to_pandas().drop("pk", axis=1),
        left_resolution="source_left",
        right_data=right_source.query.to_pandas().drop("pk", axis=1),
        right_resolution="source_right",
    )
    results: Results = linker.run()

    # Validate results against ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=left_source.entities,
        right_clusters=right_source.entities,
        sources=["source_left", "source_right"],
        threshold=0,
        verbose=True,
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
def test_partial_entity_linking(Linker: Linker, configure_linker: LinkerConfigurator):
    """Test linking when one source contains only a subset of entities.

    This tests that the linker correctly handles when the right source
    only contains a subset of the entities in the left source.
    """
    # Create features for our test sources
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="registration_id",
            base_generator="numerify",
            parameters=(("text", "######"),),
        ),
    )

    # Configure sources - full set on left, half on right
    configs = (
        SourceConfig(
            full_name="source_left",
            features=features,
            n_true_entities=10,  # Full set
        ),
        SourceConfig(
            full_name="source_right",
            features=features,
            n_true_entities=5,  # Half the entities
        ),
    )

    # Create the linked sources
    linked = linked_sources_factory(source_configs=configs, seed=42)
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    # Configure and run the linker
    linker = make_model(
        model_name="partial_match_linker",
        description="Linking with partial entity coverage",
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left_data=left_source.query.to_pandas().drop("pk", axis=1),
        left_resolution="source_left",
        right_data=right_source.query.to_pandas().drop("pk", axis=1),
        right_resolution="source_right",
    )
    results = linker.run()

    # Validate results against ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=left_source.entities,
        right_clusters=right_source.entities,
        sources=["source_left", "source_right"],
        threshold=0,
        verbose=True,
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Linker", "configure_linker"), LINKERS)
def test_no_matching_entities_linking(
    Linker: Linker, configure_linker: LinkerConfigurator
):
    """Test linking when there are no matching entities between sources.

    Verifies linkers behave correctly when there should be no matches.
    """
    # Create two completely separate sets of entities
    features = (
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="identifier", base_generator="uuid4"),
    )

    # Create the sources with disjoint entity sets
    left_source = source_factory(features=features, n_true_entities=10, seed=314)
    right_source = source_factory(features=features, n_true_entities=10, seed=159)

    # Configure and run the linker
    linker = make_model(
        model_name="no_match_linker",
        description="Linking with no matching entities",
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left_data=left_source.query.to_pandas().drop("pk", axis=1),
        left_resolution="source_left",
        right_data=right_source.query.to_pandas().drop("pk", axis=1),
        right_resolution="source_right",
    )
    results = linker.run()

    # For this case, we expect no probabilities or empty results
    assert results.probabilities.shape[0] == 0, "Expected no matching probabilities"
