"""Test probabilistic behavior of linkers."""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
from splink import SettingsCreator
from splink import comparison_library as cl

from matchbox.client.models import Model
from matchbox.client.models.linkers.base import Linker
from matchbox.client.models.linkers.splinklinker import SplinkLinker, SplinkSettings
from matchbox.client.models.linkers.weighteddeterministic import (
    WeightedDeterministicLinker,
    WeightedDeterministicSettings,
)
from matchbox.client.queries import Query
from matchbox.client.results import Results
from matchbox.common.factories.entities import (
    FeatureConfig,
    ReplaceRule,
    SuffixRule,
)
from matchbox.common.factories.sources import (
    SourceTestkit,
    SourceTestkitParameters,
    linked_sources_factory,
)

LinkerConfigurator = Callable[[SourceTestkit, SourceTestkit], dict[str, Any]]

# Methodology configuration adapters


def configure_weighted_probabilistic(
    left_testkit: SourceTestkit, right_testkit: SourceTestkit
) -> dict[str, Any]:
    """Configure WeightedDeterministicLinker with probabilistic-like behavior.

    Args:
        left_testkit: Left source object from linked_sources_factory
        right_testkit: Right source object from linked_sources_factory

    Returns:
        A dictionary with validated settings for WeightedDeterministicLinker
    """

    # Extract field names excluding key and id
    left_fields = [
        c.name
        for c in left_testkit.source_config.index_fields
        if c.name not in ("key", "id")
    ]
    right_fields = [
        c.name
        for c in right_testkit.source_config.index_fields
        if c.name not in ("key", "id")
    ]

    # Generate geometric series of weights
    weights = [1 * (0.5**i) for i in range(len(left_fields))]
    total_weight = sum(weights)

    # Normalize weights to sum to 1
    normalized_weights = [w / total_weight for w in weights]

    weighted_comparisons = []
    for l_field, r_field, weight in zip(
        left_fields, right_fields, normalized_weights, strict=False
    ):
        weighted_comparisons.append(
            {"comparison": f"l.{l_field} = r.{r_field}", "weight": weight}
        )

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "weighted_comparisons": weighted_comparisons,
        "threshold": 0.0,
    }

    # Validate the settings dictionary
    WeightedDeterministicSettings.model_validate(settings_dict)

    return settings_dict


def configure_splink_probabilistic(
    left_testkit: SourceTestkit, right_testkit: SourceTestkit
) -> dict[str, Any]:
    """Configure SplinkLinker for probabilistic matching.

    Args:
        left_testkit: Left source object from linked_sources_factory
        right_testkit: Right source object from linked_sources_factory

    Returns:
        A dictionary with validated settings for SplinkLinker
    """

    # Extract field names excluding key and id
    left_fields = [
        c.name
        for c in left_testkit.source_config.index_fields
        if c.name not in ("key", "id")
    ]
    right_fields = [
        c.name
        for c in right_testkit.source_config.index_fields
        if c.name not in ("key", "id")
    ]

    # Create comparison functions based on field type
    comparisons = []
    blocking_rules = []
    deterministic_matching_rules = []

    for l_field, r_field in zip(left_fields, right_fields, strict=True):
        # Splink requires exact name matches
        assert l_field == r_field

        field = l_field  # Use common field name after checking they match
        field_type = next(
            (
                c.type
                for c in left_testkit.source_config.index_fields
                if c.name == field
            ),
            "TEXT",
        )

        # Create deterministic matching rule for each field
        deterministic_matching_rules.append(f"l.{field} = r.{field}")

        # String fields
        if "TEXT" in field_type.upper():
            if field_type != "TEXT" or field[0:3].isalpha():
                blocking_rules.append(
                    f"SUBSTR(l.{field}, 1, 3) = SUBSTR(r.{field}, 1, 3)"
                )
            comparisons.append(cl.JaroWinklerAtThresholds(field, [0.9, 0.7]))

        # Numeric fields
        elif any(
            t in field_type.upper() for t in ["INT", "FLOAT", "DECIMAL", "DOUBLE"]
        ):
            blocking_rules.append(f"CAST(l.{field} AS INT) = CAST(r.{field} AS INT)")
            comparisons.append(cl.ExactMatch(field))

        else:
            comparisons.append(cl.ExactMatch(field))

    # Create Splink settings
    linker_settings = SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=blocking_rules,
        comparisons=comparisons,
    )

    # Create training functions
    training_functions = [
        {
            "function": "estimate_probability_two_random_records_match",
            "arguments": {
                "deterministic_matching_rules": deterministic_matching_rules,
                "recall": 0.7,
            },
        },
        {
            "function": "estimate_u_using_random_sampling",
            "arguments": {"max_pairs": 1e4},
        },
    ]

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "linker_training_functions": training_functions,
        "linker_settings": linker_settings,
        "threshold": 0.01,
    }

    # Validate the settings dictionary
    SplinkSettings.model_validate(settings_dict)

    return settings_dict


PROBABILISTIC_LINKERS = [
    pytest.param(SplinkLinker, configure_splink_probabilistic, id="Splink"),
    pytest.param(
        WeightedDeterministicLinker,
        configure_weighted_probabilistic,
        id="WeightedDeterministic",
    ),
]

# Test cases


@pytest.mark.parametrize(("Linker", "configure_linker"), PROBABILISTIC_LINKERS)
@patch.object(Query, "run")
def test_probabilistic_scores_generation(
    mock_query_run: Mock, Linker: Linker, configure_linker: LinkerConfigurator
):
    """Test that linkers can generate varying probability scores."""

    # Create sources with variations
    features = (
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[
                SuffixRule(suffix=" Ltd"),
                SuffixRule(suffix=" Limited"),
                ReplaceRule(old="&", new="and"),
            ],
            drop_base=False,
        ),
        FeatureConfig(
            name="id_number",
            base_generator="numerify",
            parameters=(("text", "######"),),
        ),
    )

    configs = (
        SourceTestkitParameters(
            name="source_left", features=features, n_true_entities=10
        ),
        SourceTestkitParameters(
            name="source_right", features=features, n_true_entities=10
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    mock_query_run.side_effect = [
        pl.from_arrow(left_source.data),
        pl.from_arrow(right_source.data),
    ]

    # Configure and run the linker
    linker = Model(
        dag=linked.dag,
        name="prob_test_linker",
        description="Testing probability generation",
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left_query=Query(left_source, dag=linked.dag),
        right_query=Query(right_source, dag=linked.dag),
    )

    results: Results = linker.run()

    # Validate results against ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=left_source.entities,
        right_clusters=right_source.entities,
        sources=["source_left", "source_right"],
        threshold=0,
    )

    assert identical, f"Expected perfect results but got: {report}"

    # Validate results over a threshold as a subset of the ground truth
    identical, report = linked.diff_results(
        probabilities=results.probabilities,
        left_clusters=left_source.entities,
        right_clusters=right_source.entities,
        sources=["source_left", "source_right"],
        threshold=np.mean(results.probabilities["probability"]),
    )

    assert not identical, f"Expected imperfect results but got: {report}"
    # Expect subsets of matches where connections weren't made
    assert report["subset"] > 0
    # Expect no wrong or invalid matches (perfect possible but unlikely)
    assert report["wrong"] == 0
    assert report["invalid"] == 0
