from typing import Any

import pyarrow as pa
import pytest
from faker import Faker

from matchbox.common.factories.entities import (
    ClusterEntity,
    EntityReference,
    FeatureConfig,
    SourceEntity,
    diff_results,
    generate_entities,
    probabilities_to_results_entities,
)


def make_cluster_entity(id: int, dataset: str, pks: list[str]) -> ClusterEntity:
    """Helper to create a ClusterEntity with specified dataset and PKs."""
    return ClusterEntity(id=id, source_pks=EntityReference({dataset: frozenset(pks)}))


def make_source_entity(dataset: str, pks: list[str], base_val: str) -> SourceEntity:
    """Helper to create a SourceEntity with specified dataset and PKs."""
    entity = SourceEntity(base_values={"name": base_val})
    entity.add_source_reference(dataset, pks)
    return entity


@pytest.mark.parametrize(
    ("name", "pks"),
    (
        ("dataset1", frozenset({"1", "2", "3"})),
        ("dataset2", frozenset({"A", "B"})),
    ),
)
def test_entity_reference_creation(name: str, pks: frozenset[str]):
    """Test basic EntityReference creation and access."""
    ref = EntityReference({name: pks})
    assert ref[name] == pks
    assert name in ref
    with pytest.raises(KeyError):
        ref["nonexistent"]


def test_entity_reference_addition():
    """Test combining EntityReferences."""
    ref1 = EntityReference({"dataset1": frozenset({"1", "2"})})
    ref2 = EntityReference(
        {"dataset1": frozenset({"2", "3"}), "dataset2": frozenset({"A"})}
    )
    combined = ref1 + ref2
    assert combined["dataset1"] == frozenset({"1", "2", "3"})
    assert combined["dataset2"] == frozenset({"A"})


def test_entity_reference_subset():
    """Test subset relationships between EntityReferences."""
    subset = EntityReference({"dataset1": frozenset({"1", "2"})})
    superset = EntityReference(
        {"dataset1": frozenset({"1", "2", "3"}), "dataset2": frozenset({"A"})}
    )

    assert subset <= superset
    assert not superset <= subset


def test_cluster_entity_creation():
    """Test basic ClusterEntity functionality."""
    ref = EntityReference({"dataset1": frozenset({"1", "2"})})
    entity = ClusterEntity(source_pks=ref)

    assert entity.source_pks == ref
    assert isinstance(entity.id, int)


def test_cluster_entity_addition():
    """Test combining ClusterEntity objects."""
    entity1 = ClusterEntity(source_pks=EntityReference({"dataset1": frozenset({"1"})}))
    entity2 = ClusterEntity(source_pks=EntityReference({"dataset1": frozenset({"2"})}))

    combined = entity1 + entity2
    assert combined.source_pks["dataset1"] == frozenset({"1", "2"})


def test_source_entity_creation():
    """Test basic SourceEntity functionality."""
    base_values = {"name": "John", "age": 30}
    ref = EntityReference({"dataset1": frozenset({"1", "2"})})

    entity = SourceEntity(base_values=base_values, source_pks=ref)

    assert entity.base_values == base_values
    assert entity.source_pks == ref
    assert isinstance(entity.id, int)


@pytest.mark.parametrize(
    ("features", "n"),
    (
        ((FeatureConfig(name="name", base_generator="name"),), 1),
        (
            (
                FeatureConfig(name="name", base_generator="name"),
                FeatureConfig(name="email", base_generator="email"),
            ),
            5,
        ),
    ),
)
def test_generate_entities(features: tuple[FeatureConfig, ...], n: int):
    """Test entity generation with different features and counts."""
    faker = Faker(seed=42)
    entities = generate_entities(faker, features, n)

    assert len(entities) == n
    for entity in entities:
        # Check all features are present
        assert all(f.name in entity.base_values for f in features)
        # Check all values are strings (given our test features)
        assert all(isinstance(v, str) for v in entity.base_values.values())


@pytest.mark.parametrize(
    (
        "probabilities",
        "left_clusters",
        "right_clusters",
        "threshold",
        "expected_count",
    ),
    [
        pytest.param(
            pa.table(
                {
                    "left_id": [1, 2],
                    "right_id": [2, 3],
                    "probability": [90, 85],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
                make_cluster_entity(3, "test", ["a3"]),
            ),
            None,
            80,
            1,  # One merged entity containing all three records
            id="basic_dedupe_chain",
        ),
        pytest.param(
            pa.table(
                {
                    "left_id": [1],
                    "right_id": [4],
                    "probability": [95],
                }
            ),
            (make_cluster_entity(1, "left", ["a1"]),),
            (make_cluster_entity(4, "right", ["b1"]),),
            0.9,
            1,  # One merged entity from the link
            id="basic_link_match",
        ),
        pytest.param(
            pa.table(
                {
                    "left_id": [1, 2],
                    "right_id": [2, 3],
                    "probability": [75, 70],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
                make_cluster_entity(3, "test", ["a3"]),
            ),
            None,
            80,
            3,  # No merging due to threshold
            id="threshold_prevents_merge",
        ),
        pytest.param(
            pa.table(
                {
                    "left_id": [],
                    "right_id": [],
                    "probability": [],
                }
            ),
            (
                make_cluster_entity(1, "test", ["a1"]),
                make_cluster_entity(2, "test", ["a2"]),
            ),
            None,
            80,
            2,  # No merging with empty probabilities
            id="empty_probabilities",
        ),
    ],
)
def test_probabilities_to_results_entities(
    probabilities: pa.Table,
    left_clusters: tuple[ClusterEntity, ...],
    right_clusters: tuple[ClusterEntity, ...] | None,
    threshold: float,
    expected_count: int,
) -> None:
    """Test probabilities_to_results_entities with various scenarios."""
    result = probabilities_to_results_entities(
        probabilities=probabilities,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        threshold=threshold,
    )

    assert len(result) == expected_count

    # For merging cases, verify all input entities are contained in the output
    all_inputs = set(left_clusters)
    if right_clusters:
        all_inputs.update(right_clusters)

    for input_entity in all_inputs:
        # Each input entity should be contained within one of the output entities
        assert any(input_entity in output_entity for output_entity in result)


def assert_deep_approx_equal(got: float | dict | list, want: float | dict | list):
    """Compare nested structures with approximate equality for floats."""
    # Handle float comparison
    if isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-2)
        return

    # Handle dictionary comparison
    if isinstance(want, dict):
        assert isinstance(got, dict)
        assert set(want.keys()) <= set(got.keys())  # All expected keys must exist
        for k, v in want.items():
            assert_deep_approx_equal(got[k], v)
        return

    # Handle list comparison
    if isinstance(want, list):
        assert isinstance(got, list)
        assert len(got) == len(want)

        # Sort lists of dictionaries by ID fields for easier comparison
        if want and all(isinstance(x, dict) for x in want + got):
            for id_key in ["entity_id", "expected_entity_id", "actual_entity_id"]:
                if all(id_key in x for x in want + got):
                    got = sorted(got, key=lambda x: x[id_key])
                    want = sorted(want, key=lambda x: x[id_key])
                    break

        for w, g in zip(want, got, strict=True):
            assert_deep_approx_equal(g, w)
        return

    # Direct comparison for all other types
    assert got == want


@pytest.mark.parametrize(
    ("expected", "actual", "verbose", "want_identical", "want_result"),
    [
        # Identical sets
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [make_cluster_entity(1, "d1", ["1", "2"])],
            False,
            True,
            {},
            id="identical_sets",
        ),
        # Completely missing entity
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [],
            True,
            False,
            {
                "perfect_matches": [],
                "fragmented_matches": [],
                "unexpected_matches": [],
                "missing_matches": [
                    {
                        "expected_entity_id": 1,
                        "source_pks": {"d1": frozenset(["1", "2"])},
                    }
                ],
                "metrics": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "fragmentation": 0.0,
                    "similarity": 0.0,
                },
            },
            id="completely_missing_entity",
        ),
        # Extra entity
        pytest.param(
            [],
            [make_cluster_entity(2, "d1", ["1", "2"])],
            True,
            False,
            {
                "perfect_matches": [],
                "fragmented_matches": [],
                # Not really an "unexpected merge" since there are no expected entities
                "unexpected_matches": [],
                "missing_matches": [],
                "metrics": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "fragmentation": 0.0,
                    "similarity": 0.0,
                },
            },
            id="extra_entity",
        ),
        # Fragmented match
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2", "3"])],
            [make_cluster_entity(2, "d1", ["1", "2", "4"])],
            True,
            False,
            {
                "perfect_matches": [],
                "fragmented_matches": [
                    {
                        "expected_entity_id": 1,
                        "expected_source_pks": {"d1": frozenset(["1", "2", "3"])},
                        "coverage_ratio": 2 / 3,  # 2 of 3 keys are covered
                        "missing_pks": {"d1": frozenset(["3"])},
                        "fragments": [
                            {
                                "actual_entity_id": 2,
                                "source_pks": {"d1": frozenset(["1", "2", "4"])},
                                # 2 common keys out of 4 total (Jaccard similarity)
                                "similarity": 0.5,
                            }
                        ],
                    }
                ],
                "unexpected_matches": [],
                "missing_matches": [],
                "metrics": {
                    "precision": 0.0,  # No perfect matches
                    "recall": 0.0,  # No perfect matches
                    "f1": 0.0,
                    "fragmentation": 1.0,  # 1 fragment per expected entity
                    "similarity": 2 / 3,  # Coverage ratio
                },
            },
            id="fragmented_match",
        ),
        # Complex scenario with fragmented and missing
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3", "4"]),
                make_cluster_entity(3, "d1", ["5", "6"]),
            ],
            [
                make_cluster_entity(4, "d1", ["1", "7"]),
                make_cluster_entity(5, "d1", ["8", "9"]),
            ],
            True,
            False,
            {
                "perfect_matches": [],
                "fragmented_matches": [
                    {
                        "expected_entity_id": 1,
                        "expected_source_pks": {"d1": frozenset(["1", "2"])},
                        "coverage_ratio": 0.5,  # 1 of 2 keys are covered
                        "missing_pks": {"d1": frozenset(["2"])},
                        "fragments": [
                            {
                                "actual_entity_id": 4,
                                "source_pks": {"d1": frozenset(["1", "7"])},
                                "similarity": 1 / 3,  # 1 common key out of 3 total keys
                            }
                        ],
                    }
                ],
                "unexpected_matches": [],
                "missing_matches": [
                    {
                        "expected_entity_id": 2,
                        "source_pks": {"d1": frozenset(["3", "4"])},
                    },
                    {
                        "expected_entity_id": 3,
                        "source_pks": {"d1": frozenset(["5", "6"])},
                    },
                ],
                "metrics": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "fragmentation": 1.0,
                    "similarity": pytest.approx(1 / 6, rel=1e-2),  # (0.5 + 0 + 0)/3
                },
            },
            id="complex_scenario",
        ),
        # Non-verbose mode (only shows metrics)
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2", "3"])],
            [make_cluster_entity(2, "d1", ["1", "2", "4"])],
            False,
            False,
            {
                "metrics": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "fragmentation": 1.0,
                    "similarity": 2 / 3,
                }
            },
            id="non_verbose_mode",
        ),
        # Mixed identical and fragmented entities
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),  # Identical entity
                make_cluster_entity(2, "d1", ["3", "4"]),  # Will be partially matched
            ],
            [
                make_cluster_entity(1, "d1", ["1", "2"]),  # Same as in expected
                make_cluster_entity(3, "d1", ["3", "5"]),  # Partial match with entity 2
            ],
            True,
            False,
            {
                "perfect_matches": [
                    {"entity_id": 1, "source_pks": {"d1": frozenset(["1", "2"])}}
                ],
                "fragmented_matches": [
                    {
                        "expected_entity_id": 2,
                        "expected_source_pks": {"d1": frozenset(["3", "4"])},
                        "coverage_ratio": 0.5,  # 1 of 2 keys are covered
                        "missing_pks": {"d1": frozenset(["4"])},
                        "fragments": [
                            {
                                "actual_entity_id": 3,
                                "source_pks": {"d1": frozenset(["3", "5"])},
                                "similarity": 1 / 3,  # 1 common key out of 3 total
                            }
                        ],
                    }
                ],
                "unexpected_matches": [],
                "missing_matches": [],
                "metrics": {
                    "precision": 0.5,  # 1 perfect out of 2 total
                    "recall": 0.5,  # 1 perfect out of 2 expected
                    "f1": 0.5,
                    "fragmentation": 1.0,
                    "similarity": 0.75,  # (1.0 + 0.5) / 2
                },
            },
            id="mixed_identical_and_fragmented",
        ),
        # Fragmentation - multiple fragments per expected entity
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2", "3", "4"])],
            [
                make_cluster_entity(2, "d1", ["1", "2"]),
                make_cluster_entity(3, "d1", ["3", "4"]),
            ],
            True,
            False,
            {
                "perfect_matches": [],
                "fragmented_matches": [
                    {
                        "expected_entity_id": 1,
                        "expected_source_pks": {"d1": frozenset(["1", "2", "3", "4"])},
                        "coverage_ratio": 1.0,  # All keys are covered by fragments
                        "missing_pks": {},
                        "fragments": [
                            {
                                "actual_entity_id": 2,
                                "source_pks": {"d1": frozenset(["1", "2"])},
                                "similarity": 0.5,  # 2 common keys out of 4 total
                            },
                            {
                                "actual_entity_id": 3,
                                "source_pks": {"d1": frozenset(["3", "4"])},
                                "similarity": 0.5,  # 2 common keys out of 4 total
                            },
                        ],
                    }
                ],
                "unexpected_matches": [],
                "missing_matches": [],
                "metrics": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "fragmentation": 2.0,  # 2 fragments for 1 expected entity
                    "similarity": 1.0,  # Full coverage
                },
            },
            id="high_threshold_creates_subsets",
        ),
        # Unexpected merge - one actual entity contains multiple expected entities
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3", "4"]),
            ],
            [make_cluster_entity(3, "d1", ["1", "2", "3", "4", "5"])],
            True,
            False,
            {
                "perfect_matches": [],
                # Both expected entities should also appear in fragmented_matches
                "fragmented_matches": [
                    {
                        "expected_entity_id": 1,
                        "expected_source_pks": {"d1": frozenset(["1", "2"])},
                        "coverage_ratio": 1.0,  # All keys are covered
                        "missing_pks": {},
                        "fragments": [
                            {
                                "actual_entity_id": 3,
                                "source_pks": {
                                    "d1": frozenset(["1", "2", "3", "4", "5"])
                                },
                                "similarity": 2 / 5,  # 2 common keys out of 5 total
                            }
                        ],
                    },
                    {
                        "expected_entity_id": 2,
                        "expected_source_pks": {"d1": frozenset(["3", "4"])},
                        "coverage_ratio": 1.0,  # All keys are covered
                        "missing_pks": {},
                        "fragments": [
                            {
                                "actual_entity_id": 3,
                                "source_pks": {
                                    "d1": frozenset(["1", "2", "3", "4", "5"])
                                },
                                "similarity": 2 / 5,  # 2 common keys out of 5 total
                            }
                        ],
                    },
                ],
                "unexpected_matches": [
                    {
                        "actual_entity_id": 3,
                        "actual_source_pks": {
                            "d1": frozenset(["1", "2", "3", "4", "5"])
                        },
                        "extra_pks": {"d1": frozenset(["5"])},
                        "merged_entities": [
                            {
                                "expected_entity_id": 1,
                                "source_pks": {"d1": frozenset(["1", "2"])},
                                "similarity": 2 / 5,  # 2 common keys out of 5 total
                            },
                            {
                                "expected_entity_id": 2,
                                "source_pks": {"d1": frozenset(["3", "4"])},
                                "similarity": 2 / 5,  # 2 common keys out of 5 total
                            },
                        ],
                    }
                ],
                "missing_matches": [],
                "metrics": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "fragmentation": 1.0,  # Because each expected entity has 1 fragment
                    "similarity": 0.7,  # (1.0 + 1.0 + 0.4 + 0.4) / 4 = 0.7
                },
            },
            id="unexpected_merge",
        ),
    ],
)
def test_diff_results(
    expected: list[ClusterEntity],
    actual: list[ClusterEntity],
    verbose: bool,
    want_identical: bool,
    want_result: dict[str, Any],
):
    """Test diff_results function handles various scenarios correctly."""
    got_identical, got_result = diff_results(expected, actual, verbose)

    assert got_identical == want_identical

    # If identical, result should be empty
    if got_identical:
        assert got_result == {}
    else:
        if not verbose:
            # For non-verbose, just check metrics
            assert "metrics" in got_result
            assert_deep_approx_equal(got_result["metrics"], want_result["metrics"])
        else:
            # For verbose mode, check the entire structure
            assert_deep_approx_equal(got_result, want_result)


def test_source_to_results_conversion():
    """Test converting source entities to cluster entities and comparing them."""
    # Create source entity present in multiple datasets
    source = SourceEntity(
        base_values={"name": "Test"},
        source_pks=EntityReference(
            {"dataset1": frozenset({"1", "2"}), "dataset2": frozenset({"A", "B"})}
        ),
    )

    # Convert different subsets to cluster entities
    results1 = source.to_cluster_entity("dataset1")
    results2 = source.to_cluster_entity("dataset1", "dataset2")
    results3 = source.to_cluster_entity("dataset2")

    # Test different comparison scenarios
    identical, report = diff_results([results1], [results1])
    assert identical
    assert report == {}

    # Compare partial overlap
    identical, report = diff_results([results1], [results2])
    assert not identical
    assert "dataset2" in str(results2 - results1)

    # Compare disjoint sets
    identical, report = diff_results([results1], [results3])
    assert not identical
    assert results1.similarity_ratio(results3) == 0.0

    # Test missing dataset returns None
    assert source.to_cluster_entity("nonexistent") is None


@pytest.mark.parametrize(
    ("base_generator", "expected_type"),
    [
        pytest.param("name", "TEXT", id="text_generator"),
        pytest.param("random_int", "INTEGER", id="integer_generator"),
        pytest.param("date_this_decade", "DATE", id="date_generator"),
    ],
)
def test_feature_config_sql_type_inference(
    base_generator: str, expected_type: str
) -> None:
    """Test that SQL types are correctly inferred from feature configurations."""
    feature_config = FeatureConfig(name=base_generator, base_generator=base_generator)
    assert feature_config.sql_type == expected_type
