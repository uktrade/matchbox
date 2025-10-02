from typing import Any

import polars as pl
import pytest
from faker import Faker

from matchbox.common.dtos import DataTypes, SourceResolutionName
from matchbox.common.factories.entities import (
    ClusterEntity,
    EntityReference,
    FeatureConfig,
    SourceEntity,
    diff_results,
    generate_entities,
    probabilities_to_results_entities,
)


def make_cluster_entity(id: int, *args) -> ClusterEntity:
    """Helper to create a ClusterEntity.

    Args:
        id: Entity ID
        *args: Variable arguments in pairs of (source_name, keys_list)
            e.g., "d1", ["1", "2"], "d2", ["3", "4"]

    Returns:
        ClusterEntity with the specified sources and primary keys
    """
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be pairs of source name and keys list")

    keys = {}
    for i in range(0, len(args), 2):
        source = args[i]
        keys_list = args[i + 1]
        if not isinstance(source, str):
            raise TypeError(f"source name must be a string, got {type(source)}")
        if not isinstance(keys_list, list):
            raise TypeError(f"keys must be a list, got {type(keys_list)}")
        keys[source] = frozenset(keys_list)

    return ClusterEntity(id=id, keys=EntityReference(keys))


def make_source_entity(
    source: SourceResolutionName, keys: list[str], base_val: str
) -> SourceEntity:
    """Helper to create a SourceEntity."""
    entity = SourceEntity(base_values={"name": base_val})
    entity.add_source_reference(source, keys)
    return entity


@pytest.mark.parametrize(
    ("name", "keys"),
    (
        ("source1", frozenset({"1", "2", "3"})),
        ("source2", frozenset({"A", "B"})),
    ),
)
def test_entity_reference_creation(name: SourceResolutionName, keys: frozenset[str]):
    """Test basic EntityReference creation and access."""
    ref = EntityReference({name: keys})
    assert ref[name] == keys
    assert name in ref
    with pytest.raises(KeyError):
        ref["nonexistent"]


def test_entity_reference_addition():
    """Test combining EntityReferences."""
    ref1 = EntityReference({"source1": frozenset({"1", "2"})})
    ref2 = EntityReference(
        {"source1": frozenset({"2", "3"}), "source2": frozenset({"A"})}
    )
    combined = ref1 + ref2
    assert combined["source1"] == frozenset({"1", "2", "3"})
    assert combined["source2"] == frozenset({"A"})


def test_entity_reference_subset():
    """Test subset relationships between EntityReferences."""
    subset = EntityReference({"source1": frozenset({"1", "2"})})
    superset = EntityReference(
        {"source1": frozenset({"1", "2", "3"}), "source2": frozenset({"A"})}
    )

    assert subset <= superset
    assert not superset <= subset


def test_cluster_entity_creation():
    """Test basic ClusterEntity functionality."""
    ref = EntityReference({"source1": frozenset({"1", "2"})})
    entity = ClusterEntity(keys=ref)

    assert entity.keys == ref
    assert isinstance(entity.id, int)


def test_cluster_entity_addition():
    """Test combining ClusterEntity objects."""
    entity1 = ClusterEntity(keys=EntityReference({"source1": frozenset({"1"})}))
    entity2 = ClusterEntity(keys=EntityReference({"source1": frozenset({"2"})}))

    combined = entity1 + entity2
    assert combined.keys["source1"] == frozenset({"1", "2"})


def test_source_entity_creation():
    """Test basic SourceEntity functionality."""
    base_values = {"name": "John", "age": 30}
    ref = EntityReference({"source1": frozenset({"1", "2"})})

    entity = SourceEntity(base_values=base_values, keys=ref)

    assert entity.base_values == base_values
    assert entity.keys == ref
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
            pl.DataFrame(
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
            pl.DataFrame(
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
            pl.DataFrame(
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
            pl.DataFrame(
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
    probabilities: pl.DataFrame,
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
    ("expected", "actual", "want_identical", "want_result"),
    [
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [make_cluster_entity(2, "d1", ["1", "2"])],
            True,
            {},
            id="identical_sets",
        ),
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3", "4"]),
            ],
            [make_cluster_entity(3, "d1", ["2", "3"])],
            False,
            {
                "perfect": 0,
                "subset": 0,
                "superset": 0,
                "wrong": 1,
                "invalid": 0,
            },
            id="completely_different_sets",
        ),
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2", "3"])],
            [make_cluster_entity(2, "d1", ["1", "2"])],
            False,
            {
                "perfect": 0,
                "subset": 1,
                "superset": 0,
                "wrong": 0,
                "invalid": 0,
            },
            id="subset_match",
        ),
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3"]),
            ],
            [make_cluster_entity(3, "d1", ["1", "2", "3"])],
            False,
            {
                "perfect": 0,
                "subset": 0,
                "superset": 1,
                "wrong": 0,
                "invalid": 0,
            },
            id="superset_match",
        ),
        pytest.param(
            [make_cluster_entity(1, "d1", ["1", "2"])],
            [make_cluster_entity(2, "d1", ["1", "2", "3", "4"])],
            False,
            {
                "perfect": 0,
                "subset": 0,
                "superset": 0,
                "wrong": 0,
                "invalid": 1,
            },
            id="invalid_entity",
        ),
        pytest.param(
            [
                make_cluster_entity(1, "d1", ["1", "2"]),
                make_cluster_entity(2, "d1", ["3", "4"]),
            ],
            [
                make_cluster_entity(3, "d1", ["1", "2"]),  # perfect match
                make_cluster_entity(4, "d1", ["3"]),  # subset
                make_cluster_entity(4, "d1", ["1", "2", "3"]),  # superset
                make_cluster_entity(5, "d1", ["2", "3"]),  # wrong
                make_cluster_entity(6, "d1", ["7", "8", "9"]),  # invalid
            ],
            False,
            {
                "perfect": 1,
                "subset": 1,
                "superset": 1,
                "wrong": 1,
                "invalid": 1,
            },
            id="mixed_scenario",
        ),
    ],
)
def test_diff_results(
    expected: list[ClusterEntity],
    actual: list[ClusterEntity],
    want_identical: bool,
    want_result: dict[str, Any],
):
    """Test diff_results function handles various scenarios correctly."""
    got_identical, got_result = diff_results(expected, actual)

    assert got_identical == want_identical
    assert dict(got_result) == want_result


def test_source_to_results_conversion():
    """Test converting source entities to cluster entities and comparing them."""
    # Create source entity present in multiple sources
    source = SourceEntity(
        base_values={"name": "Test"},
        keys=EntityReference(
            {"source1": frozenset({"1", "2"}), "source2": frozenset({"A", "B"})}
        ),
    )

    # Convert different subsets to cluster entities
    results1 = source.to_cluster_entity("source1")
    results2 = source.to_cluster_entity("source1", "source2")
    results3 = source.to_cluster_entity("source2")

    # Test different comparison scenarios
    identical, report = diff_results([results1], [results1])
    assert identical
    assert report == {}

    # Compare partial overlap
    identical, report = diff_results([results1], [results2])
    assert not identical
    assert "source2" in str(results2 - results1)

    # Compare disjoint sets
    identical, report = diff_results([results1], [results3])
    assert not identical
    assert results1.similarity_ratio(results3) == 0.0

    # Test missing source returns None
    assert source.to_cluster_entity("nonexistent") is None


@pytest.mark.parametrize(
    ("base_generator", "expected_type"),
    [
        pytest.param("name", DataTypes.STRING, id="text_generator"),
        pytest.param("random_int", DataTypes.INT64, id="integer_generator"),
        pytest.param("date_this_decade", DataTypes.DATE, id="date_generator"),
    ],
)
def test_feature_config_datatype_inference(
    base_generator: str, expected_type: str
) -> None:
    """Test that SQL types are correctly inferred from feature configurations."""
    feature_config = FeatureConfig(name=base_generator, base_generator=base_generator)
    assert feature_config.datatype == expected_type
