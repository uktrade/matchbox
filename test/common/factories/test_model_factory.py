from typing import Any, Literal

import pytest

from matchbox.client.queries import Query
from matchbox.common.arrow import SCHEMA_RESULTS
from matchbox.common.factories.models import (
    generate_dummy_probabilities,
    model_factory,
    query_to_model_factory,
)
from matchbox.common.factories.sources import linked_sources_factory, source_factory
from matchbox.common.graph import ModelResolutionName, ResolutionName


def test_model_factory_entity_preservation():
    """Test that model_factory preserves keys with incomplete probabilities."""
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)

    # Create first model
    first_model = model_factory(
        left_testkit=linked.sources["crn"],
        true_entities=all_true_sources[:1],  # Just one source entity
    )

    # Record input entities for second model
    input_entities = set(first_model.entities)
    assert len(input_entities) > 0

    # Create second model with no matching true entities
    second_model = model_factory(
        left_testkit=first_model,
        true_entities=all_true_sources[1:],  # Different source entities
    )

    # Even with no probabilities possible, should describe same keys
    assert sum(second_model.entities) == sum(first_model.entities)


@pytest.mark.parametrize(
    ("left_testkit", "right_testkit", "expected_type", "should_have_right"),
    [
        pytest.param(None, None, "deduper", False, id="default_creates_deduper"),
        pytest.param(
            "source", None, "deduper", False, id="left_source_only_creates_deduper"
        ),
        pytest.param(
            "source", "source", "linker", True, id="both_sources_creates_linker"
        ),
        pytest.param("model", None, "deduper", False, id="left_model_creates_deduper"),
        pytest.param(
            "model", "source", "linker", True, id="mixed_types_creates_linker"
        ),
    ],
)
def test_model_type_creation(
    left_testkit: None | str,
    right_testkit: None | str,
    expected_type: str,
    should_have_right: bool,
):
    """Test that model creation and core operations work correctly for each type."""
    # Create our source objects from the string parameters
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)
    half_true_sources = all_true_sources[: len(all_true_sources) // 2]

    if left_testkit == "source":
        left = linked.sources["crn"]
    elif left_testkit == "model":
        left = model_factory(
            left_testkit=linked.sources["crn"], true_entities=half_true_sources
        )
    else:
        left = None

    if right_testkit == "source":
        right = linked.sources["cdms"]
    elif right_testkit == "model":
        right = model_factory(
            left_testkit=linked.sources["cdms"], true_entities=half_true_sources
        )
    else:
        right = None

    # Create our model
    model = model_factory(
        left_testkit=left, right_testkit=right, true_entities=all_true_sources, seed=13
    )

    # Basic type verification
    assert model.model.config.type == expected_type
    assert (model.right_query is not None) == should_have_right
    assert (model.right_clusters is not None) == should_have_right

    # Verify probabilities were generated
    assert len(model.probabilities) > 0
    assert model.probabilities.schema.equals(SCHEMA_RESULTS)

    # Test threshold setting and querying
    initial_threshold = 80
    model.threshold = initial_threshold
    initial_data = model.data
    initial_ids = set(initial_data["id"].to_pylist())
    assert len(initial_ids) > 0

    # Test threshold change affects results
    new_threshold = 90
    model.threshold = new_threshold
    new_data = model.data
    new_ids = set(new_data["id"].to_pylist())

    # Higher threshold should result in more distinct entities, as fewer merge
    assert len(new_ids) >= len(initial_ids)

    # Verify schema consistency
    assert initial_data.schema == new_data.schema
    assert "id" in initial_data.column_names

    # For linkers, verify we maintain separation between left and right IDs
    if expected_type == "linker":
        left_ids = set(model.left_data["id"].to_pylist())
        right_ids = set(model.right_data["id"].to_pylist())
        assert not (left_ids & right_ids), (
            "Left and right IDs should be disjoint in linker"
        )

        prob_left_ids = set(model.probabilities["left_id"].to_pylist())
        prob_right_ids = set(model.probabilities["right_id"].to_pylist())
        assert prob_left_ids <= left_ids, (
            "Probability left IDs should be subset of left IDs"
        )
        assert prob_right_ids <= right_ids, (
            "Probability right IDs should be subset of right IDs"
        )


@pytest.mark.parametrize(
    ("left_testkit", "right_testkit", "model_type"),
    [
        pytest.param(
            "crn",
            None,
            "deduper",
            id="test_initial_deduper_methodology",
        ),
        pytest.param(
            "cdms",
            None,
            "deduper",
            id="test_second_deduper_methodology",
        ),
        pytest.param(
            "crn",
            "cdms",
            "linker",
            id="test_final_linker_methodology",
        ),
    ],
)
def test_model_pipeline_with_dummy_methodology(
    left_testkit: ResolutionName,
    right_testkit: ResolutionName | None,
    model_type: Literal["deduper", "linker"],
) -> None:
    """Tests the factories validate "real" methodologies in various pipeline positions.

    Here we show that with just a single output of a probabilities table, the factory
    and testkit system lets you evaluate the methodology of a deduper or linker.

    This test demonstrates that:
    1. We can set up pipelines in various configurations that work perfectly
        with model_factory
    2. When we swap in a simulated "real" methodology (using
        generate_dummy_probabilities), the diff can detect the errors appropriately
    3. This validation works across different pipeline positions and configurations
    """
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)

    # Create and validate perfect model
    if model_type == "deduper":
        # Get inputs to final model for later diff
        left_clusters = linked.sources[left_testkit].entities
        right_clusters = None

        perfect_model = model_factory(
            left_testkit=linked.sources[left_testkit],
            true_entities=all_true_sources,
        )
        sources = [left_testkit]
        model_entities = (tuple(linked.sources[left_testkit].entities), None)
    else:  # linker
        # Create perfect deduped models first
        left_deduped = model_factory(
            left_testkit=linked.sources[left_testkit],
            true_entities=all_true_sources,
        )
        right_deduped = model_factory(
            left_testkit=linked.sources[right_testkit],
            true_entities=all_true_sources,
        )

        # Get inputs to final model for later diff
        left_clusters = left_deduped.entities
        right_clusters = right_deduped.entities

        perfect_model = model_factory(
            left_testkit=left_deduped,
            right_testkit=right_deduped,
            true_entities=all_true_sources,
        )
        sources = [left_testkit, right_testkit]
        model_entities = (tuple(left_deduped.entities), tuple(right_deduped.entities))

    # Verify perfect model works
    identical, _ = linked.diff_results(
        probabilities=perfect_model.probabilities,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        sources=sources,
        threshold=0,
    )
    assert identical, "Perfect model_factory setup should match"

    # Test with imperfect methodology
    random_probabilities = generate_dummy_probabilities(
        left_values=model_entities[0],
        right_values=model_entities[1],
        prob_range=(0.0, 1.0),
        num_components=len(all_true_sources) - 1,  # Intentionally wrong
    )

    identical, report = linked.diff_results(
        probabilities=random_probabilities,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        sources=sources,
        threshold=0,
    )

    # Verify the imperfect methodology was detected
    assert not identical
    # Random process: can't guarantee particular problems, but can guarantee
    # that some will be present
    assert report["wrong"] > 0 or report["subset"] > 0 or report["superset"] > 0


@pytest.mark.parametrize(
    ("kwargs", "expected_error", "expected_message"),
    [
        pytest.param(
            {"model_type": "deduper", "prob_range": (0.9, 0.8)},
            ValueError,
            "Probabilities must be increasing values between 0 and 1",
            id="invalid_prob_range_decreasing",
        ),
        pytest.param(
            {"model_type": "deduper", "prob_range": (-0.1, 0.8)},
            ValueError,
            "Probabilities must be increasing values between 0 and 1",
            id="invalid_prob_range_negative",
        ),
        pytest.param(
            {"model_type": "deduper", "prob_range": (0.8, 1.1)},
            ValueError,
            "Probabilities must be increasing values between 0 and 1",
            id="invalid_prob_range_too_high",
        ),
        pytest.param(
            {"left_testkit": source_factory(), "true_entities": None},
            ValueError,
            "Must provide true entities when sources are given",
            id="missing_true_entities_with_source",
        ),
    ],
)
def test_model_factory_validation(
    kwargs: dict[str, Any], expected_error: type[Exception], expected_message: str
):
    """Test that model_factory validates inputs correctly."""
    with pytest.raises(expected_error, match=expected_message):
        model_factory(**kwargs)


@pytest.mark.parametrize(
    (
        "name",
        "description",
        "model_type",
        "n_true_entities",
        "prob_range",
        "seed",
        "expected_checks",
    ),
    [
        pytest.param(
            "basic_deduper",
            "Basic deduplication model",
            "deduper",
            5,
            (0.8, 0.9),
            42,
            {
                "type": "deduper",
                "entity_count": 5,
                "has_right": False,
                "prob_min": 0.8,
                "prob_max": 0.9,
            },
            id="basic_deduper",
        ),
        pytest.param(
            "basic_linker",
            "Basic linking model",
            "linker",
            10,
            (0.7, 0.8),
            42,
            {
                "type": "linker",
                "entity_count": 10,
                "has_right": True,
                "prob_min": 0.7,
                "prob_max": 0.8,
            },
            id="basic_linker",
        ),
        pytest.param(
            "large_deduper",
            "Deduper with many entities",
            "deduper",
            100,
            (0.9, 1.0),
            42,
            {
                "type": "deduper",
                "entity_count": 100,
                "has_right": False,
                "prob_min": 0.9,
                "prob_max": 1.0,
            },
            id="large_deduper",
        ),
        pytest.param(
            "strict_linker",
            "Linker with high probability threshold",
            "linker",
            20,
            (0.95, 1.0),
            42,
            {
                "type": "linker",
                "entity_count": 20,
                "has_right": True,
                "prob_min": 0.95,
                "prob_max": 1.0,
            },
            id="strict_linker",
        ),
    ],
)
def test_model_factory_basic_creation(
    name: ModelResolutionName,
    description: str,
    model_type: str,
    n_true_entities: int,
    prob_range: tuple[float, float],
    seed: int,
    expected_checks: dict,
) -> None:
    """Test basic model factory creation without sources."""
    model = model_factory(
        name=name,
        description=description,
        model_type=model_type,
        n_true_entities=n_true_entities,
        prob_range=prob_range,
        seed=seed,
    )

    # Basic metadata checks
    assert model.model.name == name
    assert model.model.description == description
    assert str(model.model.config.type) == expected_checks["type"]

    # Structure checks
    assert (model.right_query is not None) == expected_checks["has_right"]
    assert (model.right_clusters is not None) == expected_checks["has_right"]
    assert len(model.entities) == expected_checks["entity_count"]

    # Probability checks
    probs = model.probabilities["probability"].to_numpy() / 100
    assert all(p >= expected_checks["prob_min"] for p in probs)
    assert all(p <= expected_checks["prob_max"] for p in probs)


@pytest.mark.parametrize(
    ("source_config", "expected_checks"),
    [
        pytest.param(
            {
                "left_name": "crn",
                "right_name": None,
                "true_entities_slice": slice(None),  # All entities
                "prob_range": (0.8, 0.9),
            },
            {
                "type": "deduper",
                "has_right": False,
                "prob_min": 0.8,
                "prob_max": 0.9,
            },
            id="deduper_full_entities",
        ),
        pytest.param(
            {
                "left_name": "crn",
                "right_name": "cdms",
                "true_entities_slice": slice(None),
                "prob_range": (0.8, 0.9),
            },
            {
                "type": "linker",
                "has_right": True,
                "prob_min": 0.8,
                "prob_max": 0.9,
            },
            id="linker_full_entities",
        ),
        pytest.param(
            {
                "left_name": "crn",
                "right_name": None,
                "true_entities_slice": slice(0, 1),  # Just first entity
                "prob_range": (0.9, 1.0),
            },
            {
                "type": "deduper",
                "has_right": False,
                "prob_min": 0.9,
                "prob_max": 1.0,
            },
            id="deduper_partial_entities",
        ),
        pytest.param(
            {
                "left_name": "crn",
                "right_name": "cdms",
                "true_entities_slice": slice(0, 2),  # First two entities
                "prob_range": (0.7, 0.8),
            },
            {
                "type": "linker",
                "has_right": True,
                "prob_min": 0.7,
                "prob_max": 0.8,
            },
            id="linker_partial_entities",
        ),
    ],
)
def test_model_factory_with_sources(source_config: dict, expected_checks: dict) -> None:
    """Test model factory creation using sources."""
    # Create source data
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)

    # Get sources based on config
    left_testkit = linked.sources[source_config["left_name"]]
    right_testkit = (
        linked.sources[source_config["right_name"]]
        if source_config["right_name"]
        else None
    )

    # Create model
    model = model_factory(
        left_testkit=left_testkit,
        right_testkit=right_testkit,
        true_entities=all_true_sources[source_config["true_entities_slice"]],
        prob_range=source_config["prob_range"],
    )

    # Basic type checks
    assert str(model.model.config.type) == expected_checks["type"]
    assert (model.right_query is not None) == expected_checks["has_right"]
    assert (model.right_clusters is not None) == expected_checks["has_right"]

    # Verify probabilities
    probs = model.probabilities["probability"].to_numpy() / 100
    assert all(p >= expected_checks["prob_min"] for p in probs)
    assert all(p <= expected_checks["prob_max"] for p in probs)

    # Verify source keys are preserved
    input_keys = sum(
        set(left_testkit.entities)
        | set(right_testkit.entities if right_testkit else {})
    )
    assert input_keys == sum(model.entities), (
        "Model entities should preserve all source keys"
    )


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal"),
    [
        pytest.param(42, 42, True, id="same_seeds"),
        pytest.param(1, 2, False, id="different_seeds"),
    ],
)
def test_model_factory_seed_behavior(seed1: int, seed2: int, should_be_equal: bool):
    """Test that model_factory handles seeds correctly for reproducibility."""
    dummy1 = model_factory(seed=seed1)
    dummy2 = model_factory(seed=seed2)

    if should_be_equal:
        assert dummy1.model.name == dummy2.model.name
        assert dummy1.model.description == dummy2.model.description
        assert dummy1.left_data.equals(dummy2.left_data)
        assert set(dummy1.left_clusters) == set(dummy2.left_clusters)
        assert set(dummy1.entities) == set(dummy2.entities)
        assert dummy1.probabilities.equals(dummy2.probabilities)
    else:
        assert dummy1.model.name != dummy2.model.name
        assert dummy1.model.description != dummy2.model.description
        assert not dummy1.left_data.equals(dummy2.left_data)
        assert set(dummy1.left_clusters) != set(dummy2.left_clusters)
        assert set(dummy1.entities) != set(dummy2.entities)
        assert not dummy1.probabilities.equals(dummy2.probabilities)


def test_query_to_model_factory_validation():
    """Test validation in query_to_model_factory."""
    # Create test resources using existing factory
    linked = linked_sources_factory()
    left_testkit = linked.sources["crn"]
    true_entities = tuple(linked.true_entities)

    # Extract query and keys for our function
    left_data = left_testkit.data
    left_keys = {"crn": "key"}

    # Test invalid probability range
    with pytest.raises(ValueError, match="Probabilities must be increasing values"):
        query_to_model_factory(
            left_query=Query(left_testkit.source, dag=linked.dag),
            left_data=left_data,
            left_keys=left_keys,
            true_entities=true_entities,
            prob_range=(0.9, 0.8),
        )

    # Test inconsistent right-side arguments
    with pytest.raises(ValueError, match="all of right_"):
        query_to_model_factory(
            left_query=Query(left_testkit.source, dag=linked.dag),
            left_data=left_data,
            left_keys=left_keys,
            true_entities=true_entities,
            right_query=Query(linked.sources["duns"].source, dag=linked.dag),
        )


@pytest.mark.parametrize(
    ("test_config", "expected_checks"),
    [
        pytest.param(
            {
                "right_args": False,
                "prob_range": (0.8, 0.9),
            },
            {
                "type": "deduper",
                "has_right": False,
                "prob_min": 0.8,
                "prob_max": 0.9,
            },
            id="deduper_configuration",
        ),
        pytest.param(
            {
                "right_args": True,
                "prob_range": (0.7, 0.95),
            },
            {
                "type": "linker",
                "has_right": True,
                "prob_min": 0.7,
                "prob_max": 0.95,
            },
            id="linker_configuration",
        ),
    ],
)
def test_query_to_model_factory_creation(
    test_config: dict[str, Any], expected_checks: dict[str, Any]
):
    """Test basic model creation from queries."""
    # Create linked sources with factory
    linked = linked_sources_factory()
    true_entities = tuple(linked.true_entities)

    # Get left source
    left_testkit = linked.sources["crn"]
    left_data = left_testkit.data
    left_keys = {"crn": "key"}

    # Setup right query if needed
    right_testkit = None
    right_data = None
    right_keys = None
    right_query = None

    if test_config["right_args"]:
        right_testkit = linked.sources["cdms"]
        right_data = right_testkit.data
        right_keys = {"cdms": "key"}
        right_query = Query(right_testkit.source, dag=linked.dag)

    # Create the model using our function
    model = query_to_model_factory(
        left_query=Query(left_testkit.source, dag=linked.dag),
        left_data=left_data,
        left_keys=left_keys,
        true_entities=true_entities,
        right_query=right_query,
        right_data=right_data,
        right_keys=right_keys,
        prob_range=test_config["prob_range"],
        seed=42,
    )

    # Basic type checks
    assert str(model.model.config.type) == expected_checks["type"]
    assert (model.right_query is not None) == expected_checks["has_right"]
    assert (model.right_clusters is not None) == expected_checks["has_right"]

    # Verify probabilities
    assert model.probabilities.schema.equals(SCHEMA_RESULTS)
    if len(model.probabilities) > 0:
        probs = model.probabilities["probability"].to_numpy() / 100
        assert all(p >= expected_checks["prob_min"] for p in probs)
        assert all(p <= expected_checks["prob_max"] for p in probs)


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal"),
    [
        pytest.param(42, 42, True, id="same_seeds"),
        pytest.param(1, 2, False, id="different_seeds"),
    ],
)
def test_query_to_model_factory_seed_behavior(
    seed1: int, seed2: int, should_be_equal: bool
):
    """Test that query_to_model_factory handles seeds correctly for reproducibility."""
    # Create linked sources with factory
    linked = linked_sources_factory()
    true_entities = tuple(linked.true_entities)

    # Get source
    left_testkit = linked.sources["crn"]
    left_data = left_testkit.data
    left_keys = {"crn": "key"}

    # Create two models with different seeds
    model1 = query_to_model_factory(
        left_query=Query(left_testkit.source, dag=linked.dag),
        left_data=left_data,
        left_keys=left_keys,
        true_entities=true_entities,
        seed=seed1,
    )

    model2 = query_to_model_factory(
        left_query=Query(left_testkit.source, dag=linked.dag),
        left_data=left_data,
        left_keys=left_keys,
        true_entities=true_entities,
        seed=seed2,
    )

    if should_be_equal:
        assert model1.model.name == model2.model.name
        assert model1.model.description == model2.model.description
        assert model1.probabilities.equals(model2.probabilities)
    else:
        assert model1.model.name != model2.model.name
        assert model1.model.description != model2.model.description
        if len(model1.probabilities) > 0 and len(model2.probabilities) > 0:
            assert not model1.probabilities.equals(model2.probabilities)


def test_query_to_model_factory_compare_with_model_factory():
    """Test that query_to_model_factory produces equivalent results to model_factory."""
    # Create linked sources with factory
    linked = linked_sources_factory(seed=42)
    true_entities = tuple(linked.true_entities)

    # Create model using model_factory
    standard_model = model_factory(
        left_testkit=linked.sources["crn"],
        right_testkit=linked.sources["cdms"],
        true_entities=true_entities,
        seed=42,
    )

    # Extract queries for our new function
    left_data = linked.sources["crn"].data
    right_data = linked.sources["cdms"].data
    left_keys = {"crn": "key"}
    right_keys = {"cdms": "key"}

    # Create model using query_to_model_factory
    query_model = query_to_model_factory(
        left_query=Query(linked.sources["crn"].source, dag=linked.dag),
        left_data=left_data,
        left_keys=left_keys,
        true_entities=true_entities,
        right_query=Query(linked.sources["cdms"].source, dag=linked.dag),
        right_data=right_data,
        right_keys=right_keys,
        seed=42,
    )

    # Verify that both models produce equivalent results
    # Names and descriptions will differ because they're randomly generated,
    # but probabilities should match with the same seed
    assert standard_model.probabilities.equals(query_model.probabilities)

    # Set the same threshold and verify queries
    standard_model.threshold = 80
    query_model.threshold = 80

    # Compare entities
    assert len(standard_model.entities) == len(query_model.entities)

    # Compare model type
    assert str(standard_model.model.config.type) == str(query_model.model.config.type)
