import pytest
from sqlalchemy import create_engine

from matchbox.common.factories.sources import (
    FeatureConfig,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


def test_linked_sources_factory_default():
    """Test that factory generates sources with default parameters."""
    linked = linked_sources_factory()

    # Check that default sources were created
    assert "crn" in linked.sources
    assert "duns" in linked.sources
    assert "cdms" in linked.sources

    # Verify default entity count
    assert len(linked.true_entities) == 10

    # Check that entities are properly tracked across sources
    for entity in linked.true_entities:
        # Each entity should have references to multiple sources
        source_references = list(entity.keys.items())
        assert len(source_references) > 0

        # Each reference should have source resolution name and keys
        for source_name, keys in source_references:
            assert source_name in linked.sources
            assert len(keys) > 0


def test_linked_sources_custom_config():
    """Test linked_sources_factory with custom source configurations."""
    engine = create_engine("sqlite:///:memory:")

    features = {
        "name": FeatureConfig(
            name="name",
            base_generator="name",
            variations=[SuffixRule(suffix=" Jr")],
        ),
        "user_id": FeatureConfig(
            name="user_id",
            base_generator="uuid4",
        ),
    }

    configs = (
        SourceTestkitParameters(
            name="source_a",
            engine=engine,
            features=(features["name"], features["user_id"]),
            n_true_entities=5,
            repetition=1,
        ),
        SourceTestkitParameters(
            name="source_b",
            features=(features["name"],),
            n_true_entities=3,
            repetition=2,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)

    # Verify sources were created correctly
    assert set(linked.sources.keys()) == {"source_a", "source_b"}
    assert len(linked.true_entities) == 5  # Max entities from configs

    # Check source A entities
    source_a_entities = [e for e in linked.true_entities if "source_a" in e.keys]
    assert len(source_a_entities) == 5

    # Check source B entities
    source_b_entities = [e for e in linked.true_entities if "source_b" in e.keys]
    assert len(source_b_entities) == 3


def test_linked_sources_find_entities():
    """Test the find_entities method with different criteria."""
    linked = linked_sources_factory(n_true_entities=10)

    # Find entities that appear at least once in each source
    min_appearances = {"crn": 1, "duns": 1, "cdms": 1}
    common_entities = linked.find_entities(min_appearances=min_appearances)

    # Should be subset of total entities
    assert len(common_entities) <= len(linked.true_entities)

    # Each entity should meet minimum appearance criteria
    for entity in common_entities:
        for source, min_count in min_appearances.items():
            assert len(entity.get_keys(source)) >= min_count

    # Find entities with maximum appearances
    max_appearances = {"duns": 1}
    limited_entities = linked.find_entities(max_appearances=max_appearances)

    for entity in limited_entities:
        for source, max_count in max_appearances.items():
            assert len(entity.get_keys(source)) <= max_count

    # Combined criteria
    filtered_entities = linked.find_entities(
        min_appearances={"crn": 1}, max_appearances={"duns": 2}
    )

    for entity in filtered_entities:
        assert len(entity.get_keys("crn")) >= 1
        assert len(entity.get_keys("duns")) <= 2


def test_entity_value_consistency():
    """Test that entity base values remain consistent across sources."""
    linked = linked_sources_factory(n_true_entities=5)

    for entity in linked.true_entities:
        base_values = entity.base_values

        # Get actual values from each source
        for source_name, keys in entity.keys.items():
            source = linked.sources[source_name]
            df = source.data.to_pandas()

            # Get rows for this entity
            entity_rows = df[df["key"].isin(keys)]

            # For each feature in the source
            for feature in source.features:
                if feature.name in base_values:
                    # The base value should appear in the data
                    # (unless it's marked as drop_base)
                    if not feature.drop_base:
                        assert (
                            base_values[feature.name]
                            in entity_rows[feature.name].values
                        )


def test_source_entity_equality():
    """Test SourceEntity equality and hashing behavior."""
    linked = linked_sources_factory(n_true_entities=3)

    # Get a few entities
    entities = list(linked.true_entities)

    # Same entity should be equal to itself
    assert entities[0] == entities[0]

    # Different entities should not be equal
    assert entities[0] != entities[1]

    # Entities with same base values should be equal
    entity_copy = entities[0].model_copy()
    assert entity_copy == entities[0]

    # Should work in sets (testing hash implementation)
    entity_set = {entities[0], entity_copy, entities[1]}
    assert len(entity_set) == 2  # Only unique entities


def test_seed_reproducibility():
    """Test that linked sources generation is reproducible with same seed."""
    source_parameters = SourceTestkitParameters(
        name="test_source",
        features=(
            FeatureConfig(
                name="name",
                base_generator="name",
                variations=[SuffixRule(suffix=" Jr")],
            ),
        ),
        n_true_entities=5,
    )

    # Generate two instances with same seed
    linked1 = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    linked2 = linked_sources_factory(source_parameters=(source_parameters,), seed=42)

    # Generate one with different seed
    linked3 = linked_sources_factory(source_parameters=(source_parameters,), seed=43)

    # Same seed should produce identical results
    assert linked1.sources["test_source"].data.equals(
        linked2.sources["test_source"].data
    )
    assert len(linked1.true_entities) == len(linked2.true_entities)

    # Different seeds should produce different results
    assert not linked1.sources["test_source"].data.equals(
        linked3.sources["test_source"].data
    )


def test_empty_source_handling():
    """Test handling of sources with zero entities."""
    source_parameters = SourceTestkitParameters(
        name="empty_source",
        features=(FeatureConfig(name="name", base_generator="name"),),
        n_true_entities=0,
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,))

    # Should create source but with empty data
    assert "empty_source" in linked.sources
    assert len(linked.sources["empty_source"].data) == 0
    assert len(linked.true_entities) == 0


def test_large_entity_count():
    """Test handling of sources with large number of entities."""
    source_parameters = SourceTestkitParameters(
        name="large_source",
        features=(FeatureConfig(name="user_id", base_generator="uuid4"),),
        n_true_entities=10_000,
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,))

    # Should handle large entity counts
    assert len(linked.true_entities) == 10_000
    assert len(linked.sources["large_source"].data) == 10_000


def test_feature_inheritance():
    """Test that entities inherit all features from their source configurations."""
    features = {
        "name": FeatureConfig(name="name", base_generator="name"),
        "email": FeatureConfig(name="email", base_generator="email"),
        "phone": FeatureConfig(name="phone", base_generator="phone_number"),
    }

    configs = (
        SourceTestkitParameters(
            name="source_a", features=(features["name"], features["email"])
        ),
        SourceTestkitParameters(
            name="source_b", features=(features["name"], features["phone"])
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, n_true_entities=10)

    # Check that entities have all relevant features
    for entity in linked.true_entities:
        # All entities should have name (common feature)
        assert "name" in entity.base_values

        # Entities in source_a should have email
        if "source_a" in entity.keys:
            assert "email" in entity.base_values

        # Entities in source_b should have phone
        if "source_b" in entity.keys:
            assert "phone" in entity.base_values


def test_unique_feature_values():
    """Test that unique features generate distinct values across entities."""
    source_parameters = SourceTestkitParameters(
        name="test_source",
        features=(
            FeatureConfig(name="unique_id", base_generator="uuid4", unique=True),
            FeatureConfig(name="is_true", base_generator="boolean", unique=False),
        ),
        n_true_entities=100,
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,))

    # Get all base values
    unique_ids = set()
    categories = set()
    for entity in linked.true_entities:
        unique_ids.add(entity.base_values["unique_id"])
        categories.add(entity.base_values["is_true"])

    # Unique feature should have same number of values as entities
    assert len(unique_ids) == 100

    # Non-unique feature should have fewer unique values
    assert len(categories) < 100


def test_source_references():
    """Test adding and retrieving source references."""
    linked = linked_sources_factory(n_true_entities=2)
    entity = next(iter(linked.true_entities))

    # Add new source reference
    new_keys = {"keys1", "keys2"}
    entity.add_source_reference("new_source", new_keys)

    # Should be able to retrieve the keys
    assert entity.get_keys("new_source") == new_keys

    # Update existing reference
    updated_keys = {"keys3"}
    entity.add_source_reference("new_source", updated_keys)
    assert entity.get_keys("new_source") == updated_keys

    # Non-existent source should return empty list
    assert entity.get_keys("nonexistent") == set()


def test_linked_sources_entity_hierarchy():
    """Test that LinkedSourcesTestkit correctly maintains entity hierarchy."""
    # Create linked sources with multiple sources
    features = {
        "name": FeatureConfig(
            name="name",
            base_generator="name",
            variations=[SuffixRule(suffix=" Jr")],
        ),
        "user_id": FeatureConfig(
            name="user_id",
            base_generator="uuid4",
        ),
    }

    configs = (
        SourceTestkitParameters(
            name="source_a",
            features=(features["name"], features["user_id"]),
            n_true_entities=5,
        ),
        SourceTestkitParameters(
            name="source_b",
            features=(features["name"],),
            n_true_entities=3,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)

    # For each source, verify its entities are subsets of true_entities
    for source_name, source in linked.sources.items():
        for cluster_entity in source.entities:
            # Find all true entities that could be parents of this cluster entity
            matching_parents = [
                true_entity
                for true_entity in linked.true_entities
                if cluster_entity.is_subset_of_source_entity(true_entity)
            ]

            # Each cluster entity must have at least one parent
            assert len(matching_parents) > 0, (
                f"ClusterEntity in {source_name} has no parent in true_entities"
            )

            # The source keys from the cluster entity should be a subset of
            # at least one true entity's keys
            assert any(
                cluster_entity.keys <= true_entity.keys
                for true_entity in matching_parents
            ), f"ClusterEntity in {source_name} not a proper subset of any true entity"


def test_linked_sources_entity_count_behavior():
    """Test different n_true_entities behaviors in linked_sources_factory."""
    base_feature = FeatureConfig(name="name", base_generator="name")
    engine = create_engine("sqlite:///:memory:")

    # Test error when n_true_entities missing from configs
    configs_missing_counts = (
        SourceTestkitParameters(
            name="source_a",
            engine=engine,
            features=(base_feature,),
            n_true_entities=5,
        ),
        SourceTestkitParameters(
            name="source_b",
            features=(base_feature,),  # Deliberately missing n_true_entities
        ),
    )

    with pytest.raises(
        ValueError, match="n_true_entities not set for sources: source_b"
    ):
        linked_sources_factory(source_parameters=configs_missing_counts)

    # Test respecting different entity counts per source
    configs_different_counts = (
        SourceTestkitParameters(
            name="source_a",
            engine=engine,
            features=(base_feature,),
            n_true_entities=5,
        ),
        SourceTestkitParameters(
            name="source_b",
            features=(base_feature,),
            n_true_entities=10,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs_different_counts)

    # Should generate enough entities for max requested (10)
    assert len(linked.true_entities) == 10

    # Each source should have its specified number of entities
    source_a_entities = [e for e in linked.true_entities if "source_a" in e.keys]
    source_b_entities = [e for e in linked.true_entities if "source_b" in e.keys]
    assert len(source_a_entities) == 5, "SourceConfig A should have 5 entities"
    assert len(source_b_entities) == 10, "SourceConfig B should have 10 entities"

    # Test factory parameter override
    with pytest.warns(UserWarning, match="factory parameter will be used"):
        linked_override = linked_sources_factory(
            source_parameters=configs_different_counts, n_true_entities=15
        )

    # Both sources should now have 15 entities
    override_source_a = [
        e for e in linked_override.true_entities if "source_a" in e.keys
    ]
    override_source_b = [
        e for e in linked_override.true_entities if "source_b" in e.keys
    ]
    assert len(override_source_a) == 15, (
        "SourceConfig A should be overridden to 15 entities"
    )
    assert len(override_source_b) == 15, (
        "SourceConfig B should be overridden to 15 entities"
    )
