from typing import Any

import numpy as np
import pyarrow.compute as pc
import pytest
from sqlalchemy import create_engine

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS
from matchbox.common.dtos import ModelType
from matchbox.common.factories.models import (
    calculate_min_max_edges,
    generate_dummy_probabilities,
    model_factory,
    verify_components,
)
from matchbox.common.factories.sources import (
    DropBaseRule,
    FeatureConfig,
    SourceConfig,
    SuffixRule,
    linked_sources_factory,
    source_factory,
)
from matchbox.common.sources import SourceAddress


class TestSourceFactory:
    def test_source_factory_default(self):
        """Test that source_factory generates a dummy source with default parameters."""
        source = source_factory()

        assert len(source.entities) == 10
        assert source.data.shape[0] == 10
        assert source.data_hashes.shape[0] == 10
        assert source.data_hashes.schema.equals(SCHEMA_INDEX)

    def test_source_factory_repetition(self):
        """Test that source_factory correctly handles row repetition."""
        features = [
            FeatureConfig(
                name="company_name",
                base_generator="company",
                variations=[SuffixRule(suffix=" Inc")],
            ),
        ]

        n_true_entities = 2
        repetition = 3
        source = source_factory(
            n_true_entities=n_true_entities,
            repetition=repetition,
            features=features,
            seed=42,
        )

        # Convert to pandas for easier analysis
        data_df = source.data.to_pandas()
        hashes_df = source.data_hashes.to_pandas()

        # For each hash group, verify it contains the correct number of rows
        for _, group in hashes_df.groupby("hash"):
            # Each hash should have repetition number of PKs
            source_pks = group["source_pk"].explode()
            assert len(source_pks) == repetition

            # Get the actual rows for these PKs
            rows = data_df[data_df["pk"].isin(source_pks)]

            # Should have repetition number of rows
            assert len(rows) == repetition

            # All rows should have identical feature values
            for feature in features:
                assert len(rows[feature.name].unique()) == 1

        # Total number of unique hashes should be n_true_entities * (variations + 1)
        expected_unique_hashes = n_true_entities * (len(features[0].variations) + 1)
        assert len(hashes_df["hash"].unique()) == expected_unique_hashes

        # Total number of rows should be unique hashes * repetition
        assert len(data_df) == expected_unique_hashes * repetition

    def test_source_factory_data_hashes_integrity(self):
        """Test that data_hashes correctly identifies identical rows."""
        features = [
            FeatureConfig(
                name="company_name",
                base_generator="company",
                variations=[SuffixRule(suffix=" Inc")],
            ),
        ]

        n_true_entities = 3
        repetition = 1
        dummy_source = source_factory(
            n_true_entities=n_true_entities,
            repetition=repetition,
            features=features,
            seed=42,
        )

        # Convert to pandas for easier analysis
        hashes_df = dummy_source.data_hashes.to_pandas()
        data_df = dummy_source.data.to_pandas()

        # For each hash group, verify that the corresponding rows are identical
        for _, group in hashes_df.groupby("hash"):
            pks = group["source_pk"].explode()
            rows = data_df[data_df["pk"].isin(pks)]

            # All rows in the same hash group should have identical feature values
            for feature in features:
                assert len(rows[feature.name].unique()) == 1

        # Due to repetition=1, each unique row should appear
        # in exactly one hash group with two PKs
        assert all(
            len(group["source_pk"].explode()) == repetition
            for _, group in hashes_df.groupby("hash")
        )

        # Total number of hash groups should equal
        # number of unique rows * number of true entities
        expected_hash_groups = n_true_entities * (
            len(features[0].variations) + 1
        )  # +1 for base value
        assert len(hashes_df["hash"].unique()) == expected_hash_groups

    def test_source_dummy_to_mock(self):
        """Test that SourceDummy.to_mock() creates a correctly configured mock."""
        # Create a source dummy with some test data
        features = [
            FeatureConfig(
                name="test_field",
                base_generator="word",
                variations=[SuffixRule(suffix="_variant")],
            )
        ]

        dummy_source = source_factory(
            features=features, full_name="test.source", n_true_entities=2, seed=42
        )

        # Create the mock
        mock_source = dummy_source.to_mock()

        # Test that method calls are tracked
        mock_source.set_engine("test_engine")
        mock_source.default_columns()
        mock_source.hash_data()

        mock_source.set_engine.assert_called_once_with("test_engine")
        mock_source.default_columns.assert_called_once()
        mock_source.hash_data.assert_called_once()

        # Test method return values
        assert mock_source.set_engine("test_engine") == mock_source
        assert mock_source.default_columns() == mock_source
        assert mock_source.hash_data() == dummy_source.data_hashes

        # Test model dump methods
        original_dump = dummy_source.source.model_dump()
        mock_dump = mock_source.model_dump()
        assert mock_dump == original_dump

        original_json = dummy_source.source.model_dump_json()
        mock_json = mock_source.model_dump_json()
        assert mock_json == original_json

        # Verify side effect functions were set correctly
        mock_source.model_dump.assert_called_once()
        mock_source.model_dump_json.assert_called_once()

        # Test that to_table contains the correct data
        assert mock_source.to_table == dummy_source.data
        # Verify the number of rows matches what we created
        assert mock_source.to_table.shape[0] == dummy_source.data.shape[0]

    def test_source_factory_mock_properties(self):
        """Test that source properties set in source_factory match generated Source."""
        # Create source with specific features and name
        features = [
            FeatureConfig(
                name="company_name",
                base_generator="company",
                variations=(SuffixRule(suffix=" Ltd"),),
            ),
            FeatureConfig(
                name="registration_id",
                base_generator="numerify",
                parameters=(("text", "######"),),
            ),
        ]

        full_name = "companies"
        engine = create_engine("sqlite:///:memory:")

        dummy_source = source_factory(
            features=features, full_name=full_name, engine=engine
        ).source

        # Check source address properties
        assert dummy_source.address.full_name == full_name

        # Warehouse hash should be consistent for same engine config
        expected_address = SourceAddress.compose(engine=engine, full_name=full_name)
        assert dummy_source.address.warehouse_hash == expected_address.warehouse_hash

        # Check column configuration
        assert len(dummy_source.columns) == len(features)
        for feature, column in zip(features, dummy_source.columns, strict=False):
            assert column.name == feature.name
            assert column.alias == feature.name
            assert column.type is None

        # Check default alias (should match full_name) and default pk
        assert dummy_source.alias == full_name
        assert dummy_source.db_pk == "pk"

        # Verify source properties are preserved through model_dump
        dump = dummy_source.model_dump()
        assert dump["address"]["full_name"] == full_name
        assert dump["columns"] == [
            {"name": f.name, "alias": f.name, "type": None} for f in features
        ]

    def test_entity_variations_tracking(self):
        """Test that entity variations are correctly tracked and accessible."""
        features = [
            FeatureConfig(
                name="company",
                base_generator="company",
                variations=[
                    SuffixRule(suffix=" Inc"),
                    SuffixRule(suffix=" Ltd"),
                    DropBaseRule(),
                ],
            )
        ]

        source = source_factory(features=features, n_true_entities=2, seed=42)

        # Each entity should track its variations
        for entity in source.entities:
            # After DropBaseRule, we should only have the non-drop variations
            expected_variations = len(
                [v for v in features[0].variations if not isinstance(v, DropBaseRule)]
            )
            assert entity.total_unique_variations == expected_variations

            # Get all variations
            variations = entity.variations({source.source.address.full_name: source})

            # Should have variations for our source
            assert len(variations) == 1
            source_variations = next(iter(variations.values()))

            # Should have variations for our feature
            assert "company" in source_variations
            company_variations = source_variations["company"]

            # Should have the base value in the variation tracking (even though
            # it's dropped) plus all actual variations
            assert len(company_variations) == expected_variations + 1

            # Verify base value is marked as dropped and variations use the rules
            base_value = entity.base_values["company"]
            assert "{'drop': True}" in company_variations[base_value]
            assert any("Inc" in desc for desc in company_variations.values())
            assert any("Ltd" in desc for desc in company_variations.values())


class TestLinkedSourcesFactory:
    def test_linked_sources_factory_default(self):
        """Test that factory generates sources with default parameters."""
        linked = linked_sources_factory()

        # Check that default sources were created
        assert "crn" in linked.sources
        assert "duns" in linked.sources
        assert "cdms" in linked.sources

        # Verify default entity count
        assert len(linked.entities) == 10

        # Check that entities are properly tracked across sources
        for entity in linked.entities.values():
            # Each entity should have source references
            assert len(entity.source_pks) > 0

            # Each reference should contain PKs
            for ref in entity.source_pks:
                assert len(ref.source_pks) > 0

    def test_linked_sources_custom_config(self):
        """Test linked_sources_factory with custom source configurations."""
        engine = create_engine("sqlite:///:memory:")

        features = {
            "name": FeatureConfig(
                name="name",
                base_generator="name",
                variations=[SuffixRule(suffix=" Jr")],
            ),
            "id": FeatureConfig(
                name="id",
                base_generator="uuid4",
            ),
        }

        configs = (
            SourceConfig(
                full_name="source_a",
                engine=engine,
                features=(features["name"], features["id"]),
                n_entities=5,
                repetition=1,
            ),
            SourceConfig(
                full_name="source_b",
                features=(features["name"],),
                n_entities=3,
                repetition=2,
            ),
        )

        linked = linked_sources_factory(source_configs=configs, seed=42)

        # Verify sources were created correctly
        assert set(linked.sources.keys()) == {"source_a", "source_b"}
        assert len(linked.entities) == 5  # Max entities from configs

        # Check source A entities
        source_a_entities = [
            e
            for e in linked.entities.values()
            if any(ref.name == "source_a" for ref in e.source_pks)
        ]
        assert len(source_a_entities) == 5

        # Check source B entities
        source_b_entities = [
            e
            for e in linked.entities.values()
            if any(ref.name == "source_b" for ref in e.source_pks)
        ]
        assert len(source_b_entities) == 3

    def test_linked_sources_find_entities(self):
        """Test the find_entities method with different criteria."""
        linked = linked_sources_factory(n_entities=10)

        # Find entities that appear at least once in each source
        min_appearances = {"crn": 1, "duns": 1, "cdms": 1}
        common_entities = linked.find_entities(min_appearances=min_appearances)

        # Should be subset of total entities
        assert len(common_entities) <= len(linked.entities)

        # Each entity should meet minimum appearance criteria
        for entity in common_entities:
            for source, min_count in min_appearances.items():
                assert len(entity.get_source_pks(source)) >= min_count

        # Find entities with maximum appearances
        max_appearances = {"duns": 1}
        limited_entities = linked.find_entities(max_appearances=max_appearances)

        for entity in limited_entities:
            for source, max_count in max_appearances.items():
                assert len(entity.get_source_pks(source)) <= max_count

        # Combined criteria
        filtered_entities = linked.find_entities(
            min_appearances={"crn": 1}, max_appearances={"duns": 2}
        )

        for entity in filtered_entities:
            assert len(entity.get_source_pks("crn")) >= 1
            assert len(entity.get_source_pks("duns")) <= 2

    def test_entity_value_consistency(self):
        """Test that entity base values remain consistent across sources."""
        linked = linked_sources_factory(n_entities=5)

        for entity in linked.entities.values():
            base_values = entity.base_values

            # Get actual values from each source
            for source_ref in entity.source_pks:
                source = linked.sources[source_ref.name]
                df = source.data.to_pandas()

                # Get rows for this entity
                entity_rows = df[df["pk"].isin(source_ref.source_pks)]

                # For each feature in the source
                for feature in source.features:
                    if feature.name in base_values:
                        # The base value should appear in the data
                        # (unless it was dropped by a DropBaseRule)
                        has_drop_rule = any(
                            isinstance(rule, DropBaseRule)
                            for rule in feature.variations
                        )
                        if not has_drop_rule:
                            assert (
                                base_values[feature.name]
                                in entity_rows[feature.name].values
                            )

    def test_source_entity_equality(self):
        """Test SourceEntity equality and hashing behavior."""
        linked = linked_sources_factory(n_entities=3)

        # Get a few entities
        entities = list(linked.entities.values())

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

    def test_seed_reproducibility(self):
        """Test that linked sources generation is reproducible with same seed."""
        config = SourceConfig(
            full_name="test_source",
            features=(
                FeatureConfig(
                    name="name",
                    base_generator="name",
                    variations=[SuffixRule(suffix=" Jr")],
                ),
            ),
            n_entities=5,
        )

        # Generate two instances with same seed
        linked1 = linked_sources_factory(source_configs=(config,), seed=42)
        linked2 = linked_sources_factory(source_configs=(config,), seed=42)

        # Generate one with different seed
        linked3 = linked_sources_factory(source_configs=(config,), seed=43)

        # Same seed should produce identical results
        assert linked1.sources["test_source"].data.equals(
            linked2.sources["test_source"].data
        )
        assert len(linked1.entities) == len(linked2.entities)

        # Different seeds should produce different results
        assert not linked1.sources["test_source"].data.equals(
            linked3.sources["test_source"].data
        )

    def test_empty_source_handling(self):
        """Test handling of sources with zero entities."""
        config = SourceConfig(
            full_name="empty_source",
            features=(FeatureConfig(name="name", base_generator="name"),),
            n_entities=0,
        )

        linked = linked_sources_factory(source_configs=(config,))

        # Should create source but with empty data
        assert "empty_source" in linked.sources
        assert len(linked.sources["empty_source"].data) == 0
        assert len(linked.entities) == 0

    def test_large_entity_count(self):
        """Test handling of sources with large number of entities."""
        config = SourceConfig(
            full_name="large_source",
            features=(FeatureConfig(name="id", base_generator="uuid4"),),
            n_entities=10_000,
        )

        linked = linked_sources_factory(source_configs=(config,))

        # Should handle large entity counts
        assert len(linked.entities) == 10_000
        assert len(linked.sources["large_source"].data) == 10_000

    def test_feature_inheritance(self):
        """Test that entities inherit all features from their source configurations."""
        features = {
            "name": FeatureConfig(name="name", base_generator="name"),
            "email": FeatureConfig(name="email", base_generator="email"),
            "phone": FeatureConfig(name="phone", base_generator="phone_number"),
        }

        configs = (
            SourceConfig(
                full_name="source_a", features=(features["name"], features["email"])
            ),
            SourceConfig(
                full_name="source_b", features=(features["name"], features["phone"])
            ),
        )

        linked = linked_sources_factory(source_configs=configs)

        # Check that entities have all relevant features
        for entity in linked.entities.values():
            # All entities should have name (common feature)
            assert "name" in entity.base_values

            # Entities in source_a should have email
            if any(ref.name == "source_a" for ref in entity.source_pks):
                assert "email" in entity.base_values

            # Entities in source_b should have phone
            if any(ref.name == "source_b" for ref in entity.source_pks):
                assert "phone" in entity.base_values

    def test_unique_feature_values(self):
        """Test that unique features generate distinct values across entities."""
        config = SourceConfig(
            full_name="test_source",
            features=(
                FeatureConfig(name="unique_id", base_generator="uuid4", unique=True),
                FeatureConfig(name="is_true", base_generator="boolean", unique=False),
            ),
            n_entities=100,
        )

        linked = linked_sources_factory(source_configs=(config,))

        # Get all base values
        unique_ids = set()
        categories = set()
        for entity in linked.entities.values():
            unique_ids.add(entity.base_values["unique_id"])
            categories.add(entity.base_values["is_true"])

        # Unique feature should have same number of values as entities
        assert len(unique_ids) == 100

        # Non-unique feature should have fewer unique values
        assert len(categories) < 100

    def test_source_references(self):
        """Test adding and retrieving source references."""
        linked = linked_sources_factory(n_entities=2)
        entity = next(iter(linked.entities.values()))

        # Add new source reference
        new_pks = ["pk1", "pk2"]
        entity.add_source_reference("new_source", new_pks)

        # Should be able to retrieve the PKs
        assert entity.get_source_pks("new_source") == new_pks

        # Update existing reference
        updated_pks = ["pk3"]
        entity.add_source_reference("new_source", updated_pks)
        assert entity.get_source_pks("new_source") == updated_pks

        # Non-existent source should return empty list
        assert entity.get_source_pks("nonexistent") == []


class TestModelFactory:
    def test_model_factory_default(self):
        """Test that model_factory generates a dummy model with default parameters."""
        dummy = model_factory()

        assert dummy.metrics.n_true_entities == 10
        assert dummy.model.metadata.type == ModelType.DEDUPER
        assert dummy.model.metadata.right_resolution is None

        # Check that probabilities table was generated correctly
        assert len(dummy.data) > 0
        assert dummy.data.schema.equals(SCHEMA_RESULTS)

    def test_model_factory_with_custom_params(self):
        """Test model_factory with custom parameters."""
        name = "test_model"
        description = "test description"
        n_true_entities = 5
        prob_range = (0.9, 1.0)

        dummy = model_factory(
            name=name,
            description=description,
            n_true_entities=n_true_entities,
            prob_range=prob_range,
        )

        assert dummy.model.metadata.name == name
        assert dummy.model.metadata.description == description
        assert dummy.metrics.n_true_entities == n_true_entities

        # Check probability range
        probs = dummy.data.column("probability").to_pylist()
        assert all(90 <= p <= 100 for p in probs)

    @pytest.mark.parametrize(
        ("model_type"),
        [
            pytest.param("deduper", id="deduper"),
            pytest.param("linker", id="linker"),
        ],
    )
    def test_model_factory_different_types(self, model_type: str):
        """Test model_factory handles different model types correctly."""
        dummy = model_factory(model_type=model_type)

        assert dummy.model.metadata.type == model_type

        if model_type == ModelType.LINKER:
            assert dummy.model.metadata.right_resolution is not None

            # Check that left and right values are in different ranges
            left_vals = dummy.data.column("left_id").to_pylist()
            right_vals = dummy.data.column("right_id").to_pylist()
            left_min, left_max = min(left_vals), max(left_vals)
            right_min, right_max = min(right_vals), max(right_vals)
            assert (left_min < left_max < right_min < right_max) or (
                right_min < right_max < left_min < left_max
            )

    @pytest.mark.parametrize(
        ("seed1", "seed2", "should_be_equal"),
        [
            pytest.param(42, 42, True, id="same_seeds"),
            pytest.param(1, 2, False, id="different_seeds"),
        ],
    )
    def test_model_factory_seed_behavior(
        self, seed1: int, seed2: int, should_be_equal: bool
    ):
        """Test that model_factory handles seeds correctly for reproducibility."""
        dummy1 = model_factory(seed=seed1)
        dummy2 = model_factory(seed=seed2)

        if should_be_equal:
            assert dummy1.model.metadata.name == dummy2.model.metadata.name
            assert (
                dummy1.model.metadata.description == dummy2.model.metadata.description
            )
            assert dummy1.data.equals(dummy2.data)
        else:
            assert dummy1.model.metadata.name != dummy2.model.metadata.name
            assert (
                dummy1.model.metadata.description != dummy2.model.metadata.description
            )
            assert not dummy1.data.equals(dummy2.data)

    @pytest.mark.parametrize(
        ("left_n", "right_n", "n_components", "true_min", "true_max"),
        [
            (10, None, 2, 8, 20),
            (11, None, 2, 9, 25),
            (9, 9, 3, 15, 27),
            (8, 4, 3, 9, 11),
            (4, 8, 3, 9, 11),
            (8, 8, 3, 13, 22),
        ],
        ids=[
            "dedupe_no_mod",
            "dedup_mod",
            "link_no_mod",
            "link_left_mod",
            "link_right_mod",
            "link_same_mod",
        ],
    )
    def test_calculate_min_max_edges(
        self,
        left_n: int,
        right_n: int | None,
        n_components: int,
        true_min: int,
        true_max: int,
    ):
        deduplicate = False
        if not right_n:
            deduplicate = True
            right_n = left_n
        min_edges, max_edges = calculate_min_max_edges(
            left_n, right_n, n_components, deduplicate
        )

        assert true_min == min_edges
        assert true_max == max_edges

    @pytest.mark.parametrize(
        ("parameters"),
        [
            {
                "left_count": 5,
                "right_count": None,
                "prob_range": (0.6, 0.8),
                "num_components": 3,
                "total_rows": 2,
            },
            {
                "left_count": 1000,
                "right_count": None,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1000, 1000, 10, True)[0],
            },
            {
                "left_count": 1_000,
                "right_count": None,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1000, 1000, 10, True)[1],
            },
            {
                "left_count": 1_000,
                "right_count": 1_000,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1000, 1000, 10, False)[0],
            },
            {
                "left_count": 1_000,
                "right_count": 1_000,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1000, 1000, 10, False)[1],
            },
        ],
        ids=[
            "dedupe_no_edges",
            "dedupe_min",
            "dedupe_max",
            "link_min",
            "link_max",
        ],
    )
    def test_generate_dummy_probabilities(self, parameters: dict[str, Any]):
        len_left = parameters["left_count"]
        len_right = parameters["right_count"]
        if len_right:
            total_len = len_left + len_right
            len_right = parameters["right_count"]
            rand_vals = np.random.choice(a=total_len, replace=False, size=total_len)
            left_values = list(rand_vals[:len_left])
            right_values = list(rand_vals[len_left:])
        else:
            rand_vals = np.random.choice(a=len_left, replace=False, size=len_left)
            left_values = list(rand_vals[:len_left])
            right_values = None

        n_components = parameters["num_components"]
        total_rows = parameters["total_rows"]

        probabilities = generate_dummy_probabilities(
            left_values=left_values,
            right_values=right_values,
            prob_range=parameters["prob_range"],
            num_components=n_components,
            total_rows=total_rows,
        )
        report = verify_components(table=probabilities, all_nodes=rand_vals)
        p_left = probabilities["left_id"].to_pylist()
        p_right = probabilities["right_id"].to_pylist()

        assert report["num_components"] == n_components

        # Link job
        if right_values:
            assert set(p_left) <= set(left_values)
            assert set(p_right) <= set(right_values)
        # Dedupe
        else:
            assert set(p_left) | set(p_right) <= set(left_values)

        assert (
            pc.max(probabilities["probability"]).as_py() / 100
            <= parameters["prob_range"][1]
        )
        assert (
            pc.min(probabilities["probability"]).as_py() / 100
            >= parameters["prob_range"][0]
        )

        assert len(probabilities) == total_rows

        edges = zip(p_left, p_right, strict=True)
        edges_set = {tuple(sorted(e)) for e in edges}
        assert len(edges_set) == total_rows

        self_references = [e for e in edges if e[0] == e[1]]
        assert len(self_references) == 0

    @pytest.mark.parametrize(
        ("parameters"),
        [
            {
                "left_range": (0, 10_000),
                "right_range": (10_000, 20_000),
                "num_components": 2,
                "total_rows": 1,
            },
            {
                "left_range": (0, 10),
                "right_range": (10, 20),
                "num_components": 2,
                "total_rows": 8_000,
            },
        ],
        ids=["lower_than_min", "higher_than_max"],
    )
    def test_generate_dummy_probabilities_errors(self, parameters: dict[str, Any]):
        left_values = range(*parameters["left_range"])
        right_values = range(*parameters["right_range"])

        with pytest.raises(ValueError):
            generate_dummy_probabilities(
                left_values=left_values,
                right_values=right_values,
                prob_range=(0.6, 0.8),
                num_components=parameters["num_components"],
                total_rows=parameters["total_rows"],
            )
