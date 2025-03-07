import functools

import pytest
from faker import Faker
from sqlalchemy import create_engine

from matchbox.common.arrow import SCHEMA_INDEX
from matchbox.common.factories.entities import (
    FeatureConfig,
    ReplaceRule,
    SourceEntity,
    SuffixRule,
)
from matchbox.common.factories.sources import (
    generate_rows,
    source_factory,
)
from matchbox.common.sources import SourceAddress


def test_source_factory_default():
    """Test that source_factory generates a source testkit with default parameters."""
    source = source_factory()

    assert len(source.entities) == 10
    assert source.data.shape[0] == 10
    assert source.data_hashes.shape[0] == 10
    assert source.data_hashes.schema.equals(SCHEMA_INDEX)


def test_source_factory_repetition():
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
        # Each hash should have repetition + 1 (base) number of PKs
        source_pks = group["source_pk"].explode()
        assert len(source_pks) == repetition + 1

        # Get the actual rows for these PKs
        rows = data_df[data_df["pk"].isin(source_pks)]

        # Should have repetition + 1 (base) number of rows
        assert len(rows) == repetition + 1

        # All rows should have identical feature values
        for feature in features:
            assert len(rows[feature.name].unique()) == 1

    # Total number of unique hashes should be n_true_entities * (variations + 1)
    expected_unique_hashes = n_true_entities * (len(features[0].variations) + 1)
    assert len(hashes_df["hash"].unique()) == expected_unique_hashes

    # Total number of rows should be unique hashes * repetition + 1 (base)
    assert len(data_df) == expected_unique_hashes * (repetition + 1)


def test_source_factory_data_hashes_integrity():
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
    source_testkit = source_factory(
        n_true_entities=n_true_entities,
        repetition=repetition,
        features=features,
        seed=42,
    )

    # Convert to pandas for easier analysis
    hashes_df = source_testkit.data_hashes.to_pandas()
    data_df = source_testkit.data.to_pandas()

    # For each hash group, verify that the corresponding rows are identical
    for _, group in hashes_df.groupby("hash"):
        pks = group["source_pk"].explode()
        rows = data_df[data_df["pk"].isin(pks)]

        # All rows in the same hash group should have identical feature values
        for feature in features:
            assert len(rows[feature.name].unique()) == 1

    # Due to repetition=1, each unique row should appear
    # in exactly one hash group with two PKs
    # Repetition + 1 because we include the base value
    assert all(
        len(group["source_pk"].explode()) == repetition + 1
        for _, group in hashes_df.groupby("hash")
    )

    # Total number of hash groups should equal
    # number of unique rows * number of true entities
    expected_hash_groups = n_true_entities * (
        len(features[0].variations) + 1
    )  # +1 for base value
    assert len(hashes_df["hash"].unique()) == expected_hash_groups


def test_source_testkit_to_mock():
    """Test that SourceTestkit.mock creates a correctly configured mock."""
    # Create a source testkit with some test data
    features = [
        FeatureConfig(
            name="test_field",
            base_generator="word",
            variations=[SuffixRule(suffix="_variant")],
        )
    ]

    source_testkit = source_factory(
        features=features, full_name="test.source", n_true_entities=2, seed=42
    )

    # Create the mock
    mock_source = source_testkit.mock

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
    assert mock_source.hash_data() == source_testkit.data_hashes

    # Test model dump methods
    original_dump = source_testkit.source.model_dump()
    mock_dump = mock_source.model_dump()
    assert mock_dump == original_dump

    original_json = source_testkit.source.model_dump_json()
    mock_json = mock_source.model_dump_json()
    assert mock_json == original_json

    # Verify side effect functions were set correctly
    mock_source.model_dump.assert_called_once()
    mock_source.model_dump_json.assert_called_once()

    # Test that to_table contains the correct data
    assert mock_source.to_table == source_testkit.data
    # Verify the number of rows matches what we created
    assert mock_source.to_table.shape[0] == source_testkit.data.shape[0]


def test_source_factory_mock_properties():
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

    source_testkit = source_factory(
        features=features, full_name=full_name, engine=engine
    ).source

    # Check source address properties
    assert source_testkit.address.full_name == full_name

    # Warehouse hash should be consistent for same engine config
    expected_address = SourceAddress.compose(engine=engine, full_name=full_name)
    assert source_testkit.address.warehouse_hash == expected_address.warehouse_hash

    # Check column configuration
    assert len(source_testkit.columns) == len(features)
    for feature, column in zip(features, source_testkit.columns, strict=False):
        assert column.name == feature.name
        assert column.alias == feature.name
        assert column.type == feature.sql_type

    # Check default alias (should match full_name) and default pk
    assert source_testkit.alias == full_name
    assert source_testkit.db_pk == "pk"

    # Verify source properties are preserved through model_dump
    dump = source_testkit.model_dump()
    assert dump["address"]["full_name"] == full_name
    assert dump["columns"] == [
        {"name": f.name, "alias": f.name, "type": f.sql_type} for f in features
    ]


def test_entity_variations_tracking():
    """Test that entity variations are correctly tracked and accessible.

    Asserts that ClusterEntity objects are proper subsets of their parent entities.
    """
    features = [
        FeatureConfig(
            name="company",
            base_generator="company",
            variations=[
                SuffixRule(suffix=" Inc"),
                SuffixRule(suffix=" Ltd"),
            ],
            drop_base=True,
        )
    ]

    source = source_factory(features=features, n_true_entities=2, seed=42)
    source_name = source.source.address.full_name

    # Process each ClusterEntity group
    for cluster_entity in source.entities:
        # Get the values for this entity
        entity_values = cluster_entity.get_values({source_name: source})

        # Calculate total unique variations (equivalent to total_unique_variations)
        unique_variations = 0
        for features_values in entity_values.values():
            for values in features_values.values():
                unique_variations += len(values)

        # With drop_base=True, we should only have the non-drop variations
        # Each entity should have exactly one variation
        assert unique_variations == 1

        # Verify the data values match expectations
        data_df = source.data.to_pandas()

        # Get PKs for this cluster entity
        result_pks = cluster_entity.get_source_pks(source_name)
        assert result_pks is not None

        # All rows for a given cluster entity should share the same company value
        result_rows = data_df[data_df["pk"].isin(result_pks)]
        assert len(result_rows["company"].unique()) == 1

        company_values = result_rows["company"]
        # With drop_base=True, should only see variation values
        assert all(
            value.endswith(" Inc") or value.endswith(" Ltd") for value in company_values
        )


def test_base_and_variation_entities():
    """Test that base values and variations create correct ClusterEntity objects."""
    features = [
        FeatureConfig(
            name="company",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
            drop_base=False,  # Keep base value
        )
    ]

    source = source_factory(features=features, n_true_entities=1, seed=42)
    source_name = source.source.address.full_name

    # Should have two ClusterEntity objects - one for base, one for variation
    assert len(source.entities) == 2

    # Get the base and variation entities
    data_df = source.data.to_pandas()

    # We'll need to find the base value by examining the data
    # Get all unique company values
    all_company_values = data_df["company"].unique().tolist()

    # Identify which value is the base (doesn't end with " Inc")
    base_value = next(
        value for value in all_company_values if not value.endswith(" Inc")
    )
    variation_value = next(
        value for value in all_company_values if value.endswith(" Inc")
    )

    base_entity = None
    variation_entity = None

    for entity in source.entities:
        entity_pks = entity.get_source_pks(source_name)
        rows = data_df[data_df["pk"].isin(entity_pks)]
        values = rows["company"]
        assert len(values.unique()) == 1
        value = values.iloc[0]

        if value == base_value:
            base_entity = entity
        elif value == variation_value:
            variation_entity = entity

    assert base_entity is not None, "No ClusterEntity found for base value"
    assert variation_entity is not None, "No ClusterEntity found for variation"

    # Verify that each entity only contains its own variation
    base_values = base_entity.get_values({source_name: source})
    assert base_values[source_name]["company"] == [base_value]

    variation_values = variation_entity.get_values({source_name: source})
    assert variation_values[source_name]["company"] == [variation_value]

    # Together they should compose the full set of entity data
    combined = base_entity + variation_entity

    # Verify that the combined entity contains both variations
    combined_values = combined.get_values({source_name: source})
    assert sorted(combined_values[source_name]["company"]) == sorted(
        [base_value, variation_value]
    )

    # Verify that adding the entities produces the same result as having all PKs
    assert (
        combined.source_pks[source_name]
        == base_entity.source_pks[source_name]
        | variation_entity.source_pks[source_name]
    )

    # The diff between entities should match their respective PKs
    base_diff = base_entity - variation_entity
    assert base_diff.get(source_name) == base_entity.source_pks[source_name]

    variation_diff = variation_entity - base_entity
    assert variation_diff.get(source_name) == variation_entity.source_pks[source_name]


def test_source_factory_id_generation():
    """Test that source_factory generates unique IDs for rows."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
        ),
    ]

    n_true_entities = 2
    repetition = 2
    source = source_factory(
        n_true_entities=n_true_entities,
        repetition=repetition,
        features=features,
        seed=42,
    )

    # Convert to pandas for easier analysis
    data_df = source.data.to_pandas()

    # Each unique row combination (excluding pk) should get a different ID
    for _, group in data_df.groupby("company_name"):
        # All rows with same features should have same ID
        assert len(group["id"].unique()) == 1

    # Verify we're generating int64 IDs
    assert data_df["id"].dtype == "int64"

    # Different rows should have different IDs
    assert len(data_df["id"].unique()) == len(data_df["company_name"].unique())


@pytest.mark.parametrize(
    ("selected_entities", "features"),
    [
        pytest.param(
            # Base case: Two entities, one feature, unique values
            (
                SourceEntity(base_values={"name": "alpha"}),
                SourceEntity(base_values={"name": "beta"}),
            ),
            (FeatureConfig(name="name", base_generator="name"),),
            id="two_entities_unique_values",
        ),
        pytest.param(
            # Case: Two entities with identical values - should share IDs
            (
                SourceEntity(base_values={"name": "alpha"}),
                SourceEntity(base_values={"name": "alpha"}),
            ),
            (FeatureConfig(name="name", base_generator="name"),),
            id="two_entities_same_values",
        ),
        pytest.param(
            # Case: Multiple features, tests tuple-based identity
            (
                SourceEntity(base_values={"name": "alpha", "user_id": "123"}),
                SourceEntity(base_values={"name": "alpha", "user_id": "456"}),
            ),
            (
                FeatureConfig(name="name", base_generator="name"),
                FeatureConfig(name="user_id", base_generator="uuid4"),
            ),
            id="multiple_features_partial_match",
        ),
        pytest.param(
            # Case: Empty entities list - should handle gracefully
            (),
            (FeatureConfig(name="name", base_generator="name"),),
            id="empty_entities",
        ),
        pytest.param(
            # Case: Entity with variations and drop_base
            (SourceEntity(base_values={"name": "alpha"}),),
            (
                FeatureConfig(
                    name="name",
                    base_generator="name",
                    drop_base=True,
                    variations=(
                        ReplaceRule(old="a", new="@"),  # alpha -> @lph@
                        ReplaceRule(old="a", new="4"),  # alpha -> 4lph4
                    ),
                ),
            ),
            id="variations_with_drop_base",
        ),
        pytest.param(
            # Case: Entity with variations and drop_base
            (SourceEntity(base_values={"name": "alpha", "user_id": "123"}),),
            (
                FeatureConfig(
                    name="name",
                    base_generator="name",
                    drop_base=True,
                    variations=(
                        ReplaceRule(old="a", new="@"),  # alpha -> @lph@
                        ReplaceRule(old="a", new="4"),  # alpha -> 4lph4
                    ),
                ),
                FeatureConfig(
                    name="user_id",
                    base_generator="uuid4",
                    # No variations, keeps base value
                ),
            ),
            id="mixed_variations_and_drop_base",
        ),
        pytest.param(
            # Case: Multiple entities with mixed variation configs
            (
                SourceEntity(base_values={"name": "alpha", "title": "ceo"}),
                SourceEntity(base_values={"name": "beta", "title": "cto"}),
            ),
            (
                FeatureConfig(
                    name="name",
                    base_generator="name",
                    drop_base=False,  # Keeps original
                    variations=(ReplaceRule(old="a", new="@"),),
                ),
                FeatureConfig(
                    name="title",
                    base_generator="job",
                    drop_base=True,  # Drops original
                    variations=(
                        ReplaceRule(old="o", new="0"),
                        ReplaceRule(old="e", new="3"),  # Won't affect CTO
                    ),
                ),
            ),
            id="multiple_entities_mixed_variations",
        ),
    ],
)
def test_generate_rows(
    selected_entities: tuple[SourceEntity, ...],
    features: tuple[FeatureConfig, ...],
):
    """Test generate_rows correctly tracks entities and row identities."""
    generator = Faker(seed=42)
    raw_data, entity_pks, id_pks = generate_rows(generator, selected_entities, features)

    # Check arrays have consistent lengths
    n_rows = len(raw_data["pk"])
    assert len(raw_data["id"]) == n_rows
    assert all(len(values) == n_rows for values in raw_data.values())

    # Check entity tracking - each entity appears exactly once
    assert len(selected_entities) == len(entity_pks)

    # Check row identity tracking - each unique value combo gets one ID
    unique_values = {
        tuple(raw_data[f.name][i] for f in features) for i in range(n_rows)
    }
    assert len(unique_values) == len(id_pks)

    # When we have duplicate values, verify correct ID sharing
    value_counts = {}
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]
        value_counts[values] = value_counts.get(values, 0) + 1

    # Each ID's PKs set should match the number of times those values appear
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]
        assert len(id_pks[row_id]) == value_counts[values]

    # Verify all PKs are accounted for
    all_pks = set(raw_data["pk"])
    assert all(pk in all_pks for pks in entity_pks.values() for pk in pks)
    assert all(pk in all_pks for pks in id_pks.values() for pk in pks)

    # For empty entities case, verify empty results
    if not selected_entities:
        assert not raw_data["pk"]
        assert not entity_pks
        assert not id_pks

    # Verify core variation behavior
    for entity in selected_entities:
        entity_rows = {
            i for i, pk in enumerate(raw_data["pk"]) if pk in entity_pks[entity.id]
        }

        for feature in features:
            values = {raw_data[feature.name][i] for i in entity_rows}
            base_value = entity.base_values[feature.name]

            # Check if base values are included/excluded correctly
            if feature.drop_base:
                assert base_value not in values
            elif not feature.variations:
                assert values == {base_value}

            # Count effective variations (those that actually change the value)
            effective_variations = [
                rule.apply(base_value)
                for rule in feature.variations
                if rule.apply(base_value) != base_value
            ]

            # Check that variations were generated
            if feature.variations:
                expected_count = len(effective_variations) + (
                    0 if feature.drop_base else 1
                )
                assert len(values) == expected_count

    # Verify row count matches expectations
    for entity in selected_entities:
        # Count effective variations for each feature
        variation_counts = []
        for feature in features:
            base_value = entity.base_values[feature.name]
            effective_variations = [
                rule.apply(base_value)
                for rule in feature.variations
                if rule.apply(base_value) != base_value
            ]
            # Count options: variations + (base if keeping it)
            if feature.drop_base and effective_variations:
                variation_counts.append(len(effective_variations))
            else:
                variation_counts.append(len(effective_variations) + 1)

        # Multiply all counts together to get total combinations
        expected_rows = functools.reduce(lambda x, y: x * y, variation_counts, 1)
        assert len(entity_pks[entity.id]) == expected_rows
