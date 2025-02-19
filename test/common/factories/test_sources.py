from sqlalchemy import create_engine

from matchbox.common.arrow import SCHEMA_INDEX
from matchbox.common.factories.sources import (
    DropBaseRule,
    FeatureConfig,
    SuffixRule,
    source_factory,
)
from matchbox.common.sources import SourceAddress


def test_source_factory_default():
    """Test that source_factory generates a dummy source with default parameters."""
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


def test_source_dummy_to_mock():
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


def test_entity_variations_tracking():
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
