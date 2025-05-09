from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from pydantic import AnyUrl, ValidationError
from sqlalchemy import (
    Engine,
    create_engine,
)
from sqlalchemy.exc import OperationalError
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from matchbox.client.helpers.selector import Match
from matchbox.common.dtos import DataTypes
from matchbox.common.exceptions import (
    MatchboxSourceCredentialsError,
    MatchboxSourceExtractTransformError,
)
from matchbox.common.factories.locations import location_factory
from matchbox.common.factories.sources import source_factory
from matchbox.common.sources import (
    Location,
    RelationalDBLocation,
    SourceAddress,
    SourceConfig,
    SourceField,
)

# Locations


def test_location_factory():
    """Test we can construct appropriate Location classes from raw data."""
    location = Location.create(
        {
            "type": "rdbms",
            "uri": "postgresql://host:1234/db2",
        }
    )
    assert isinstance(location, RelationalDBLocation)


def test_location_empty_credentials_error():
    """Test that operations requiring credentials fail when credentials are not set."""
    location = RelationalDBLocation(uri="postgresql://host:1234/db2")

    # Attempting to connect without credentials should raise an error
    with pytest.raises(MatchboxSourceCredentialsError):
        location.connect()

    # Invalid location type
    with pytest.raises(ValueError, match="Unknown location type"):
        Location.create({"type": "unknown", "uri": "http://example.com"})

    # Missing required fields
    with pytest.raises(ValueError):
        Location.create({"type": "rdbms"})


def test_location_serialisation():
    """Test serialisation and deserialisation of Location objects."""
    original = RelationalDBLocation(uri="postgresql://host:1234/db2")

    # Convert to dict and back - credentials should be excluded
    location_dict = original.model_dump()
    assert "credentials" not in location_dict

    # Deserialize back to a Location
    reconstructed = RelationalDBLocation.model_validate(location_dict)
    assert reconstructed.uri == original.uri
    assert reconstructed.type == original.type
    assert reconstructed.credentials is None


def test_relational_db_location_instantiation():
    """Test that RelationalDBLocation can be instantiated with valid parameters."""
    location = RelationalDBLocation(uri="sqlite:///test.db")
    assert location.type == "rdbms"
    assert str(location.uri) == "sqlite:///test.db"
    assert location.credentials is None


@pytest.mark.parametrize(
    ["uri_str", "should_pass"],
    [
        pytest.param("sqlite:///test.db", True, id="valid-sqlite"),
        pytest.param("postgresql://localhost:5432/testdb", True, id="valid-postgres"),
        pytest.param(
            "postgresql://user:pass@localhost:5432/testdb",
            False,
            id="invalid-credentials-in-uri",
        ),
        pytest.param(
            "postgresql+psycopg://localhost:5432/testdb",
            False,
            id="driver-in-uri",
        ),
        pytest.param("sqlite:///test.db?mode=ro", False, id="invalid-query-params"),
        pytest.param("sqlite:///test.db#fragment", False, id="invalid-fragment"),
    ],
)
def test_relational_db_location_uri_validation(uri_str: str, should_pass: bool):
    """Test URI validation in RelationalDBLocation."""
    if should_pass:
        location = RelationalDBLocation(uri=uri_str)
        assert str(location.uri) == uri_str
    else:
        with pytest.raises(ValueError):
            RelationalDBLocation(uri=AnyUrl(uri_str))


def test_relational_db_add_credentials(sqlite_warehouse: Engine):
    """Test the public interface for adding credentials to a RelationalDBLocation.

    This test verifies:
        1. Credentials are properly added and validated for matching engines
        2. Validation fails for non-matching engines
        3. Connect succeeds only with valid credentials
    """
    # Create engines and URIs
    test_engines = {
        # PostgreSQL engines with different connection parameters
        "pg": create_engine("postgresql://user:pass@host:5432/db"),  # trufflehog:ignore
        "pg_diff_host": create_engine(
            "postgresql://user:pass@otherhost:5432/db"  # trufflehog:ignore
        ),
        "pg_diff_port": create_engine(
            "postgresql://user:pass@host:5433/db"  # trufflehog:ignore
        ),
        "pg_diff_db": create_engine(
            "postgresql://user:pass@host:5432/otherdb"  # trufflehog:ignore
        ),
        # These should match the same URI as "pg"
        # (different user/pass/dialect don't affect matching)
        "pg_diff_user": create_engine(
            "postgresql://user2:pass@host:5432/db"  # trufflehog:ignore
        ),
        "pg_diff_pass": create_engine(
            "postgresql://user:pass2@host:5432/db"  # trufflehog:ignore
        ),
        "pg_diff_dialect": create_engine(
            "postgresql+psycopg://user:pass@host:5432/db"  # trufflehog:ignore
        ),
    }

    uris = {
        "pg": "postgresql://host:5432/db",
        "pg_diff_host": "postgresql://otherhost:5432/db",
        "pg_diff_port": "postgresql://host:5433/db",
        "pg_diff_db": "postgresql://host:5432/otherdb",
        "sqlite": str(sqlite_warehouse.url),
    }

    # Test case 1: Adding matching credentials succeeds
    location1 = RelationalDBLocation(uri=uris["sqlite"])
    location1.add_credentials(sqlite_warehouse)
    assert location1.credentials == sqlite_warehouse
    assert location1.connect() is True

    # Test case 2: Adding mismatched credentials fails
    location2 = RelationalDBLocation(uri=uris["pg"])
    with pytest.raises(ValueError, match="does not match"):
        location2.add_credentials(sqlite_warehouse)

    # Test case 3: Check which parameters affect URI matching
    # Group by expected URI matching behavior
    should_match_pg = {"pg", "pg_diff_user", "pg_diff_pass", "pg_diff_dialect"}
    should_not_match_pg = {"pg_diff_host", "pg_diff_port", "pg_diff_db"}

    # Verify that add_credentials works for matching engines
    base_pg_location = RelationalDBLocation(uri=uris["pg"])
    base_pg_location.add_credentials(test_engines["pg"])
    assert base_pg_location.credentials == test_engines["pg"]

    # All engines in should_match_pg should work with this location
    for engine_key in should_match_pg:
        test_location = RelationalDBLocation(uri=uris["pg"])
        test_location.add_credentials(test_engines[engine_key])
        assert test_location.credentials == test_engines[engine_key]

    # All engines in should_not_match_pg should fail with this location
    for engine_key in should_not_match_pg:
        test_location = RelationalDBLocation(uri=uris["pg"])
        with pytest.raises(ValueError, match="does not match"):
            test_location.add_credentials(test_engines[engine_key])


@pytest.mark.parametrize(
    ["sql", "is_valid"],
    [
        pytest.param("SELECT * FROM test_table", True, id="valid-select"),
        pytest.param(
            "SELECT id, name FROM test_table WHERE id > 1", True, id="valid-where"
        ),
        pytest.param("SLECT * FROM test_table", False, id="invalid-syntax"),
        pytest.param("", False, id="empty-string"),
        pytest.param("ALTER TABLE test_table", False, id="alter-sql"),
        pytest.param(
            "INSERT INTO users (name, age) VALUES ('John', '25')",
            False,
            id="insert-sql",
        ),
        pytest.param("DROP TABLE test_table", False, id="drop-sql"),
        pytest.param("SELECT * FROM users /* with a comment */", True, id="comment"),
        pytest.param(
            "WITH user_cte AS (SELECT * FROM users) SELECT * FROM user_cte",
            True,
            id="valid-with",
        ),
        pytest.param(
            (
                "WITH user_cte AS (SELECT * FROM users) "
                "INSERT INTO temp_users SELECT * FROM user_cte"
            ),
            False,
            id="invalid-with",
        ),
        pytest.param(
            "SELECT * FROM users; DROP TABLE users;", False, id="multiple-statements"
        ),
        pytest.param("SELECT * INTO new_table FROM users", False, id="select-into"),
        pytest.param(
            """
            WITH updated_rows AS (
                UPDATE employees
                SET salary = salary * 1.1
                WHERE department = 'Sales'
                RETURNING *
            )
            SELECT * FROM updated_rows;
            """,
            False,
            id="non-query-cte",
        ),
    ],
)
def test_relational_db_extract_transform(sql: str, is_valid: bool):
    """Test SQL validation in validate_extract_transform."""
    location = RelationalDBLocation(uri="postgresql://host:1234/db2")

    if is_valid:
        assert location.validate_extract_transform(sql)
    else:
        with pytest.raises((MatchboxSourceExtractTransformError, ParseError)):
            location.validate_extract_transform(sql)


def test_relational_db_execute(sqlite_warehouse: Engine):
    """Test executing a query and returning results using a real SQLite database."""
    location_config = location_factory(
        location_type="rdbms", uri=str(sqlite_warehouse.url)
    )
    source_testkit = source_factory(location_config=location_config, n_true_entities=10)
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    location = RelationalDBLocation.from_engine(sqlite_warehouse)
    batch_size = 2

    # Execute the query
    results = list(
        location.execute(source_testkit.config.extract_transform, batch_size)
    )

    # Check that we got expected results
    assert len(results) > 0

    # Combine all batches to check total row count
    combined_df = pl.concat(results)
    assert len(combined_df) == 10


def test_relational_db_execute_invalid(sqlite_warehouse: Engine):
    """Test that invalid queries are handled correctly when executing."""
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Invalid SQL query
    sql = "SELECT * FROM nonexistent_table"

    # Should raise an exception when executed
    with pytest.raises(OperationalError):
        list(location.execute(sql, batch_size=10))


def test_relational_db_from_engine(sqlite_warehouse: Engine):
    """Test creating a RelationalDBLocation from an engine."""
    # Create a location from the engine
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Check that it was created correctly
    assert location.credentials == sqlite_warehouse
    assert location.type == "rdbms"

    # Verify it works by connecting
    assert location.connect() is True


def test_relational_db_retrieval_and_transformation(sqlite_warehouse: Engine):
    """Test a more complete workflow with data retrieval and transformation."""
    location_config = location_factory(
        location_type="rdbms", uri=str(sqlite_warehouse.url)
    )
    source_testkit = source_factory(location_config=location_config, n_true_entities=10)
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Execute a query with transformation
    sql = f"""
    SELECT 
        company_name as name,
        UPPER(company_name) as company_name,
        crn as crn
    FROM (
        {parse_one(source_testkit.config.extract_transform).sql()}
    );
    """

    results = list(location.execute(sql, batch_size=1))
    assert len(results) == 10  # 10 batches of 1 row

    df = pl.concat(results)

    # Verify the result structure
    assert set(df.columns) == {"name", "company_name", "crn"}

    # Verify the calculated columns
    sample_str: str = df.select("company_name").row(0)[0]
    assert sample_str == sample_str.upper()


# SourceConfigs


def test_source_init():
    """Test basic SourceConfig instantiation with a Location object."""
    # Create a basic location
    location = RelationalDBLocation(uri="sqlite:///:memory:")

    # Create fields
    identifier = SourceField(name="id", type=DataTypes.STRING)
    fields = (
        SourceField(name="name", type=DataTypes.STRING),
        SourceField(name="age", type=DataTypes.INT64),
    )

    # Create SourceConfig
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT id, name, age FROM users",
        identifier=identifier,
        fields=fields,
    )

    # Verify attributes
    assert source.location == location
    assert source.name == "test_source"
    assert source.extract_transform == "SELECT id, name, age FROM users"
    assert source.identifier == identifier
    assert source.fields == fields


def test_source_model_validation():
    """Test that SourceConfig validation works for fields and identifier."""
    # Create a basic location
    location = RelationalDBLocation(uri="sqlite:///:memory:")

    # Test identifier in fields validation
    identifier = SourceField(name="id", type=DataTypes.STRING)
    fields = (identifier, SourceField(name="name", type=DataTypes.STRING))

    with pytest.raises(ValidationError, match="Identifier must not be in the fields"):
        SourceConfig(
            location=location,
            name="test_source",
            extract_transform="SELECT id, name FROM users",
            identifier=identifier,
            fields=fields,
        )


def test_source_identifier_validation():
    """Test that identifier validation requires a string type."""
    # Create a basic location
    location = RelationalDBLocation(uri="sqlite:///:memory:")
    fields = (SourceField(name="name", type=DataTypes.STRING),)

    # Valid case: String identifier
    string_identifier = SourceField(name="id", type=DataTypes.STRING)
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT id, name FROM users",
        identifier=string_identifier,
        fields=fields,
    )
    assert source.identifier.type == DataTypes.STRING

    # Invalid case: Non-string identifier
    int_identifier = SourceField(name="id", type=DataTypes.INT64)
    with pytest.raises(ValidationError, match="Identifier must be a string"):
        SourceConfig(
            location=location,
            name="test_source",
            extract_transform="SELECT id, name FROM users",
            identifier=int_identifier,
            fields=fields,
        )


def test_source_from_location(sqlite_warehouse: Engine):
    """Test the from_location factory method with minimal parameters."""
    # Create a location with credentials
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Create test data and write to warehouse
    location_config = location_factory(
        location_type="rdbms", uri=str(sqlite_warehouse.url)
    )
    source_testkit = source_factory(
        location_config=location_config,
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "sql_type": "TEXT"},
        ],
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    # Use the factory method
    table_name = (
        parse_one(source_testkit.config.extract_transform).find(exp.Table).alias_or_name
    )
    extract_transform = f"SELECT pk as id, name FROM {table_name}"
    source = SourceConfig.from_location(
        location=location, extract_transform=extract_transform
    )

    # Verify the created source
    assert source.location == location
    assert source.extract_transform == extract_transform
    assert source.identifier.name == "id"
    assert source.identifier.type == DataTypes.STRING
    assert len(source.fields) == 1
    assert source.fields[0].name == "name"
    assert source.fields[0].type == DataTypes.STRING
    assert source.name.startswith(Path(str(sqlite_warehouse.url)).stem)


def test_source_field_detection_from_location(sqlite_warehouse: Engine):
    """Test automatic field detection through from_location factory method."""
    # Create a location with credentials
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Create test data with different column types
    location_config = location_factory(
        location_type="rdbms", uri=str(sqlite_warehouse.url)
    )
    source_testkit = source_factory(
        location_config=location_config,
        n_true_entities=5,
        features=[
            {"name": "age", "base_generator": "random_int", "sql_type": "INTEGER"},
            {"name": "score", "base_generator": "pyfloat", "sql_type": "REAL"},
        ],
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    # Use the from_location factory method which internally uses field detection
    table_name = (
        parse_one(source_testkit.config.extract_transform).find(exp.Table).alias_or_name
    )
    extract_transform = f"SELECT pk as id, age, score FROM {table_name}"
    source = SourceConfig.from_location(
        location=location, extract_transform=extract_transform
    )

    # Verify detection results through the created source
    assert source.identifier.name == "id"
    assert source.identifier.type == DataTypes.STRING
    assert len(source.fields) == 2

    # Check field names and types
    field_dict = {field.name: field.type for field in source.fields}
    assert "age" in field_dict
    assert "score" in field_dict
    assert field_dict["age"] == DataTypes.INT64
    assert field_dict["score"] == DataTypes.FLOAT64


def test_source_set_credentials(sqlite_warehouse: Engine):
    """Test that credentials can be set on the Location via SourceConfig."""
    # Create a location without credentials
    location = RelationalDBLocation(uri=str(sqlite_warehouse.url))
    assert location.credentials is None

    # Create a source with this location
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT id, name FROM users",
        identifier=SourceField(name="id", type=DataTypes.STRING),
        fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Set credentials through the source
    source.set_credentials(sqlite_warehouse)

    # Verify credentials are set on the location
    assert source.location.credentials == sqlite_warehouse


def test_source_query(sqlite_warehouse: Engine):
    """Test the query method with default parameters."""
    # Create test data
    location_config = location_factory(
        location_type="rdbms", uri=str(sqlite_warehouse.url)
    )
    source_testkit = source_factory(
        location_config=location_config,
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "sql_type": "TEXT"},
        ],
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    # Create location and source
    location = RelationalDBLocation.from_engine(sqlite_warehouse)
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform=source_testkit.config.extract_transform,
        identifier=SourceField(name="pk", type=DataTypes.STRING),
        fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Execute query
    result = source.query()

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5
    assert "pk" in result.columns
    assert "name" in result.columns


@pytest.mark.parametrize(
    "qualify_names",
    [
        pytest.param(False, id="no_name_qualification"),
        pytest.param(True, id="with_name_qualification"),
    ],
)
@patch("matchbox.common.sources.RelationalDBLocation.execute")
def test_source_query_name_qualification(
    mock_execute: Mock,
    sqlite_warehouse: Engine,
    qualify_names: bool,
):
    """Test that column names are qualified when requested."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = MagicMock()
    location = RelationalDBLocation(uri=str(sqlite_warehouse.url))

    # Create source
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT id, name FROM users",
        identifier=SourceField(name="id", type=DataTypes.STRING),
        fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Call query with qualification parameter
    source.query(qualify_names=qualify_names)

    # Verify the rename parameter passed to execute
    _, kwargs = mock_execute.call_args
    rename_param = kwargs.get("rename")

    if qualify_names:
        assert rename_param is not None
        assert callable(rename_param)
        # Test the rename function
        sample_col = "test_col"
        assert "test_source_" in source.name + "_" + sample_col
    else:
        assert rename_param is None


@pytest.mark.parametrize(
    ("return_batches", "batch_size", "expected_call_kwargs"),
    [
        pytest.param(
            False,
            None,
            {"return_batches": False, "batch_size": None},
            id="single_return",
        ),
        pytest.param(
            True, 3, {"return_batches": True, "batch_size": 3}, id="multiple_batches"
        ),
    ],
)
@patch("matchbox.common.sources.RelationalDBLocation.execute")
def test_source_query_batching(
    mock_execute: Mock,
    sqlite_warehouse: Engine,
    return_batches: bool,
    batch_size: int,
    expected_call_kwargs: dict,
):
    """Test query with batching options."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = MagicMock()
    location = RelationalDBLocation(uri=str(sqlite_warehouse.url))

    # Create source
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT id, name FROM users",
        identifier=SourceField(name="id", type=DataTypes.STRING),
        fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Call query with batching parameters
    source.query(return_batches=return_batches, batch_size=batch_size)

    # Verify parameters passed to execute
    _, kwargs = mock_execute.call_args
    for key, value in expected_call_kwargs.items():
        assert kwargs.get(key) == value


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(None, id="no_batching"),
        pytest.param(2, id="with_batching"),
    ],
)
def test_source_hash_data(sqlite_warehouse: Engine, batch_size: int):
    """Test the hash_data method produces expected hash format."""
    # Create test data with unique values
    n_true_entities = 3
    location_config = location_factory(
        location_type="rdbms", uri=str(sqlite_warehouse.url)
    )
    source_testkit = source_factory(
        location_config=location_config,
        n_true_entities=n_true_entities,
        features=[
            {"name": "name", "base_generator": "name", "sql_type": "TEXT"},
            {"name": "age", "base_generator": "random_int", "sql_type": "INTEGER"},
        ],
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    # Create location and source
    location = RelationalDBLocation.from_engine(sqlite_warehouse)
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform=source_testkit.config.extract_transform,
        identifier=SourceField(name="pk", type=DataTypes.STRING),
        fields=(
            SourceField(name="name", type=DataTypes.STRING),
            SourceField(name="age", type=DataTypes.INT64),
        ),
    )

    # Execute hash_data with different batching parameters
    result = source.hash_data(batch_size=batch_size)

    # Verify result
    assert isinstance(result, pa.Table)
    assert "hash" in result.column_names
    assert "source_identifier" in result.column_names
    assert len(result) == n_true_entities


@patch("matchbox.common.sources.SourceConfig.query")
def test_source_hash_data_null_identifier(mock_query: Mock, sqlite_warehouse: Engine):
    """Test hash_data raises an error when source primary keys contain nulls."""
    # Create a source
    location = RelationalDBLocation(uri=str(sqlite_warehouse.url))
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT id, name FROM users",
        identifier=SourceField(name="id", type=DataTypes.STRING),
        fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Mock query to return data with null PKs
    mock_df = pl.DataFrame({"id": ["1", None], "name": ["a", "b"]})
    mock_query.return_value = mock_df

    # hash_data should raise ValueErrors for null PKs
    with pytest.raises(
        ValueError, match="source_identifier column contains null values"
    ):
        source.hash_data()


def test_source_validation_location_et_fields(sqlite_warehouse: Engine):
    """Test SourceConfig validation of location, ext/trans, and fields alignment.

    Tests three scenarios:
    1. Valid alignment between location, extract_transform, and fields
    2. Skip validation when credentials are not set
    3. Error when fields don't match the results of extract_transform
    """
    # Create test data
    location_config = location_factory(
        location_type="rdbms", uri=str(sqlite_warehouse.url)
    )
    source_testkit = source_factory(
        location_config=location_config,
        n_true_entities=2,
        features=[
            {"name": "name", "base_generator": "word", "sql_type": "TEXT"},
        ],
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    # Scenario 1: Valid alignment
    location = RelationalDBLocation.from_engine(sqlite_warehouse)
    extract_transform = source_testkit.config.extract_transform

    # This should validate successfully
    SourceConfig(
        location=location,
        name="test_source",
        extract_transform=extract_transform,
        identifier=SourceField(name="pk", type=DataTypes.STRING),
        fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Scenario 2: Skip validation when credentials are not set
    location_no_creds = RelationalDBLocation(
        uri=str(sqlite_warehouse.url).split("?")[0]
    )

    # This should not raise validation errors as credentials aren't set
    SourceConfig(
        location=location_no_creds,
        name="test_source",
        extract_transform=extract_transform,
        identifier=SourceField(name="pk", type=DataTypes.STRING),
        # Fields don't match what extract_transform would return, but we don't validate
        fields=(
            SourceField(name="name", type=DataTypes.STRING),
            SourceField(name="nonexistent", type=DataTypes.STRING),
        ),
    )

    # Scenario 3: Error when fields don't match
    with pytest.raises(
        ValidationError, match="do not match the extract/transform logic"
    ):
        SourceConfig(
            location=location,
            name="test_source",
            extract_transform=extract_transform,
            identifier=SourceField(name="pk", type=DataTypes.STRING),
            # This doesn't match the extract_transform
            fields=(
                SourceField(name="name", type=DataTypes.STRING),
                SourceField(name="nonexistent", type=DataTypes.STRING),
            ),
        )

    # Additional test: identifier doesn't match
    with pytest.raises(
        ValidationError, match="do not match the extract/transform logic"
    ):
        SourceConfig(
            location=location,
            name="test_source",
            extract_transform=extract_transform,
            # Wrong identifier name
            identifier=SourceField(name="wrong_id", type=DataTypes.STRING),
            fields=(SourceField(name="name", type=DataTypes.STRING),),
        )


# Match


def test_match_validation():
    """Test Match validation rules."""
    # Create source addresses
    source_addr = SourceAddress(full_name="source_table", warehouse_hash=b"hash1")
    target_addr = SourceAddress(full_name="target_table", warehouse_hash=b"hash2")

    # Valid match with cluster and source_id
    valid_match = Match(
        cluster=1,
        source=source_addr,
        source_id={"src1", "src2"},
        target=target_addr,
        target_id={"tgt1", "tgt2"},
    )

    assert valid_match.cluster == 1
    assert valid_match.source_id == {"src1", "src2"}
    assert valid_match.target_id == {"tgt1", "tgt2"}

    # Invalid: target_id but no cluster
    with pytest.raises(ValueError):
        Match(
            cluster=None,
            source=source_addr,
            source_id={"src1", "src2"},
            target=target_addr,
            target_id={"tgt1", "tgt2"},
        )

    # Invalid: cluster but no source_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=source_addr,
            source_id=set(),
            target=target_addr,
            target_id={"tgt1", "tgt2"},
        )

    # Valid: No target_id with cluster and source_id
    valid_no_target = Match(
        cluster=1,
        source=source_addr,
        source_id={"src1", "src2"},
        target=target_addr,
        target_id=set(),
    )

    assert valid_no_target.cluster == 1
    assert valid_no_target.source_id == {"src1", "src2"}
    assert valid_no_target.target_id == set()


def test_match_validates():
    """Match objects are validated when they're instantiated."""
    Match(
        cluster=1,
        source=SourceAddress(full_name="test.config", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.config", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(
            source=SourceAddress(full_name="test.config", warehouse_hash=b"bar"),
            source_id={"a"},
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
            target_id={"b"},
        )

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source=SourceAddress(full_name="test.config", warehouse_hash=b"bar"),
            target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        )
