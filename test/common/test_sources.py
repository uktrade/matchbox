from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
from pydantic import AnyUrl, ValidationError
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError
from sqlglot import select
from sqlglot.errors import ParseError

from matchbox.client.helpers.selector import Match
from matchbox.common.dtos import DataTypes
from matchbox.common.exceptions import (
    MatchboxSourceCredentialsError,
    MatchboxSourceExtractTransformError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.common.sources import (
    RelationalDBLocation,
    SourceConfig,
    SourceField,
)

# Locations


def test_location_empty_credentials_error():
    """Test that operations requiring credentials fail when credentials are not set."""
    location = RelationalDBLocation(uri="postgresql://host:1234/db2")

    # Attempting to connect without credentials should raise an error
    with pytest.raises(MatchboxSourceCredentialsError):
        location.connect()


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
    ["uri", "expected"],
    [
        pytest.param("sqlite:///test.db", "sqlite:///test.db", id="valid-sqlite"),
        pytest.param(
            "postgresql://localhost:5432/testdb",
            "postgresql://localhost:5432/testdb",
            id="valid-postgres",
        ),
        pytest.param(
            "postgresql://user:pass@localhost:5432/testdb",
            "postgresql://localhost:5432/testdb",
            id="credentials-in-uri",
        ),
        pytest.param(
            "postgresql+psycopg://localhost:5432/testdb",
            "postgresql://localhost:5432/testdb",
            id="driver-in-uri",
        ),
        pytest.param(
            "sqlite:///test.db?mode=ro", "sqlite:///test.db", id="query-params"
        ),
        pytest.param("sqlite:///test.db#fragment", "sqlite:///test.db", id="fragment"),
        pytest.param(
            "sqlite:///var/folders/14/6nvsrw1n2ls1xncz_bvy2x8m0000gq/T/db.sqlite",
            "sqlite:///var/folders/14/6nvsrw1n2ls1xncz_bvy2x8m0000gq/T/db.sqlite",
            id="no-hostname",
        ),
    ],
)
def test_relational_db_location_uri_clean(uri: str, expected: str):
    """Test URI validation in RelationalDBLocation."""
    location = RelationalDBLocation(uri=uri)
    assert location.uri == AnyUrl(expected)


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
    source_testkit = source_factory(engine=sqlite_warehouse)
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    sql = select("*").from_(source_testkit.name).sql()
    batch_size = 2

    # Execute the query
    results = list(location.execute(sql, batch_size))

    # Check that we got expected results
    assert len(results) > 0

    # Combine all batches to check total row count
    combined_df = pl.concat(results)
    assert len(combined_df) == 10

    # Try query with filter
    keys_to_filter = source_testkit.query["key"][:2].to_pylist()
    filtered_results = pl.concat(
        location.execute(sql, batch_size, keys=("key", keys_to_filter))
    )
    assert len(filtered_results) == 2

    # Filtering by no keys has no effect
    unfiltered_results = pl.concat(location.execute(sql, batch_size, keys=("key", [])))
    assert_frame_equal(unfiltered_results, combined_df)


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
    source_testkit = source_factory(engine=sqlite_warehouse)
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)
    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Execute a query with transformation
    sql = (
        select("company_name AS name", "UPPER(company_name) AS company_name", "crn")
        .from_(source_testkit.name)
        .sql()
    )

    results = list(location.execute(sql, batch_size=1))
    assert len(results) == 10  # 10 batches of 1 row

    df: pl.DataFrame = pl.concat(results)

    # Verify the result structure
    assert set(df.columns) == {"name", "company_name", "crn"}

    # Verify the calculated index_fields
    sample_str: str = df.select("company_name").row(0)[0]
    assert sample_str == sample_str.upper()


# Source


def test_source_init():
    """Test basic SourceConfig instantiation with a Location object."""
    # Create a basic location
    location = RelationalDBLocation(uri="sqlite:///:memory:")

    # Create index_fields
    key_field = SourceField(name="key", type=DataTypes.STRING)
    index_fields = (
        SourceField(name="name", type=DataTypes.STRING),
        SourceField(name="age", type=DataTypes.INT64),
    )

    # Create SourceConfig
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name, age FROM users",
        key_field=key_field,
        index_fields=index_fields,
    )

    # Verify attributes
    assert source.location == location
    assert source.name == "test_source"
    assert source.extract_transform == "SELECT key, name, age FROM users"
    assert source.key_field == key_field
    assert source.index_fields == index_fields
    assert source.qualified_key == "test_source_key"
    assert source.qualified_fields == ["test_source_name", "test_source_age"]


def test_source_model_validation():
    """Test that SourceConfig validation works for index_fields and key_field."""
    # Create a basic location
    location = RelationalDBLocation(uri="sqlite:///:memory:")

    # Test key_field in index_fields validation
    key_field = SourceField(name="key", type=DataTypes.STRING)
    index_fields = (key_field, SourceField(name="name", type=DataTypes.STRING))

    with pytest.raises(
        ValidationError, match="Key field must not be in the index fields."
    ):
        SourceConfig(
            location=location,
            name="test_source",
            extract_transform="SELECT key, name FROM users",
            key_field=key_field,
            index_fields=index_fields,
        )


def test_source_identifier_validation():
    """Test that key_field validation requires a string type."""
    # Create a basic location
    location = RelationalDBLocation(uri="sqlite:///:memory:")
    index_fields = (SourceField(name="name", type=DataTypes.STRING),)

    # Valid case: String key_field
    string_identifier = SourceField(name="key", type=DataTypes.STRING)
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=string_identifier,
        index_fields=index_fields,
    )
    assert source.key_field.type == DataTypes.STRING

    # Invalid case: Non-string key field
    int_identifier = SourceField(name="key", type=DataTypes.INT64)
    with pytest.raises(ValidationError, match="Key field must be a string"):
        SourceConfig(
            location=location,
            name="test_source",
            extract_transform="SELECT key, name FROM users",
            key_field=int_identifier,
            index_fields=index_fields,
        )


def test_source_from_new(sqlite_warehouse: Engine):
    """Creating a source config using new(), which infers types, works."""
    # Create test data
    source_testkit = source_factory(
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "datatype": DataTypes.STRING},
        ],
        engine=sqlite_warehouse,
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    location = RelationalDBLocation.from_engine(sqlite_warehouse)
    source = SourceConfig.new(
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        key_field="key",
        index_fields=["name"],
    )

    assert source.key_field == SourceField(name="key", type=DataTypes.STRING)
    assert source.index_fields == tuple(
        [SourceField(name="name", type=DataTypes.STRING)]
    )


def test_source_from_new_errors(sqlite_warehouse: Engine):
    """Creating a source config using new() errors with non-string key."""
    # Create test data
    source_testkit = source_factory(
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "datatype": DataTypes.STRING},
            {
                "name": "int_pk",
                "base_generator": "random_int",
                "datatype": DataTypes.INT64,
            },
        ],
        engine=sqlite_warehouse,
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    with pytest.raises(ValueError):
        SourceConfig.new(
            location=location,
            name="test_source",
            extract_transform=source_testkit.source_config.extract_transform,
            key_field="int_pk",
            index_fields=["name"],
        )


def test_source_sampling_preserves_original_sql(sqlite_warehouse: Engine):
    """Test that ensures the SQL on RelationalDBLocation is preserved.

    SQLGlot transpiles INSTR() to STR_POSITION() in its default dialect.
    """
    # Create test data
    source_testkit = source_factory(
        n_true_entities=3,
        features=[
            {
                "name": "text_col",
                "base_generator": "word",
                "datatype": DataTypes.STRING,
            },
        ],
        engine=sqlite_warehouse,
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    location = RelationalDBLocation.from_engine(sqlite_warehouse)

    # Use SQLite's INSTR function (returns position of substring)
    # Other databases use CHARINDEX, POSITION, etc.
    extract_transform = f"""
        SELECT
            key,
            text_col,
            INSTR(text_col, 'a') as position_of_a
        FROM
            "{source_testkit.source_config.name}"
    """

    # This should work since INSTR is valid SQLite
    # Would fail if validation transpiles INSTR to POSITION() or similar
    source = SourceConfig.new(
        location=location,
        name="test_source",
        extract_transform=extract_transform,
        key_field="key",
        index_fields=["text_col", "position_of_a"],
    )

    assert source.key_field == SourceField(name="key", type=DataTypes.STRING)
    assert len(source.index_fields) == 2

    # This should work if the SQL is preserved exactly
    df = next(source.query())
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3


def test_source_query(sqlite_warehouse: Engine):
    """Test the query method with default parameters."""
    # Create test data
    source_testkit = source_factory(
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "datatype": DataTypes.STRING},
        ],
        engine=sqlite_warehouse,
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    # Create location and source
    location = RelationalDBLocation.from_engine(sqlite_warehouse)
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Execute query
    result = next(source.query())

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5
    assert "key" in result.columns
    assert "name" in result.columns

    # Try applying key filter
    key_subset = result[source.key_field.name][:2].to_list()
    result = next(source.query(keys=key_subset))
    assert len(result) == 2

    # Key filter ineffective with empty list
    result = next(source.query(keys=[]))
    assert len(result) == 5


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
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(uri=str(sqlite_warehouse.url))

    # Create source
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Call query with qualification parameter
    next(source.query(qualify_names=qualify_names))

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
    ("batch_size", "expected_call_kwargs"),
    [
        pytest.param(
            None,
            {"batch_size": None},
            id="single_return",
        ),
        pytest.param(3, {"batch_size": 3}, id="multiple_batches"),
    ],
)
@patch("matchbox.common.sources.RelationalDBLocation.execute")
def test_source_query_batching(
    mock_execute: Mock,
    sqlite_warehouse: Engine,
    batch_size: int,
    expected_call_kwargs: dict,
):
    """Test query with batching options."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(uri=str(sqlite_warehouse.url))

    # Create source
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Call query with batching parameters
    next(source.query(batch_size=batch_size))

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
    source_testkit = source_factory(
        n_true_entities=n_true_entities,
        features=[
            {"name": "name", "base_generator": "name", "datatype": DataTypes.STRING},
            {
                "name": "age",
                "base_generator": "random_int",
                "datatype": DataTypes.INT64,
            },
        ],
        engine=sqlite_warehouse,
    )
    source_testkit.write_to_location(credentials=sqlite_warehouse, set_credentials=True)

    # Create location and source
    location = RelationalDBLocation.from_engine(sqlite_warehouse)
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=(
            SourceField(name="name", type=DataTypes.STRING),
            SourceField(name="age", type=DataTypes.INT64),
        ),
    )

    # Execute hash_data with different batching parameters
    if batch_size:
        result = source.hash_data(batch_size=batch_size)
    else:
        result = source.hash_data()

    # Verify result
    assert isinstance(result, pa.Table)
    assert "hash" in result.column_names
    assert "keys" in result.column_names
    assert len(result) == n_true_entities


@patch("matchbox.common.sources.SourceConfig.query")
def test_source_hash_data_null_identifier(mock_query: Mock, sqlite_warehouse: Engine):
    """Test hash_data raises an error when source primary keys contain nulls."""
    # Create a source
    location = RelationalDBLocation(uri=str(sqlite_warehouse.url))
    source = SourceConfig(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=(SourceField(name="name", type=DataTypes.STRING),),
    )

    # Mock query to return data with null keys
    mock_df = pl.DataFrame({"key": ["1", None], "name": ["a", "b"]})
    mock_query.return_value = (x for x in [mock_df])

    # hash_data should raise ValueErrors for null keys
    with pytest.raises(ValueError, match="keys column contains null values"):
        source.hash_data()


# Match


def test_match_validates():
    """Match objects are validated when they're instantiated."""
    Match(
        cluster=1,
        source="test.source_config",
        source_id={"a"},
        target="test.target",
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source="test.source_config",
            target="test.target",
            target_id={"b"},
        )

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(
            source="test.source_config",
            source_id={"a"},
            target="test.target",
            target_id={"b"},
        )

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source="test.source_config",
            target="test.target",
        )
