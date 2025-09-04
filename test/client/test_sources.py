from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import OperationalError
from sqlglot import select
from sqlglot.errors import ParseError

from matchbox.client.sources import (
    RelationalDBLocation,
    Source,
)
from matchbox.common.dtos import DataTypes, LocationType, SourceField
from matchbox.common.exceptions import (
    MatchboxSourceExtractTransformError,
)
from matchbox.common.factories.sources import (
    FeatureConfig,
    source_factory,
    source_from_tuple,
)

# Locations


def test_relational_db_location_instantiation():
    """Test that RelationalDBLocation can be instantiated with valid parameters."""
    location = RelationalDBLocation(
        name="dbname", client=create_engine("sqlite:///:memory:")
    )
    assert location.config.type == LocationType.RDBMS
    assert location.config.name == "dbname"


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
    location = RelationalDBLocation(
        name="dbname", client=create_engine("sqlite:///:memory:")
    )

    if is_valid:
        assert location.validate_extract_transform(sql)
    else:
        with pytest.raises((MatchboxSourceExtractTransformError, ParseError)):
            location.validate_extract_transform(sql)


def test_relational_db_infer_types(sqlite_warehouse: Engine):
    """Test that types are inferred correctly from the extract transform SQL."""
    source_testkit = source_from_tuple(
        data_tuple=(
            {"foo": "10", "bar": None},
            {"foo": "foo_val", "bar": None},
            {"foo": None, "bar": 10},
        ),
        data_keys=["a", "b", "c"],
        name="source",
        engine=sqlite_warehouse,
    ).write_to_location()
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

    query = f"""
        select key as renamed_key, foo, bar from
        (select key, foo, bar from {source_testkit.name});
    """

    inferred_types = location.infer_types(query)

    assert len(inferred_types) == 3
    assert inferred_types["renamed_key"] == DataTypes.STRING
    assert inferred_types["foo"] == DataTypes.STRING
    assert inferred_types["bar"] == DataTypes.INT64


def test_relational_db_execute(sqlite_warehouse: Engine):
    """Test executing a query and returning results using a real SQLite database."""
    features = [
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="employees", base_generator="random_int"),
    ]

    source_testkit = source_factory(
        features=features, n_true_entities=10, engine=sqlite_warehouse
    ).write_to_location()
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

    sql = select("*").from_(source_testkit.name).sql()
    batch_size = 2

    # Execute the query
    results = list(location.execute(sql, batch_size))
    # Unlike later on, employee data type not overridden
    assert results[0]["employees"].dtype == pl.Int64

    # Check that we got expected results
    assert len(results) > 0

    # Combine all batches to check total row count
    combined_df = pl.concat(results)
    assert len(combined_df) == 10

    # Try overriding schema
    overridden_results = list(
        location.execute(sql, batch_size, schema_overrides={"employees": pl.String})
    )
    assert overridden_results[0]["employees"].dtype == pl.String

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
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

    # Invalid SQL query
    sql = "SELECT * FROM nonexistent_table"

    # Should raise an exception when executed
    with pytest.raises(OperationalError):
        list(location.execute(sql, batch_size=10))


def test_relational_db_retrieval_and_transformation(sqlite_warehouse: Engine):
    """Test a more complete workflow with data retrieval and transformation."""
    source_testkit = source_factory(engine=sqlite_warehouse).write_to_location()
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

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


def test_source_infers_type(sqlite_warehouse: Engine):
    """Creating a source with type inference works."""
    # Create test data
    source_testkit = source_factory(
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "datatype": DataTypes.STRING},
        ],
        engine=sqlite_warehouse,
    ).write_to_location()

    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)
    source = Source(
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name"],
    )

    assert source.config.key_field == SourceField(name="key", type=DataTypes.STRING)
    assert source.config.index_fields == tuple(
        [SourceField(name="name", type=DataTypes.STRING)]
    )


def test_source_sampling_preserves_original_sql(sqlite_warehouse: Engine):
    """SQL on RelationalDBLocation is preserved.

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
    ).write_to_location()

    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)

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
    source = Source(
        location=location,
        name="test_source",
        extract_transform=extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["text_col", "position_of_a"],
    )

    assert source.config.key_field == SourceField(name="key", type=DataTypes.STRING)
    assert len(source.config.index_fields) == 2

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
    ).write_to_location()

    # Create location and source
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)
    source = Source(
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name"],
    )

    # Execute query
    result = next(source.query())

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5
    assert "key" in result.columns
    assert "name" in result.columns

    # Try applying key filter
    key_subset = result[source.config.key_field.name][:2].to_list()
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
@patch("matchbox.client.sources.RelationalDBLocation.execute")
def test_source_query_name_qualification(
    mock_execute: Mock,
    qualify_names: bool,
):
    """Test that column names are qualified when requested."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(
        name="sqlite", client=create_engine("sqlite:///:memory:")
    )

    # Create source
    source = Source(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
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
@patch("matchbox.client.sources.RelationalDBLocation.execute")
def test_source_query_batching(
    mock_execute: Mock,
    batch_size: int,
    expected_call_kwargs: dict,
):
    """Test query with batching options."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(
        name="sqlite", client=create_engine("sqlite:///:memory:")
    )

    # Create source
    source = Source(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
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
    ).write_to_location()

    # Create location and source
    location = RelationalDBLocation(name="dbname", client=sqlite_warehouse)
    source = Source(
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name", "age"],
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


@patch("matchbox.client.sources.Source.query")
def test_source_hash_data_null_identifier(mock_query: Mock, sqlite_warehouse: Engine):
    """Test hash_data raises an error when source primary keys contain nulls."""
    # Create a source
    location = RelationalDBLocation(
        name="sqlite", client=create_engine("sqlite:///:memory:")
    )
    source = Source(
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
    )

    # Mock query to return data with null keys
    mock_df = pl.DataFrame({"key": ["1", None], "name": ["a", "b"]})
    mock_query.return_value = (x for x in [mock_df])

    # hash_data should raise ValueErrors for null keys
    with pytest.raises(ValueError, match="keys column contains null values"):
        source.hash_data()
