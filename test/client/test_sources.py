from datetime import datetime
from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlglot import select
from sqlglot.errors import ParseError

from matchbox.client.dags import DAG
from matchbox.client.sources import (
    RelationalDBLocation,
    Source,
)
from matchbox.common.dtos import (
    BackendResourceType,
    BackendUploadType,
    CRUDOperation,
    DataTypes,
    LocationType,
    NotFoundError,
    Resolution,
    ResolutionOperationStatus,
    SourceField,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxServerFileError,
    MatchboxSourceExtractTransformError,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import (
    FeatureConfig,
    source_factory,
    source_from_tuple,
)
from matchbox.common.graph import ResolutionType

# Locations


def test_relational_db_location_instantiation(sqlite_in_memory_warehouse: Engine):
    """Test that RelationalDBLocation can be instantiated with valid parameters."""
    location = RelationalDBLocation(name="dbname", client=sqlite_in_memory_warehouse)
    assert location.config.type == LocationType.RDBMS
    assert location.config.name == "dbname"


@pytest.mark.parametrize(
    ["sql", "is_valid"],
    [
        pytest.param("SELECT * FROM test_table", True, id="valid-select"),
        pytest.param(
            "SELECT id, name FROM test_table WHERE id > 1",
            True,
            id="valid-where",
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
        pytest.param(
            """
            SELECT foo, bar FROM baz
            UNION
            SELECT foo, bar FROM qux;
            """,
            True,
            id="valid-union",
        ),
        # This test only works with postgres
        pytest.param(
            """
            SELECT 'ciao' ~ 'hello'
            """,
            True,
            id="valid-tilde",
        ),
    ],
)
def test_relational_db_extract_transform(
    sql: str, is_valid: bool, postgres_warehouse: Engine
):
    """Test SQL validation in validate_extract_transform."""
    location = RelationalDBLocation(name="dbname", client=postgres_warehouse)

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
    keys_to_filter = source_testkit.data["key"][:2].to_pylist()
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
        dag=source_testkit.source.dag,
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
            "{source_testkit.source.to_resolution().name}"
    """

    # This should work since INSTR is valid SQLite
    # Would fail if validation transpiles INSTR to POSITION() or similar
    source = Source(
        dag=source_testkit.source.dag,
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
    df = next(source.fetch())
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3


def test_source_fetch(sqlite_warehouse: Engine):
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
        dag=source_testkit.source.dag,
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name"],
    )

    # Execute query
    result = next(source.fetch())

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5
    assert "key" in result.columns
    assert "name" in result.columns

    # Try applying key filter
    key_subset = result[source.config.key_field.name][:2].to_list()
    result = next(source.fetch(keys=key_subset))
    assert len(result) == 2

    # Key filter ineffective with empty list
    result = next(source.fetch(keys=[]))
    assert len(result) == 5


@pytest.mark.parametrize(
    "qualify_names",
    [
        pytest.param(False, id="no_name_qualification"),
        pytest.param(True, id="with_name_qualification"),
    ],
)
@patch("matchbox.client.sources.RelationalDBLocation.execute")
def test_source_fetch_name_qualification(
    mock_execute: Mock, qualify_names: bool, sqlite_in_memory_warehouse: Engine
):
    """Test that column names are qualified when requested."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(name="sqlite", client=sqlite_in_memory_warehouse)

    # Create source
    source = Source(
        dag=DAG("collection"),
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
    )

    # Call query with qualification parameter
    next(source.fetch(qualify_names=qualify_names))

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
def test_source_fetch_batching(
    mock_execute: Mock,
    batch_size: int,
    expected_call_kwargs: dict,
    sqlite_in_memory_warehouse: Engine,
):
    """Test query with batching options."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(name="sqlite", client=sqlite_in_memory_warehouse)

    # Create source
    source = Source(
        dag=DAG("collection"),
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
    )

    # Call query with batching parameters
    next(source.fetch(batch_size=batch_size))

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
def test_source_run(sqlite_warehouse: Engine, batch_size: int):
    """Test the run method produces expected hash format."""
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
        dag=source_testkit.source.dag,
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name", "age"],
    )

    # Execute run with different batching parameters
    result = source.run(batch_size=batch_size) if batch_size else source.run()

    # Verify result
    assert isinstance(result, pa.Table)
    assert "hash" in result.column_names
    assert "keys" in result.column_names
    assert len(result) == n_true_entities

    with pytest.warns(match="already run"):
        source.run()

    source.run(full_rerun=True)


@patch("matchbox.client.sources.Source.fetch")
def test_source_run_null_identifier(
    mock_fetch: Mock, sqlite_in_memory_warehouse: Engine
):
    """Test hashing data raises an error when source primary keys contain nulls."""
    # Create a source
    location = RelationalDBLocation(name="sqlite", client=sqlite_in_memory_warehouse)
    source = Source(
        dag=DAG("collection"),
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
    )

    # Mock query to return data with null keys
    mock_df = pl.DataFrame({"key": ["1", None], "name": ["a", "b"]})
    mock_fetch.return_value = (x for x in [mock_df])

    # hashing data should raise ValueErrors for null keys
    with pytest.raises(ValueError, match="keys column contains null values"):
        source.run()


def test_source_sync(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test source syncing flow through the API."""
    # Mock Source
    testkit = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}],
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock the routes
    matchbox_api.get(f"/resolutions/{testkit.source.name}").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Model not found", entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        )
    )
    insert_config_route = matchbox_api.post("/resolutions").mock(
        return_value=Response(
            201,
            json=ResolutionOperationStatus(
                success=True,
                name=testkit.source.name,
                operation=CRUDOperation.CREATE,
            ).model_dump(),
        )
    )

    # Mock the data upload
    upload_route = matchbox_api.post(f"/resolutions/{testkit.source.name}/data").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                name=testkit.source.name,
                stage=UploadStage.COMPLETE,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    # Index the source
    testkit.source.run()
    testkit.source.sync()

    # Verify the API calls
    resolution_call = Resolution.model_validate_json(
        insert_config_route.calls.last.request.content.decode("utf-8")
    )
    # Check key fields match (allowing for different descriptions)
    assert resolution_call.name == testkit.source.to_resolution().name
    assert resolution_call.resolution_type == ResolutionType.SOURCE
    assert resolution_call.config == testkit.source.to_resolution().config
    assert testkit.source.name in upload_route.calls.last.request.url.path
    assert b"Content-Disposition: form-data;" in upload_route.calls.last.request.content
    assert b"PAR1" in upload_route.calls.last.request.content

    # Now check client handling of server error
    matchbox_api.post(f"/resolutions/{testkit.source.name}/data").mock(
        return_value=Response(
            400,
            content=UploadStatus(
                name=testkit.source.name,
                stage=UploadStage.FAILED,
                update_timestamp=datetime.now(),
                details="Invalid schema",
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    # Verify the error is propagated
    with pytest.raises(MatchboxServerFileError):
        testkit.source.sync()

    # Mock earlier endpoint generating a name clash
    model = model_factory().model
    matchbox_api.get(f"/resolutions/{testkit.source.name}").mock(
        return_value=Response(200, json=model.to_resolution().model_dump())
    )

    with pytest.raises(ValueError, match="existing resolution"):
        testkit.source.sync()
