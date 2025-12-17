import adbc_driver_postgresql.dbapi as adbc_postgres
import polars as pl
import pytest
from adbc_driver_manager import ProgrammingError
from polars.testing import assert_frame_equal
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from matchbox.client.locations import ClientType, RelationalDBLocation
from matchbox.common.datatypes import DataTypes
from matchbox.common.dtos import (
    LocationType,
)
from matchbox.common.exceptions import MatchboxSourceExtractTransformError
from matchbox.common.factories.sources import (
    FeatureConfig,
    source_factory,
    source_from_tuple,
)
from test.fixtures.db import WarehouseConnectionType


@pytest.mark.parametrize(
    ("warehouse", "expected_client_type"),
    [
        pytest.param("sqlite_in_memory", ClientType.SQLALCHEMY, id="sqlalchemy"),
        pytest.param("adbc_sqlite", ClientType.ADBC, id="adbc"),
    ],
    indirect=["warehouse"],
)
def test_relational_db_location_instantiation(
    warehouse: WarehouseConnectionType,
    expected_client_type: ClientType,
) -> None:
    """Test that RelationalDBLocation can be instantiated with valid parameters."""
    location = RelationalDBLocation(name="dbname")
    assert location.config.type == LocationType.RDBMS
    assert location.config.name == "dbname"

    # Client can be set and validated
    assert location.client is None

    with pytest.raises(ValueError):
        location.set_client(12)

    assert location.set_client(warehouse).client == warehouse
    assert location.client_type == expected_client_type


@pytest.mark.parametrize(
    "warehouse",
    ["sqla_sqlite", "adbc_sqlite"],
    indirect=True,
)
def test_relational_db_connect(warehouse: WarehouseConnectionType) -> None:
    """Test connecting to database."""
    location = RelationalDBLocation(name="dbname").set_client(warehouse)
    assert location.connect() is True


@pytest.mark.parametrize(
    ("sql", "dialects"),
    [
        pytest.param("SELECT * FROM test_table", "all", id="valid-select"),
        pytest.param(
            "SELECT id, name FROM test_table WHERE id > 1", "all", id="valid-where"
        ),
        pytest.param("SLECT * FROM test_table", "none", id="invalid-syntax"),
        pytest.param("", "none", id="empty-string"),
        pytest.param("ALTER TABLE test_table", "none", id="alter-sql"),
        pytest.param(
            "INSERT INTO users (name, age) VALUES ('John', '25')",
            "none",
            id="insert-sql",
        ),
        pytest.param("DROP TABLE test_table", "none", id="drop-sql"),
        pytest.param("SELECT * FROM users /* with a comment */", "all", id="comment"),
        pytest.param(
            "WITH user_cte AS (SELECT * FROM users) SELECT * FROM user_cte",
            "all",
            id="valid-with",
        ),
        pytest.param(
            (
                "WITH user_cte AS (SELECT * FROM users) "
                "INSERT INTO temp_users SELECT * FROM user_cte"
            ),
            "none",
            id="invalid-with",
        ),
        pytest.param(
            "SELECT * FROM users; DROP TABLE users;", "none", id="multiple-statements"
        ),
        pytest.param("SELECT * INTO new_table FROM users", "none", id="select-into"),
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
            "none",
            id="non-query-cte",
        ),
        pytest.param(
            """
            SELECT foo, bar FROM baz
            UNION
            SELECT foo, bar FROM qux;
            """,
            "all",
            id="valid-union",
        ),
        pytest.param(
            """
            SELECT 'ciao' ~ 'hello'
            """,
            "postgres",
            id="postgres-only",
        ),
        pytest.param("""select `name` from user""", "sqlite", id="sqlite-only"),
    ],
)
def test_relational_db_extract_transform(
    sql: str,
    dialects: str,
    sqla_postgres_warehouse: Engine,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Test SQL validation in validate_extract_transform."""

    if dialects == "none":
        invalid_clients = [sqla_postgres_warehouse, sqla_sqlite_warehouse]
        valid_clients = []
    if dialects == "all":
        invalid_clients = []
        valid_clients = [sqla_postgres_warehouse, sqla_sqlite_warehouse]
    if dialects == "postgres":
        invalid_clients = [sqla_sqlite_warehouse]
        valid_clients = [sqla_postgres_warehouse]
    if dialects == "sqlite":
        invalid_clients = [sqla_postgres_warehouse]
        valid_clients = [sqla_sqlite_warehouse]

    # Dialect-agnostic check
    if dialects == "none":
        with pytest.raises(MatchboxSourceExtractTransformError):
            RelationalDBLocation(name="dbname").validate_extract_transform(sql)
    elif dialects == "all":
        RelationalDBLocation(name="dbname").validate_extract_transform(sql)

    # Dialect-specific checks
    for client in valid_clients:
        RelationalDBLocation(name="dbname").set_client(
            client
        ).validate_extract_transform(sql)

    for client in invalid_clients:
        with pytest.raises(MatchboxSourceExtractTransformError):
            RelationalDBLocation(name="dbname").set_client(
                client
            ).validate_extract_transform(sql)


@pytest.mark.parametrize(
    "warehouse",
    ["sqla_sqlite", "adbc_sqlite"],
    indirect=True,
)
def test_relational_db_infer_types(
    warehouse: WarehouseConnectionType,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Test that types are inferred correctly from the extract transform SQL."""
    source_testkit = source_from_tuple(
        data_tuple=(
            {"foo": "10", "bar": None},
            {"foo": "foo_val", "bar": None},
            {"foo": None, "bar": 10},
        ),
        data_keys=["a", "b", "c"],
        name="source",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()

    location = RelationalDBLocation(name="dbname").set_client(warehouse)

    query = f"""
        select key as renamed_key, foo, bar from
        (select key, foo, bar from {source_testkit.name});
    """

    inferred_types = location.infer_types(query)

    assert len(inferred_types) == 3
    assert inferred_types["renamed_key"] == DataTypes.STRING
    assert inferred_types["foo"] == DataTypes.STRING
    assert inferred_types["bar"] == DataTypes.INT64


@pytest.mark.docker
@pytest.mark.parametrize("warehouse", ["adbc_postgres", "sqla_postgres"], indirect=True)
def test_relational_db_infer_complex_types_postgres(
    warehouse: WarehouseConnectionType,
    adbc_postgres_warehouse: adbc_postgres.Connection,
) -> None:
    """Test that complex Postgres types (Arrays) are correctly inferred.

    Verifies support for:
    - TEXT[] -> List(String)
    - INTEGER[] -> List(Int64)
    """
    # 1. Create a table with native Postgres Array types
    # .write_to_location() fails if we use the SQLAlchemy engine as it goes
    # via Pandas and Numpy arrays
    source_testkit = source_from_tuple(
        data_tuple=(
            {
                "tags": ["urgent", "legacy"],
                "numbers": [10, 20],
            },
            {
                "tags": ["draft"],
                "numbers": [5],
            },
        ),
        data_keys={"a", "b"},
        name="test_complex_inference",
        engine=adbc_postgres_warehouse,
    ).write_to_location()

    # Run inference
    location = RelationalDBLocation(name="postgres_db").set_client(warehouse)
    query = f"SELECT tags, numbers FROM {source_testkit.name}"

    inferred_types = location.infer_types(query)

    # TEXT[] should map to List(String)
    assert inferred_types["tags"] == DataTypes.LIST(DataTypes.STRING)

    # INTEGER[] should map to List(Int64)
    assert inferred_types["numbers"] == DataTypes.LIST(DataTypes.INT64)


@pytest.mark.parametrize(
    "warehouse",
    ["sqla_sqlite", "adbc_sqlite"],
    indirect=True,
)
def test_relational_db_execute(
    warehouse: WarehouseConnectionType,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Test executing a query and returning results using a real SQLite database."""
    features = [
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="employees", base_generator="random_int"),
    ]

    source_testkit = source_factory(
        features=features, n_true_entities=10, engine=sqla_sqlite_warehouse
    ).write_to_location()

    location = RelationalDBLocation(name="dbname").set_client(warehouse)

    sql = f"select key, upper(company) up_company, employees from {source_testkit.name}"

    batch_size = 2

    # Execute the query
    results = list(location.execute(sql, batch_size))

    # Right number of batches and total rows
    if isinstance(warehouse, Engine):
        # SQLite ADBC driver doesn't batch, can't cover this
        assert len(results[0]) == batch_size
    combined_df: pl.DataFrame = pl.concat(results)
    assert len(combined_df) == 10

    # Right fields, types and transformations
    assert set(combined_df.columns) == {"key", "up_company", "employees"}
    assert combined_df["employees"].dtype == pl.Int64
    sample_str = combined_df.select("up_company").row(0)[0]
    assert sample_str == sample_str.upper()

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


@pytest.mark.parametrize(
    "warehouse",
    ["sqla_sqlite", "adbc_sqlite"],
    indirect=True,
)
def test_relational_db_execute_invalid(warehouse: WarehouseConnectionType) -> None:
    """Test that invalid queries are handled correctly when executing."""
    location = RelationalDBLocation(name="dbname").set_client(warehouse)

    # Invalid SQL query
    sql = "SELECT * FROM nonexistent_table"

    # Should raise an exception when executed
    with pytest.raises((OperationalError, ProgrammingError)):
        list(location.execute(sql, batch_size=10))
