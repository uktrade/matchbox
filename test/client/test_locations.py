import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from matchbox.client.locations import RelationalDBLocation
from matchbox.common.dtos import (
    DataTypes,
    LocationType,
)
from matchbox.common.exceptions import MatchboxSourceExtractTransformError
from matchbox.common.factories.sources import (
    FeatureConfig,
    source_factory,
    source_from_tuple,
)


def test_relational_db_location_instantiation(sqlite_in_memory_warehouse: Engine):
    """Test that RelationalDBLocation can be instantiated with valid parameters."""
    location = RelationalDBLocation(name="dbname")
    assert location.config.type == LocationType.RDBMS
    assert location.config.name == "dbname"

    # Client can be set and validated
    assert location.client is None

    with pytest.raises(ValueError):
        location.set_client(12)

    assert (
        location.set_client(sqlite_in_memory_warehouse).client
        == sqlite_in_memory_warehouse
    )


@pytest.mark.parametrize(
    ["sql", "dialects"],
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
    sql: str, dialects: str, postgres_warehouse: Engine, sqlite_warehouse: Engine
):
    """Test SQL validation in validate_extract_transform."""

    if dialects == "none":
        invalid_clients = [postgres_warehouse, sqlite_warehouse]
        valid_clients = []
    if dialects == "all":
        invalid_clients = []
        valid_clients = [postgres_warehouse, sqlite_warehouse]
    if dialects == "postgres":
        invalid_clients = [sqlite_warehouse]
        valid_clients = [postgres_warehouse]
    if dialects == "sqlite":
        invalid_clients = [postgres_warehouse]
        valid_clients = [sqlite_warehouse]

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
    location = RelationalDBLocation(name="dbname").set_client(sqlite_warehouse)

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
    location = RelationalDBLocation(name="dbname").set_client(sqlite_warehouse)

    sql = f"select key, upper(company) up_company, employees from {source_testkit.name}"

    batch_size = 2

    # Execute the query
    results = list(location.execute(sql, batch_size))

    # Right number of batches and total rows
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


def test_relational_db_execute_invalid(sqlite_warehouse: Engine):
    """Test that invalid queries are handled correctly when executing."""
    location = RelationalDBLocation(name="dbname").set_client(sqlite_warehouse)

    # Invalid SQL query
    sql = "SELECT * FROM nonexistent_table"

    # Should raise an exception when executed
    with pytest.raises(OperationalError):
        list(location.execute(sql, batch_size=10))
