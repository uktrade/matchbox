import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from pandas import DataFrame as PandasDataFrame
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine
from sqlglot.errors import ParseError

from matchbox.client.queries import Query, clean
from matchbox.common.arrow import (
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
    table_to_buffer,
)
from matchbox.common.dtos import BackendResourceType, NotFoundError, QueryConfig
from matchbox.common.exceptions import (
    MatchboxEmptyServerResponse,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory, source_from_tuple


def test_init_query():
    """Test that query is initialised correctly"""
    source = source_factory().source
    model = model_factory(dag=source.dag).model
    query = Query(
        source,
        dag=source.dag,
        model=model,
        combine_type="explode",
        threshold=0.32,
        cleaning={"hello": "hello"},
    )

    assert query.config == QueryConfig(
        source_resolutions=[source.resolution_path],
        model_resolution=model.resolution_path,
        combine_type="explode",
        threshold=32,
        cleaning={"hello": "hello"},
    )


def test_query_multiple_runs(sqlite_warehouse: Engine, matchbox_api: MockRouter):
    """Can run a query multiple times and clean separately."""
    source = (
        source_from_tuple(
            data_tuple=({"col1": " a "}, {"col1": " b "}),
            data_keys=["0", "1"],
            name="foo",
            engine=sqlite_warehouse,
        )
        .write_to_location()
        .source
    )

    query = Query(
        source,
        dag=source.dag,
        cleaning={"col1": f"upper({source.f('col1')})"},
    )

    query_route = matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"key": "0", "id": 1},
                        {"key": "1", "id": 2},
                    ],
                    schema=SCHEMA_QUERY,
                )
            ).read(),
        )
    )

    assert not query.last_run
    query.run()
    assert query.last_run
    assert query_route.call_count == 1

    # Re-running does nothing
    with pytest.warns(match="already run"):
        query.run()
    assert query_route.call_count == 1

    # Re-run must be forced
    query.run(full_rerun=True)
    assert query_route.call_count == 2

    new_cleaning = {"col1": f"trim(upper({source.f('col1')}))"}

    with pytest.raises(RuntimeError, match="raw data"):
        query.clean(new_cleaning)

    cleaned1_expected = pl.DataFrame(
        [
            {"id": 1, "col1": " A ", "foo_key": "0"},
            {"id": 2, "col1": " B ", "foo_key": "1"},
        ]
    )
    cleaned1 = query.run(full_rerun=True, cache_raw=True)
    assert_frame_equal(
        cleaned1,
        cleaned1_expected,
        check_column_order=False,
        check_row_order=False,
    )
    cleaned2_expected = pl.DataFrame(
        [
            {"id": 1, "col1": "A", "foo_key": "0"},
            {"id": 2, "col1": "B", "foo_key": "1"},
        ]
    )
    cleaned2 = query.clean(cleaning=new_cleaning)
    assert_frame_equal(
        cleaned2,
        cleaned2_expected,
        check_column_order=False,
        check_row_order=False,
    )
    assert query.config.cleaning == new_cleaning


def test_query_single_source(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Tests that we can query from a single source."""
    # Dummy data and source
    testkit = source_from_tuple(
        data_tuple=({"a": 1, "b": "2"}, {"a": 10, "b": "20"}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock API
    query_route = matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"key": "0", "id": 1},
                        {"key": "1", "id": 2},
                    ],
                    schema=SCHEMA_QUERY,
                )
            ).read(),
        )
    )
    # Tests with no optional params
    results = Query(testkit.source, dag=testkit.source.dag).run()
    assert len(results) == 2
    assert {"foo_a", "foo_b", "foo_key", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "collection": testkit.source.dag.name,
        "version": testkit.source.dag.version,
        "source": testkit.source.name,
        "return_leaf_id": "False",
    }

    # Tests with optional params
    results = Query(testkit.source, threshold=0.5, dag=testkit.source.dag).run(
        return_type="pandas"
    )

    assert isinstance(results, PandasDataFrame)
    assert len(results) == 2
    assert {"foo_a", "foo_b", "foo_key", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "collection": testkit.source.dag.name,
        "version": testkit.source.dag.version,
        "source": testkit.source.name,
        "threshold": "50",
        "return_leaf_id": "False",
    }


def test_query_multiple_sources(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Tests that we can query multiple sources."""
    # Dummy data and source
    testkit1 = source_from_tuple(
        data_tuple=({"a": 1, "b": "2"}, {"a": 10, "b": "20"}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqlite_warehouse,
    ).write_to_location()

    testkit2 = source_from_tuple(
        data_tuple=({"c": "val"}, {"c": "val"}),
        data_keys=["2", "3"],
        name="foo2",
        engine=sqlite_warehouse,
        dag=testkit1.source.dag,
    ).write_to_location()

    # Mock API
    query_route = matchbox_api.get("/query").mock(
        side_effect=[
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"key": "0", "id": 1},
                            {"key": "1", "id": 2},
                        ],
                        schema=SCHEMA_QUERY,
                    )
                ).read(),
            ),
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"key": "2", "id": 1},
                            {"key": "3", "id": 2},
                        ],
                        schema=SCHEMA_QUERY,
                    )
                ).read(),
            ),
        ]
        * 2  # 2 calls to `query()` in this test, each querying server twice
    )

    model = model_factory(dag=testkit1.source.dag).model
    # Validate results
    results = Query(
        testkit1.source, testkit2.source, model=model, dag=testkit1.source.dag
    ).run()
    assert len(results) == 4
    assert {
        "foo_a",
        "foo_b",
        "foo_key",
        "foo2_c",
        "foo2_key",
        "id",
    } == set(results.columns)

    assert dict(query_route.calls[-2].request.url.params) == {
        "collection": testkit1.source.dag.name,
        "version": testkit1.source.dag.version,
        "source": testkit1.source.name,
        "resolution": model.name,
        "return_leaf_id": "False",
    }
    assert dict(query_route.calls[-1].request.url.params) == {
        "collection": testkit2.source.dag.name,
        "version": testkit2.source.dag.version,
        "source": testkit2.source.name,
        "resolution": model.name,
        "return_leaf_id": "False",
    }


def test_queries_clean(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test that cleaning in a query is applied."""
    testkit = source_from_tuple(
        data_tuple=({"val": "a", "val2": 1}, {"val": "b", "val2": 2}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock API
    matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"key": "0", "id": 1},
                        {"key": "1", "id": 2},
                    ],
                    schema=SCHEMA_QUERY,
                )
            ).read(),
        )
    )

    result = Query(
        testkit.source,
        cleaning={"new_val": f"lower({testkit.source.f('val')})"},
        dag=testkit.source.dag,
    ).run()

    assert len(result) == 2
    assert result["new_val"].to_list() == ["a", "b"]
    assert set(result.columns) == {"id", "foo_key", "new_val", "foo_val2"}


@pytest.mark.parametrize(
    "combine_type",
    ["set_agg", "explode"],
)
def test_query_combine_type(
    combine_type: str, matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Various ways of combining multiple sources are supported."""
    # Dummy data and source
    testkit1 = source_from_tuple(
        data_tuple=({"col": 20}, {"col": 40}, {"col": 60}),
        data_keys=["0", "1", "2"],
        name="foo",
        engine=sqlite_warehouse,
    ).write_to_location()

    testkit2 = source_from_tuple(
        data_tuple=({"col": "val1"}, {"col": "val2"}, {"col": "val3"}),
        data_keys=["3", "4", "5"],
        name="bar",
        engine=sqlite_warehouse,
        dag=testkit1.source.dag,
    ).write_to_location()

    # Mock API
    matchbox_api.get("/query").mock(
        side_effect=[
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"key": "0", "id": 1},
                            {"key": "1", "id": 1},
                            {"key": "2", "id": 2},
                        ],
                        schema=SCHEMA_QUERY,
                    )
                ).read(),
            ),
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            # Creating a duplicate value for the same Matchbox ID
                            {"key": "3", "id": 2},
                            {"key": "3", "id": 2},
                            {"key": "4", "id": 3},
                        ],
                        schema=SCHEMA_QUERY,
                    )
                ).read(),
            ),
        ]  # two sources to query
    )

    model = model_factory(dag=testkit1.source.dag).model

    # Validate results
    results = Query(
        testkit1.source,
        testkit2.source,
        model=model,
        combine_type=combine_type,
        dag=testkit1.source.dag,
    ).run()

    if combine_type == "set_agg":
        expected_len = 3

        # Iterate over rows
        for row in results.drop("id").iter_rows(named=True):
            for cell in row.values():
                assert isinstance(cell, list)
                # No duplicates
                assert len(cell) == len(set(cell))

    else:
        expected_len = 5

    assert len(results) == expected_len
    assert {
        "foo_col",
        "foo_key",
        "bar_col",
        "bar_key",
        "id",
    } == set(results.columns)


@pytest.mark.parametrize(
    "combine_type",
    ["concat", "set_agg", "explode"],
)
def test_query_leaf_ids(
    combine_type: str, matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Leaf IDs can be derived as a query byproduct."""
    testkit1 = source_from_tuple(
        data_tuple=({"col": 20}, {"col": 40}, {"col": 60}),
        data_keys=["0", "1", "2"],
        name="foo",
        engine=sqlite_warehouse,
    ).write_to_location()

    testkit2 = source_from_tuple(
        data_tuple=({"col": "val1"}, {"col": "val2"}, {"col": "val3"}),
        data_keys=["3", "4", "5"],
        name="bar",
        engine=sqlite_warehouse,
        dag=testkit1.source.dag,
    ).write_to_location()

    matchbox_api.get("/query").mock(
        side_effect=[
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"key": "0", "id": 12, "leaf_id": 1},
                            {"key": "1", "id": 12, "leaf_id": 2},
                            {"key": "2", "id": 345, "leaf_id": 3},
                        ],
                        schema=SCHEMA_QUERY_WITH_LEAVES,
                    )
                ).read(),
            ),
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            # Creating a duplicate value for the same Matchbox ID
                            {"key": "3", "id": 345, "leaf_id": 4},
                            {"key": "3", "id": 345, "leaf_id": 5},
                            {"key": "4", "id": 6, "leaf_id": 6},
                        ],
                        schema=SCHEMA_QUERY_WITH_LEAVES,
                    )
                ).read(),
            ),
        ]  # two sources to query
    )

    model = model_factory(dag=testkit1.source.dag).model

    query = Query(
        testkit1.source,
        testkit2.source,
        model=model,
        combine_type=combine_type,
        dag=testkit1.source.dag,
    )
    data: pl.DataFrame = query.run(return_leaf_id=True)
    assert set(data.columns) == {"foo_key", "foo_col", "bar_key", "bar_col", "id"}

    assert_frame_equal(
        pl.DataFrame(query.leaf_id),
        pl.DataFrame(
            [
                {"leaf_id": 1, "id": 12},
                {"leaf_id": 2, "id": 12},
                {"leaf_id": 3, "id": 345},
                {"leaf_id": 4, "id": 345},
                {"leaf_id": 5, "id": 345},
                {"leaf_id": 6, "id": 6},
            ]
        ),
        check_column_order=False,
        check_row_order=False,
    )


def test_query_404_resolution(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    testkit = source_factory(engine=sqlite_warehouse, name="foo").write_to_location()

    # Mock API
    matchbox_api.get("/query").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendResourceType.RESOLUTION,
            ).model_dump(),
        )
    )

    # Test with no optional params
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        Query(testkit.source, dag=testkit.source.dag).run()


def test_query_empty_results_raises_exception(
    matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Test that query raises MatchboxEmptyServerResponse when no data is returned."""
    testkit = source_factory(engine=sqlite_warehouse, name="foo").write_to_location()

    # Mock empty results
    matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist([], schema=SCHEMA_QUERY)
            ).read(),
        )
    )

    # Test that empty results raise MatchboxEmptyServerResponse
    with pytest.raises(
        MatchboxEmptyServerResponse, match="The query operation returned no data"
    ):
        Query(testkit.source, dag=testkit.source.dag).run()


@pytest.mark.parametrize(
    ("cleaning_dict", "expected_columns", "expected_values"),
    [
        pytest.param(
            {"name": "lower(foo_name)"},
            ["id", "name", "foo_status"],
            {"name": ["a", "b", "c"], "foo_status": ["active", "inactive", "active"]},
            id="basic_cleaning_with_passthrough",
        ),
        pytest.param(
            {"new_status": "foo_status", "lower_name": "lower(foo_name)"},
            ["id", "new_status", "lower_name"],
            {
                "new_status": ["active", "inactive", "active"],
                "lower_name": ["a", "b", "c"],
            },
            id="column_dropping_and_renaming",
        ),
    ],
)
def test_clean_basic_functionality(
    cleaning_dict: dict[str, str],
    expected_columns: list[str],
    expected_values: dict[str, list],
):
    """Test that clean() basic functionality works."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "foo_name": ["A", "B", "C"],
            "foo_status": ["active", "inactive", "active"],
        }
    )

    result = clean(test_data, cleaning_dict)
    assert len(result) == 3
    assert set(result.columns) == set(expected_columns)

    for column, values in expected_values.items():
        assert result[column].to_list() == values


def test_clean_none_returns_original():
    """Test that None cleaning_dict returns original data."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "foo_name": ["A", "B", "C"],
            "foo_status": ["active", "inactive", "active"],
        }
    )

    result = clean(test_data, None)
    assert set(result.columns) == {"id", "foo_name", "foo_status"}

    result_sorted = result.select(sorted(result.columns))
    test_data_sorted = test_data.select(sorted(test_data.columns))
    assert result_sorted.equals(test_data_sorted)


@pytest.mark.parametrize(
    ("extra_columns", "expected_columns"),
    [
        pytest.param(
            {
                "leaf_id": ["a", "b", "c"],
                "key": ["x", "y", "z"],
                "status": ["active", "inactive", "pending"],
            },
            ["id", "leaf_id", "key", "processed_value", "status"],
            id="both_special_columns",
        ),
        pytest.param(
            {"leaf_id": ["a", "b", "c"]},
            ["id", "leaf_id", "processed_value"],
            id="only_leaf_id",
        ),
        pytest.param(
            {"key": ["x", "y", "z"]}, ["id", "key", "processed_value"], id="only_key"
        ),
    ],
)
def test_clean_special_columns_handling(
    extra_columns: dict[str, list], expected_columns: list[str]
):
    """Test that leaf_id and key columns are automatically passed through."""
    base_data = {
        "id": [1, 2, 3],
        "value": [10, 20, 30],
    }

    test_data = pl.DataFrame({**base_data, **extra_columns})
    cleaning_dict = {"processed_value": "value * 2"}
    result = clean(test_data, cleaning_dict)

    assert set(result.columns) == set(expected_columns)
    assert result["processed_value"].to_list() == [20, 40, 60]

    # Check passthrough columns if they exist
    if "status" in extra_columns:
        assert result["status"].to_list() == ["active", "inactive", "pending"]


def test_clean_multiple_column_references():
    """Test expressions that reference multiple columns."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "first": ["John", "Jane", "Bob"],
            "last": ["Doe", "Smith", "Johnson"],
            "salary": [50000, 60000, 55000],
        }
    )

    cleaning_dict = {
        "name": "first || ' ' || last",  # References both 'first' and 'last'
        "high_earner": "salary > 55000",
    }

    result = clean(test_data, cleaning_dict)

    # first, last, and salary are dropped (used in expressions)
    assert set(result.columns) == {"id", "name", "high_earner"}
    assert result["name"].to_list() == ["John Doe", "Jane Smith", "Bob Johnson"]
    assert result["high_earner"].to_list() == [False, True, False]


def test_clean_complex_sql_expressions():
    """Test more complex SQL expressions."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "price": [10.5, 20.0, 15.75],
            "quantity": [2, 1, 3],
            "category": ["A", "B", "A"],
        }
    )

    cleaning_dict = {
        "total": "price * quantity",
        "expensive": "price > 15.0",
        "category_upper": "upper(category)",
    }

    result = clean(test_data, cleaning_dict)

    # Use set comparison for columns
    assert set(result.columns) == {"id", "total", "expensive", "category_upper"}
    assert result["total"].to_list() == [21.0, 20.0, 47.25]
    assert result["expensive"].to_list() == [False, True, True]
    assert result["category_upper"].to_list() == ["A", "B", "A"]


def test_clean_empty_cleaning_dict():
    """Test with empty cleaning dict."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
        }
    )

    result = clean(test_data, {})

    # Only id is selected, plus all unused columns (name, value)
    assert set(result.columns) == {"id", "value", "name"}

    result_sorted = result.select(sorted(result.columns))
    test_data_sorted = test_data.select(sorted(test_data.columns))
    assert result_sorted.equals(test_data_sorted)


def test_clean_invalid_sql():
    """Test that invalid SQL raises ParseError."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        }
    )

    cleaning_dict = {
        "invalid": "foo bar baz",  # Invalid SQL
    }

    with pytest.raises(ParseError):
        clean(test_data, cleaning_dict)


def test_clean_column_passthrough():
    """Test that unused columns are passed through unchanged."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["John", "Jane", "Bob"],
            "age": [25, 30, 35],
            "city": ["London", "Hull", "Stratford-upon-Avon"],
        }
    )

    cleaning_dict = {
        "full_name": "name"  # Only references 'name' column
    }

    result = clean(test_data, cleaning_dict)

    # name is dropped because it was used in cleaning_dict
    # age and city are passed through unchanged
    assert set(result.columns) == {"id", "full_name", "city", "age"}
    assert result["full_name"].to_list() == ["John", "Jane", "Bob"]
    assert result["age"].to_list() == [25, 30, 35]
    assert result["city"].to_list() == ["London", "Hull", "Stratford-upon-Avon"]
