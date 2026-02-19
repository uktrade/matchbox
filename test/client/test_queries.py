import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from pandas import DataFrame as PandasDataFrame
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine
from sqlglot.errors import ParseError

from matchbox.client.queries import Query, QueryCombineType, _clean
from matchbox.common.arrow import (
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
    table_to_buffer,
)
from matchbox.common.dtos import ErrorResponse, QueryConfig
from matchbox.common.exceptions import (
    MatchboxEmptyServerResponse,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.resolvers import resolver_factory
from matchbox.common.factories.sources import (
    linked_sources_factory,
    source_factory,
    source_from_tuple,
)


def _resolver_for_sources() -> tuple:
    """Create two source-backed models and a resolver for query tests."""
    foo_testkit = source_factory(name="foo")
    dag = foo_testkit.source.dag
    foo = dag.source(**foo_testkit.into_dag())

    bar_testkit = source_factory(name="bar", dag=dag)
    bar = dag.source(**bar_testkit.into_dag())

    foo_model = foo.query().deduper(
        name="foo_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    bar_model = bar.query().deduper(
        name="bar_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    resolver = resolver_factory(
        dag=foo.dag,
        name="resolver",
        inputs=[foo_model, bar_model],
        thresholds={foo_model.name: 0, bar_model.name: 0},
    ).resolver
    return foo, bar, resolver


def test_init_query() -> None:
    """Test that query is initialised correctly"""
    source, _, resolver = _resolver_for_sources()
    query = Query(
        source,
        dag=source.dag,
        resolver=resolver,
        combine_type="explode",
        cleaning={"hello": "hello"},
    )

    assert query.config == QueryConfig(
        source_resolutions=(source.name,),
        resolver_resolution=resolver.name,
        combine_type="explode",
        cleaning={"hello": "hello"},
    )


def test_query_with_resolver_uses_get(
    matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    source_testkit = source_from_tuple(
        data_tuple=({"a": 1}, {"a": 2}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()
    source = source_testkit.source.dag.source(**source_testkit.into_dag())
    source.run()
    model = source.query().deduper(
        name="foo_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    resolver = resolver_factory(
        dag=source.dag,
        name="resolver",
        inputs=[model],
        thresholds={model.name: 0},
    ).resolver
    get_route = matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [{"key": "0", "id": 1}, {"key": "1", "id": 2}],
                    schema=SCHEMA_QUERY,
                )
            ).read(),
        )
    )

    resolver.query(source).data_raw()

    request = get_route.calls.last.request
    assert dict(request.url.params) == {
        "collection": source.dag.name,
        "run_id": str(source.dag.run),
        "source": source.name,
        "resolution": resolver.name,
        "return_leaf_id": "False",
    }


def test_query_single_source(
    matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    """Tests that we can query from a single source."""
    # Dummy data and source
    testkit = source_from_tuple(
        data_tuple=({"a": 1, "b": "2"}, {"a": 10, "b": "20"}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()
    testkit.source.run()

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
    results = testkit.source.query().data()
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "collection": testkit.source.dag.name,
        "run_id": str(testkit.source.dag.run),
        "source": testkit.source.name,
        "return_leaf_id": "False",
    }

    # Tests with optional params
    results = testkit.source.query(cleaning={"foo_a": "foo_a"}).data(
        return_type="pandas"
    )

    assert isinstance(results, PandasDataFrame)
    assert len(results) == 2
    assert {"foo_a", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "collection": testkit.source.dag.name,
        "run_id": str(testkit.source.dag.run),
        "source": testkit.source.name,
        "return_leaf_id": "False",
    }


def test_query_multiple_sources(
    matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    """Tests that we can query multiple sources."""
    # Dummy data and source
    testkit1 = source_from_tuple(
        data_tuple=({"a": 1, "b": "2"}, {"a": 10, "b": "20"}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()
    testkit1.source.run()

    testkit2 = source_from_tuple(
        data_tuple=({"c": "val"}, {"c": "val"}),
        data_keys=["2", "3"],
        name="foo2",
        engine=sqla_sqlite_warehouse,
        dag=testkit1.source.dag,
    ).write_to_location()
    testkit2.source.run()

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

    foo_source = testkit1.source.dag.source(**testkit1.into_dag())
    bar_source = testkit2.source.dag.source(**testkit2.into_dag())

    model_foo = foo_source.query().deduper(
        name="foo_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    model_bar = bar_source.query().deduper(
        name="bar_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    resolver = resolver_factory(
        dag=foo_source.dag,
        name="resolver",
        inputs=[model_foo, model_bar],
        thresholds={model_foo.name: 0, model_bar.name: 0},
    ).resolver

    # Validate results (no cleaning, so all columns passed through)
    results = resolver.query(foo_source, bar_source).data()
    assert len(results) == 4
    assert {"foo_a", "foo_b", "foo2_c", "id"} == set(results.columns)

    assert dict(query_route.calls[-2].request.url.params) == {
        "collection": testkit1.source.dag.name,
        "run_id": str(testkit1.source.dag.run),
        "source": testkit1.source.name,
        "resolution": resolver.name,
        "return_leaf_id": "False",
    }
    assert dict(query_route.calls[-1].request.url.params) == {
        "collection": testkit2.source.dag.name,
        "run_id": str(testkit2.source.dag.run),
        "source": testkit2.source.name,
        "resolution": resolver.name,
        "return_leaf_id": "False",
    }


def test_query_cleaning(
    sqla_sqlite_warehouse: Engine, matchbox_api: MockRouter
) -> None:
    """Can iterate on cleaning functions."""
    # Set up mocks
    source = (
        source_from_tuple(
            data_tuple=({"col1": " a "}, {"col1": " b "}),
            data_keys=["0", "1"],
            name="foo",
            engine=sqla_sqlite_warehouse,
        )
        .write_to_location()
        .source
    )
    source.run()

    query = source.query(cleaning={"col1": f"upper({source.f('col1')})"})

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

    # Original query can be run
    cleaned1 = query.data()

    cleaned1_expected = pl.DataFrame(
        [
            {"id": 1, "col1": " A "},
            {"id": 2, "col1": " B "},
        ]
    )

    assert_frame_equal(
        cleaned1, cleaned1_expected, check_column_order=False, check_row_order=False
    )

    # After, we can iterate on the cleaning
    query.cleaning = {"col1": f"trim(upper({source.f('col1')}))"}
    cleaned2 = query.data()

    cleaned2_expected = pl.DataFrame(
        [
            {"id": 1, "col1": "A"},
            {"id": 2, "col1": "B"},
        ]
    )

    assert_frame_equal(
        cleaned2, cleaned2_expected, check_column_order=False, check_row_order=False
    )


def test_query_prefetching(
    matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    """Tests that we can iterate on cleaning without re-fetching raw data."""
    # Dummy data and source
    testkit = source_from_tuple(
        data_tuple=({"a": 1, "b": "2"}, {"a": 10, "b": "20"}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()
    testkit.source.run()

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

    query = testkit.source.query()

    raw = query.data_raw()
    assert query_route.call_count == 1

    data_prefetched = query.data(raw)
    assert query_route.call_count == 1

    assert_frame_equal(query.data(), data_prefetched, check_row_order=False)
    assert query_route.call_count == 2


@pytest.mark.parametrize(
    "combine_type",
    [QueryCombineType.SET_AGG, QueryCombineType.EXPLODE],
    ids=["set_agg", "explode"],
)
def test_query_combine_type(
    combine_type: str, matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    """Various ways of combining multiple sources are supported."""
    # Dummy data and source
    testkit1 = source_from_tuple(
        data_tuple=({"col": 20}, {"col": 40}, {"col": 60}),
        data_keys=["0", "1", "2"],
        name="foo",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()
    testkit1.source.run()

    testkit2 = source_from_tuple(
        data_tuple=({"col": "val1"}, {"col": "val2"}, {"col": "val3"}),
        data_keys=["3", "4", "5"],
        name="bar",
        engine=sqla_sqlite_warehouse,
        dag=testkit1.source.dag,
    ).write_to_location()
    testkit2.source.run()

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

    foo_source = testkit1.source.dag.source(**testkit1.into_dag())
    bar_source = testkit2.source.dag.source(**testkit2.into_dag())

    foo_model = foo_source.query().deduper(
        name="foo_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    bar_model = bar_source.query().deduper(
        name="bar_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    resolver = resolver_factory(
        dag=foo_source.dag,
        name="resolver",
        inputs=[foo_model, bar_model],
        thresholds={foo_model.name: 0, bar_model.name: 0},
    ).resolver

    # Validate results
    results = resolver.query(foo_source, bar_source, combine_type=combine_type).data()

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
    assert {"foo_col", "bar_col", "id"} == set(results.columns)


@pytest.mark.parametrize(
    "combine_type",
    [QueryCombineType.CONCAT, QueryCombineType.SET_AGG, QueryCombineType.EXPLODE],
    ids=["concat", "set_agg", "explode"],
)
def test_query_leaf_ids(
    combine_type: str, matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    """Leaf IDs can be derived as a query byproduct."""
    testkit1 = source_from_tuple(
        data_tuple=({"col": 20}, {"col": 40}, {"col": 60}),
        data_keys=["0", "1", "2"],
        name="foo",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()
    testkit1.source.run()

    testkit2 = source_from_tuple(
        data_tuple=({"col": "val1"}, {"col": "val2"}, {"col": "val3"}),
        data_keys=["3", "4", "5"],
        name="bar",
        engine=sqla_sqlite_warehouse,
        dag=testkit1.source.dag,
    ).write_to_location()
    testkit2.source.run()

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

    foo_source = testkit1.source.dag.source(**testkit1.into_dag())
    bar_source = testkit2.source.dag.source(**testkit2.into_dag())

    foo_model = foo_source.query().deduper(
        name="foo_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    bar_model = bar_source.query().deduper(
        name="bar_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    resolver = resolver_factory(
        dag=foo_source.dag,
        name="resolver",
        inputs=[foo_model, bar_model],
        thresholds={foo_model.name: 0, bar_model.name: 0},
    ).resolver

    query = resolver.query(foo_source, bar_source, combine_type=combine_type)
    data: pl.DataFrame = query.data(return_leaf_id=True)
    assert set(data.columns) == {"foo_col", "bar_col", "id"}

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


def test_query_404_resolution(
    matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    testkit = source_factory(
        engine=sqla_sqlite_warehouse, name="foo"
    ).write_to_location()

    # Mock API
    matchbox_api.get("/query").mock(
        return_value=Response(
            404,
            json=ErrorResponse(
                exception_type="MatchboxResolutionNotFoundError",
                message="Resolution 42 not found",
            ).model_dump(),
        )
    )

    # Test with no optional params
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        testkit.source.query().data()


def test_query_empty_results_raises_exception(
    matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    """Test that query raises MatchboxEmptyServerResponse when no data is returned."""
    testkit = source_factory(
        engine=sqla_sqlite_warehouse, name="foo"
    ).write_to_location()

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
        testkit.source.query().data()


def test_query_from_config() -> None:
    """Test reconstructing a Query from a QueryConfig."""
    dag = TestkitDAG().dag

    # Create test sources
    linked_testkit = linked_sources_factory(dag=dag)
    crn_testkit = linked_testkit.sources["crn"]
    dh_testkit = linked_testkit.sources["dh"]

    dag.source(**crn_testkit.into_dag())
    dag.source(**dh_testkit.into_dag())

    model_testkit = (
        dag.get_source(crn_testkit.name)
        .query()
        .linker(
            dag.get_source(dh_testkit.name).query(),
            name="link",
            model_class="DeterministicLinker",
            model_settings={"comparisons": "l.key=r.key"},
        )
    )
    dedupe_testkit = (
        dag.get_source(crn_testkit.name)
        .query()
        .deduper(
            name="dedupe",
            model_class="NaiveDeduper",
            model_settings={"unique_fields": []},
        )
    )

    resolver = resolver_factory(
        dag=dag,
        name="resolver",
        inputs=[model_testkit, dedupe_testkit],
        thresholds={model_testkit.name: 0, dedupe_testkit.name: 0},
    ).resolver

    # Create original query
    original_query = resolver.query(
        dag.get_source(crn_testkit.name),
        dag.get_source(dh_testkit.name),
        combine_type="explode",
        cleaning={"new_col": "crn_a * 2"},
    )

    # Get config and reconstruct
    config = original_query.config
    reconstructed_query = Query.from_config(config, dag=dag)

    # Verify reconstruction matches original
    assert reconstructed_query.config == original_query.config
    assert reconstructed_query.sources == original_query.sources
    assert reconstructed_query.resolver.name == original_query.resolver.name
    assert reconstructed_query.combine_type == original_query.combine_type
    assert reconstructed_query.cleaning == original_query.cleaning


def test_query_from_config_no_model() -> None:
    """Test reconstructing a Query without a resolver."""
    dag = TestkitDAG().dag

    # Create test source
    testkit = source_factory(dag=dag)

    # Add to DAG
    dag.source(**testkit.into_dag())

    # Create query without resolver
    original_query = testkit.source.query()

    # Reconstruct from config
    config = original_query.config
    reconstructed_query = Query.from_config(config, dag=dag)

    # Verify
    assert reconstructed_query.config == original_query.config
    assert reconstructed_query.resolver is None


def test_query_from_config_resolver_roundtrip() -> None:
    """Round-trip a resolver-backed query via QueryConfig."""
    dag = TestkitDAG().dag

    linked_testkit = linked_sources_factory(dag=dag)
    crn_testkit = linked_testkit.sources["crn"]
    dh_testkit = linked_testkit.sources["dh"]

    dag.source(**crn_testkit.into_dag())
    dag.source(**dh_testkit.into_dag())

    linker = (
        dag.get_source(crn_testkit.name)
        .query()
        .linker(
            dag.get_source(dh_testkit.name).query(),
            name="link",
            model_class="DeterministicLinker",
            model_settings={"comparisons": "l.key=r.key"},
        )
    )
    dedupe = (
        dag.get_source(crn_testkit.name)
        .query()
        .deduper(
            name="dedupe",
            model_class="NaiveDeduper",
            model_settings={"unique_fields": []},
        )
    )

    resolver = resolver_factory(
        dag=dag,
        name="resolver",
        inputs=[linker, dedupe],
        thresholds={linker.name: 0, dedupe.name: 0},
    ).resolver

    original_query = resolver.query(
        dag.get_source(crn_testkit.name),
        dag.get_source(dh_testkit.name),
        combine_type="explode",
        cleaning={"new_col": "crn_a * 2"},
    )

    config = original_query.config
    reconstructed_query = Query.from_config(config, dag=dag)
    assert reconstructed_query.config == original_query.config
    assert reconstructed_query.resolver is not None
    assert reconstructed_query.resolver.name == resolver.name


@pytest.mark.parametrize(
    ("cleaning_dict", "expected_columns", "expected_values"),
    [
        pytest.param(
            {"name": "foo_name"},
            ["id", "name"],
            {"name": ["A", "B", "C"]},
            id="simple_column_rename",
        ),
        pytest.param(
            {"upper_name": "upper(foo_name)"},
            ["id", "upper_name"],
            {"upper_name": ["A", "B", "C"]},
            id="simple_transformation",
        ),
        pytest.param(
            {"name": "foo_name", "is_active": "foo_status = 'active'"},
            ["id", "name", "is_active"],
            {"name": ["A", "B", "C"], "is_active": [True, False, True]},
            id="multiple_columns",
        ),
    ],
)
def test_clean_basic_functionality(
    cleaning_dict: dict[str, str],
    expected_columns: list[str],
    expected_values: dict[str, list],
) -> None:
    """Test that clean() basic functionality works."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "foo_name": ["A", "B", "C"],
            "foo_status": ["active", "inactive", "active"],
        }
    )

    result = _clean(test_data, cleaning_dict)
    assert len(result) == 3
    assert set(result.columns) == set(expected_columns)

    for column, values in expected_values.items():
        assert result[column].to_list() == values


def test_clean_none_returns_original() -> None:
    """Test that None cleaning_dict returns original data unchanged."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["John", "Jane", "Bob"],
            "age": [25, 30, 35],
            "city": ["London", "Hull", "Stratford-upon-Avon"],
        }
    )

    result = _clean(test_data, cleaning_dict=None)

    assert_frame_equal(result, test_data)


def test_clean_column_passthrough() -> None:
    """Test that id is always included plus columns referenced in cleaning_dict."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["John", "Jane", "Bob"],
            "age": [25, 30, 35],
            "city": ["London", "Hull", "Stratford-upon-Avon"],
        }
    )

    cleaning_dict = {"full_name": "name"}
    result = _clean(test_data, cleaning_dict)

    assert set(result.columns) == {"id", "full_name"}
    assert result["id"].to_list() == [1, 2, 3]
    assert result["full_name"].to_list() == ["John", "Jane", "Bob"]


def test_clean_multiple_column_references() -> None:
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
        "name": "first || ' ' || last",
        "high_earner": "salary > 55000",
    }

    result = _clean(test_data, cleaning_dict)

    assert set(result.columns) == {"id", "name", "high_earner"}
    assert result["name"].to_list() == ["John Doe", "Jane Smith", "Bob Johnson"]
    assert result["high_earner"].to_list() == [False, True, False]


def test_clean_complex_sql_expressions() -> None:
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

    result = _clean(test_data, cleaning_dict)

    assert set(result.columns) == {"id", "total", "expensive", "category_upper"}
    assert result["total"].to_list() == [21.0, 20.0, 47.25]
    assert result["expensive"].to_list() == [False, True, True]
    assert result["category_upper"].to_list() == ["A", "B", "A"]


def test_clean_empty_cleaning_dict() -> None:
    """Test with empty cleaning dict returns only id."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
        }
    )

    result = _clean(test_data, {})

    # Only id is returned when cleaning_dict is empty
    assert set(result.columns) == {"id"}
    assert result["id"].to_list() == [1, 2, 3]


def test_clean_invalid_sql() -> None:
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
        _clean(test_data, cleaning_dict)


def test_clean_leaf_id_passed_through() -> None:
    """Test that leaf_id is automatically passed through if present."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "leaf_id": ["a", "b", "c"],
            "value": [10, 20, 30],
            "status": ["active", "inactive", "pending"],
        }
    )

    cleaning_dict = {"processed_value": "value * 2"}

    result = _clean(test_data, cleaning_dict)

    assert set(result.columns) == {"id", "leaf_id", "processed_value"}
    assert result["id"].to_list() == [1, 2, 3]
    assert result["leaf_id"].to_list() == ["a", "b", "c"]
    assert result["processed_value"].to_list() == [20, 40, 60]


def test_clean_multi_source_data() -> None:
    """Test cleaning with data from multiple sources."""
    test_data = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "foo_key": ["a", "a", "b", "b"],
            "bar_key": ["x", "y", "z", "w"],
            "foo_name": ["Alice", "Alice", "Bob", "Bob"],
            "bar_value": [10, 20, 30, 40],
        }
    )

    cleaning_dict = {
        "combined": "foo_name || ': ' || bar_value",
    }

    result = _clean(test_data, cleaning_dict)

    # Id is always included, plus the cleaned column
    assert set(result.columns) == {"id", "combined"}
    assert result["combined"].to_list() == [
        "Alice: 10",
        "Alice: 20",
        "Bob: 30",
        "Bob: 40",
    ]
