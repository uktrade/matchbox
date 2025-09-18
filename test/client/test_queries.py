import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from pandas import DataFrame as PandasDataFrame
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine

from matchbox.client.queries import Query
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
    model = model_factory().model
    query = Query(
        source,
        model=model,
        combine_type="explode",
        threshold=0.32,
        cleaning={"hello": "hello"},
    )

    assert query.config == QueryConfig(
        source_resolutions=[source.name],
        model_resolution=model.name,
        combine_type="explode",
        threshold=32,
        cleaning={"hello": "hello"},
    )


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
    results = Query(testkit.source).run()
    assert len(results) == 2
    assert {"foo_a", "foo_b", "foo_key", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "source": testkit.source.name,
        "return_leaf_id": "False",
    }

    # Tests with optional params
    results = Query(testkit.source, threshold=0.5).run(return_type="pandas")

    assert isinstance(results, PandasDataFrame)
    assert len(results) == 2
    assert {"foo_a", "foo_b", "foo_key", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
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

    model = model_factory().model
    # Validate results
    results = Query(testkit1.source, testkit2.source, model=model).run()
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
        "source": testkit1.source.name,
        "resolution": model.name,
        "return_leaf_id": "False",
    }
    assert dict(query_route.calls[-1].request.url.params) == {
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
        testkit.source, cleaning={"new_val": f"lower({testkit.source.f('val')})"}
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

    model = model_factory().model

    # Validate results
    results = Query(
        testkit1.source, testkit2.source, model=model, combine_type=combine_type
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

    model = model_factory().model

    query = Query(
        testkit1.source, testkit2.source, model=model, combine_type=combine_type
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
        Query(testkit.source).run()


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
        Query(testkit.source).run()
