import pyarrow as pa
import pytest
from httpx import Response
from numpy import ndarray
from respx import MockRouter
from sqlalchemy import Engine

from matchbox import query
from matchbox.client.helpers import select
from matchbox.common.arrow import (
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
    table_to_buffer,
)
from matchbox.common.dtos import BackendResourceType, NotFoundError
from matchbox.common.exceptions import (
    MatchboxEmptyServerResponse,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.sources import source_factory, source_from_tuple
from matchbox.common.graph import DEFAULT_RESOLUTION


def test_query_no_resolution_ok_various_params(
    matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Tests that we can avoid passing resolution name, with a variety of parameters."""
    # Dummy data and source
    testkit = source_from_tuple(
        data_tuple=({"a": 1, "b": "2"}, {"a": 10, "b": "20"}),
        data_keys=["0", "1"],
        name="foo",
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock API
    matchbox_api.get(f"/sources/{testkit.source_config.name}").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
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

    selectors = select({"foo": ["a", "b"]}, client=sqlite_warehouse)

    # Tests with no optional params
    results = query(selectors, return_leaf_id=False)
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "source": testkit.source_config.name,
        "return_leaf_id": "False",
    }

    # Tests with optional params
    results = query(
        selectors, return_type="arrow", threshold=50, return_leaf_id=False
    ).to_pandas()
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "source": testkit.source_config.name,
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
    matchbox_api.get(f"/sources/{testkit1.source_config.name}").mock(
        return_value=Response(200, json=testkit1.source_config.model_dump(mode="json"))
    )

    matchbox_api.get(f"/sources/{testkit2.source_config.name}").mock(
        return_value=Response(200, json=testkit2.source_config.model_dump(mode="json"))
    )

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

    sels = select("foo", {"foo2": ["c"]}, client=sqlite_warehouse)

    # Validate results
    results = query(sels, return_leaf_id=False)
    assert len(results) == 4
    assert {
        # All fields except key automatically selected for `foo`
        "foo_a",
        "foo_b",
        # Only one column selected for `foo2`
        "foo2_c",
        # The id always comes back
        "id",
    } == set(results.columns)

    assert dict(query_route.calls[-2].request.url.params) == {
        "source": testkit1.source_config.name,
        "resolution": DEFAULT_RESOLUTION,
        "return_leaf_id": "False",
    }
    assert dict(query_route.calls[-1].request.url.params) == {
        "source": testkit2.source_config.name,
        "resolution": DEFAULT_RESOLUTION,
        "return_leaf_id": "False",
    }

    # It also works with the selectors specified separately
    query([sels[0]], [sels[1]], return_leaf_id=False)


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
    matchbox_api.get(f"/sources/{testkit1.source_config.name}").mock(
        return_value=Response(200, json=testkit1.source_config.model_dump(mode="json"))
    )

    matchbox_api.get(f"/sources/{testkit2.source_config.name}").mock(
        return_value=Response(200, json=testkit2.source_config.model_dump(mode="json"))
    )

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

    sels = select("foo", "bar", client=sqlite_warehouse)

    # Validate results
    results = query(sels, combine_type=combine_type, return_leaf_id=False)

    if combine_type == "set_agg":
        expected_len = 3
        for _, row in results.drop(columns=["id"]).iterrows():
            for cell in row.values:
                assert isinstance(cell, ndarray)
                # No duplicates
                assert len(cell) == len(set(cell))
    else:
        expected_len = 5

    assert len(results) == expected_len
    assert {
        "foo_col",
        "bar_col",
        "id",
    } == set(results.columns)


def test_query_404_resolution(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    testkit = source_factory(engine=sqlite_warehouse, name="foo").write_to_location()

    # Mock API
    matchbox_api.get(f"/sources/{testkit.source_config.name}").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
    )

    matchbox_api.get("/query").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendResourceType.RESOLUTION,
            ).model_dump(),
        )
    )

    selectors = select({"foo": ["crn", "company_name"]}, client=sqlite_warehouse)

    # Test with no optional params
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        query(selectors)


def test_query_empty_results_raises_exception(
    matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Test that query raises MatchboxEmptyServerResponse when no data is returned."""
    testkit = source_factory(engine=sqlite_warehouse, name="foo").write_to_location()

    # Mock API
    matchbox_api.get(f"/sources/{testkit.source_config.name}").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
    )

    # Mock empty results
    matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist([], schema=SCHEMA_QUERY_WITH_LEAVES)
            ).read(),
        )
    )

    selectors = select({"foo": ["crn", "company_name"]}, client=sqlite_warehouse)

    # Test that empty results raise MatchboxEmptyServerResponse
    with pytest.raises(
        MatchboxEmptyServerResponse, match="The query operation returned no data"
    ):
        query(selectors)
