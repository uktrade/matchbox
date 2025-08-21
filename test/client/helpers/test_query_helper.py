from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from httpx import Response
from numpy import ndarray
from respx import MockRouter
from sqlalchemy import Engine

from matchbox import query
from matchbox.client.helpers import select
from matchbox.common.arrow import SCHEMA_QUERY
from matchbox.common.dtos import BackendResourceType, NotFoundError
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
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
    )
    testkit.write_to_location(sqlite_warehouse, set_client=True)

    # Mock API
    matchbox_api.get(f"/sources/{testkit.source_config.name}").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
    )

    # Create mock table that matches parquet schema (int32/string dictionary)
    id_array = pa.array([1, 2], type=pa.int64())
    key_array = pa.array(["0", "1"], type=pa.large_string())

    # Create dictionary with int32 indices and string values (parquet compatible)
    source_indices = pa.array([0, 0], type=pa.int32())
    source_dict = pa.array(["foo"], type=pa.string())
    source_array = pa.DictionaryArray.from_arrays(source_indices, source_dict)

    mock_table = pa.table([id_array, key_array, source_array], schema=SCHEMA_QUERY)

    # Create parquet buffer instead of Arrow IPC
    parquet_buffer = BytesIO()
    pq.write_table(mock_table, parquet_buffer)
    parquet_buffer.seek(0)

    query_route = matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=parquet_buffer.read(),
        )
    )  # The query route mock above will handle both calls automatically
    # since respx allows multiple calls to the same route

    selectors = select({"foo": ["a", "b"]}, client=sqlite_warehouse)

    # Tests with no optional params
    results = query(selectors, return_leaf_id=False)
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    # Check first call (without threshold)
    first_call_params = dict(query_route.calls[0].request.url.params)
    assert first_call_params == {
        "sources": testkit.source_config.name,
        "return_leaf_id": "False",
    }

    # Tests with optional params
    results = query(
        selectors, return_type="arrow", threshold=50, return_leaf_id=False
    ).to_pandas()
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    # Check second call (with threshold)
    second_call_params = dict(query_route.calls[1].request.url.params)
    assert second_call_params == {
        "sources": testkit.source_config.name,
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
    )
    testkit1.write_to_location(sqlite_warehouse, set_client=True)

    testkit2 = source_from_tuple(
        data_tuple=({"c": "val"}, {"c": "val"}),
        data_keys=["2", "3"],
        name="foo2",
        engine=sqlite_warehouse,
    )
    testkit2.write_to_location(sqlite_warehouse, set_client=True)

    # Mock API
    matchbox_api.get(f"/sources/{testkit1.source_config.name}").mock(
        return_value=Response(200, json=testkit1.source_config.model_dump(mode="json"))
    )

    matchbox_api.get(f"/sources/{testkit2.source_config.name}").mock(
        return_value=Response(200, json=testkit2.source_config.model_dump(mode="json"))
    )

    # Create combined mock table for both sources using parquet-compatible schema
    id_array = pa.array([1, 2, 1, 2], type=pa.int64())
    key_array = pa.array(["0", "1", "2", "3"], type=pa.large_string())

    # Create categorical source array with parquet-compatible types
    source_indices = pa.array([0, 0, 1, 1], type=pa.int32())
    source_dict = pa.array(
        [testkit1.source_config.name, testkit2.source_config.name], type=pa.string()
    )
    source_array = pa.DictionaryArray.from_arrays(source_indices, source_dict)

    combined_mock_table = pa.table(
        [id_array, key_array, source_array], schema=SCHEMA_QUERY
    )

    # Mock for multi-source query - use general matching
    combined_parquet_buffer = BytesIO()
    pq.write_table(combined_mock_table, combined_parquet_buffer)
    combined_parquet_buffer.seek(0)

    query_route = matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=combined_parquet_buffer.read(),
        )
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

    # Check that the multi-source query was made correctly
    from urllib.parse import parse_qs, urlparse

    last_request_url = str(query_route.calls.last.request.url)
    parsed_url = urlparse(last_request_url)
    url_params = parse_qs(parsed_url.query)

    assert url_params == {
        "sources": [testkit1.source_config.name, testkit2.source_config.name],
        "resolution": [DEFAULT_RESOLUTION],
        "return_leaf_id": ["False"],
    }

    # It also works with the selectors specified separately
    # But with the optimization, this also makes a single multi-source call
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
    )
    testkit1.write_to_location(sqlite_warehouse, set_client=True)

    testkit2 = source_from_tuple(
        data_tuple=({"col": "val1"}, {"col": "val2"}, {"col": "val3"}),
        data_keys=["3", "4", "5"],
        name="bar",
        engine=sqlite_warehouse,
    )
    testkit2.write_to_location(sqlite_warehouse, set_client=True)

    # Mock API
    matchbox_api.get(f"/sources/{testkit1.source_config.name}").mock(
        return_value=Response(200, json=testkit1.source_config.model_dump(mode="json"))
    )

    matchbox_api.get(f"/sources/{testkit2.source_config.name}").mock(
        return_value=Response(200, json=testkit2.source_config.model_dump(mode="json"))
    )

    # Create combined mock table for multi-source query using parquet-compatible schema
    id_array = pa.array([1, 1, 2, 2, 2, 3], type=pa.int64())
    key_array = pa.array(["0", "1", "2", "3", "3", "4"], type=pa.large_string())

    # Create categorical source array with parquet-compatible types
    source_indices = pa.array([0, 0, 0, 1, 1, 1], type=pa.int32())
    source_dict = pa.array(
        [testkit1.source_config.name, testkit2.source_config.name], type=pa.string()
    )
    source_array = pa.DictionaryArray.from_arrays(source_indices, source_dict)

    combined_mock_table = pa.table(
        [id_array, key_array, source_array], schema=SCHEMA_QUERY
    )

    # Create parquet buffer for mock response
    combined_parquet_buffer2 = BytesIO()
    pq.write_table(combined_mock_table, combined_parquet_buffer2)
    combined_parquet_buffer2.seek(0)

    matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=combined_parquet_buffer2.read(),
        )
    )

    sels = select("foo", "bar", client=sqlite_warehouse)
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
    testkit = source_factory(engine=sqlite_warehouse, name="foo")
    testkit.write_to_location(sqlite_warehouse, set_client=True)

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
