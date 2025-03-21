from typing import Callable
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
from httpx import Response
from numpy import ndarray
from pandas import DataFrame
from respx import MockRouter
from sqlalchemy import Engine

from matchbox import index, match, process, query
from matchbox.client.clean import company_name, company_number
from matchbox.client.helpers import cleaner, cleaners, comparison, select
from matchbox.client.helpers.selector import Match, Selector
from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.dtos import (
    BackendRetrievableType,
    BackendUploadType,
    NotFoundError,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import (
    linked_sources_factory,
    source_factory,
)
from matchbox.common.graph import DEFAULT_RESOLUTION
from matchbox.common.hash import hash_to_base64
from matchbox.common.sources import Source, SourceAddress


def test_cleaners():
    cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
    cleaner_number = cleaner(
        function=company_number, arguments={"column": "company_number"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    assert cleaner_name_number is not None


def test_process():
    crn = source_factory().query
    # Duplicate column
    crn = crn.append_column("company_name_again", crn.column("company_name"))

    cleaner_name = cleaner(
        function=company_name,
        arguments={"column": "company_name"},
    )
    # I can add the same function twice
    cleaner_name_again = cleaner(
        function=company_name,
        arguments={"column": "company_name_again"},
    )
    cleaner_number = cleaner(
        function=company_number,
        arguments={"column": "crn"},
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_name_again, cleaner_number)

    df_name_cleaned = process(data=crn, pipeline=cleaner_name_number)

    assert isinstance(df_name_cleaned, DataFrame)
    assert df_name_cleaned.shape[0] == 10


def test_comparisons():
    comparison_name_id = comparison(
        sql_condition=(
            "l.company_name = r.company_name and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None


def test_select_default_engine(
    matchbox_api: MockRouter,
    sqlite_warehouse: Engine,
    env_setter: Callable[[str, str], None],
):
    """We can select without explicit engine if default is set."""
    default_engine = sqlite_warehouse.url.render_as_string(hide_password=False)
    env_setter("MB__CLIENT__DEFAULT_WAREHOUSE", default_engine)

    # Set up mocks and test data
    testkit = source_factory(full_name="bar", engine=sqlite_warehouse)
    source = testkit.source

    matchbox_api.get(
        f"/sources/{hash_to_base64(source.address.warehouse_hash)}/bar"
    ).mock(return_value=Response(200, content=source.model_dump_json()))

    testkit.to_warehouse(engine=sqlite_warehouse)

    # Select sources
    selection = select("bar")

    # Check they contain what we expect
    assert selection[0].source.model_dump() == source.model_dump()
    # Check the engine is set by the selector
    assert selection[0].source.engine.url == sqlite_warehouse.url


def test_select_missing_engine():
    """We must pass an engine if a default is not set"""
    with pytest.raises(ValueError, match="engine"):
        select("test.bar")


def test_select_mixed_style(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """We can select specific columns from some of the sources"""
    linked = linked_sources_factory(engine=sqlite_warehouse)

    source1 = linked.sources["crn"].source
    source2 = linked.sources["cdms"].source

    matchbox_api.get(
        f"/sources/{hash_to_base64(source1.address.warehouse_hash)}/crn"
    ).mock(return_value=Response(200, content=source1.model_dump_json()))
    matchbox_api.get(
        f"/sources/{hash_to_base64(source2.address.warehouse_hash)}/cdms"
    ).mock(return_value=Response(200, content=source2.model_dump_json()))

    linked.sources["crn"].to_warehouse(engine=sqlite_warehouse)
    linked.sources["cdms"].to_warehouse(engine=sqlite_warehouse)

    selection = select({"crn": ["company_name"]}, "cdms", engine=sqlite_warehouse)

    assert selection[0].fields == ["company_name"]
    assert not selection[1].fields
    assert selection[0].source.model_dump() == source1.model_dump()
    assert selection[1].source.model_dump() == source2.model_dump()
    assert selection[0].source.engine == sqlite_warehouse
    assert selection[1].source.engine == sqlite_warehouse


def test_select_non_indexed_columns(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Selecting columns not declared to backend generates warning."""
    source_testkit = source_factory(full_name="foo", engine=sqlite_warehouse)

    source = source_testkit.source
    source = source.model_copy(update={"columns": source.columns[:1]})

    matchbox_api.get(
        f"/sources/{hash_to_base64(source.address.warehouse_hash)}/foo"
    ).mock(return_value=Response(200, content=source.model_dump_json()))

    source_testkit.to_warehouse(engine=sqlite_warehouse)

    with pytest.warns(Warning):
        select({"foo": ["company_name", "crn"]}, engine=sqlite_warehouse)


def test_select_missing_columns(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Selecting columns not in the warehouse errors."""
    source_testkit = source_factory(full_name="foo", engine=sqlite_warehouse)

    source = source_testkit.source

    matchbox_api.get(
        f"/sources/{hash_to_base64(source.address.warehouse_hash)}/foo"
    ).mock(return_value=Response(200, content=source.model_dump_json()))

    source_testkit.to_warehouse(engine=sqlite_warehouse)

    with pytest.raises(ValueError):
        select(
            {"foo": ["company_name", "non_existent_column"]}, engine=sqlite_warehouse
        )


@patch.object(Source, "to_arrow")
def test_query_no_resolution_ok_various_params(
    to_arrow: Mock, matchbox_api: MockRouter
):
    """Tests that we can avoid passing resolution name, with a variety of parameters."""
    # Mock API
    query_route = matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"source_pk": "0", "id": 1},
                        {"source_pk": "1", "id": 2},
                    ],
                    schema=SCHEMA_MB_IDS,
                )
            ).read(),
        )
    )

    # Mock `Source.to_arrow`
    to_arrow.return_value = pa.Table.from_arrays(
        [
            pa.array(["0", "1"], type=pa.large_string()),
            pa.array([1, 10], type=pa.int64()),
            pa.array(["2", "20"], type=pa.string()),
        ],
        names=["foo_pk", "foo_a", "foo_b"],
    )

    # Well-formed selector for these mocks
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    # Tests with no optional params
    results = query(sels)
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "full_name": sels[0].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
    }
    to_arrow.assert_called_once()
    assert set(to_arrow.call_args.kwargs["fields"]) == {"a", "b"}
    assert set(to_arrow.call_args.kwargs["pks"]) == {"0", "1"}

    # Tests with optional params
    results = query(sels, return_type="arrow", threshold=50, limit=2).to_pandas()
    assert len(results) == 2
    assert {"foo_a", "foo_b", "id"} == set(results.columns)

    assert dict(query_route.calls.last.request.url.params) == {
        "full_name": sels[0].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
        "threshold": "50",
        "limit": "2",
    }


@patch.object(Source, "to_arrow")
def test_query_multiple_sources_with_limits(to_arrow: Mock, matchbox_api: MockRouter):
    """Tests that we can query multiple sources and distribute the limit among them."""
    # Mock API
    query_route = matchbox_api.get("/query").mock(
        side_effect=[
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"source_pk": "0", "id": 1},
                            {"source_pk": "1", "id": 2},
                        ],
                        schema=SCHEMA_MB_IDS,
                    )
                ).read(),
            ),
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"source_pk": "2", "id": 1},
                            {"source_pk": "3", "id": 2},
                        ],
                        schema=SCHEMA_MB_IDS,
                    )
                ).read(),
            ),
        ]
        * 2  # 2 calls to `query()` in this test, each querying server twice
    )

    # Mock `Source.to_arrow`
    to_arrow.side_effect = [
        pa.Table.from_arrays(
            [
                pa.array(["0", "1"], type=pa.large_string()),
                pa.array([1, 10], type=pa.int64()),
                pa.array(["2", "20"], type=pa.string()),
            ],
            names=["foo_pk", "foo_a", "foo_b"],
        ),
        pa.Table.from_arrays(
            [
                pa.array(["2", "3"], type=pa.large_string()),
                pa.array(["val", "val"], type=pa.string()),
            ],
            names=["foo2_pk", "foo2_c"],
        ),
    ] * 2  # 2 calls to `query()` in this test, each dealing with 2 sources

    # Well-formed select from these mocks
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
        ),
        Selector(
            source=Source(
                address=SourceAddress(full_name="foo2", warehouse_hash=b"bar2"),
                db_pk="pk",
            ),
            fields=["c"],
        ),
    ]

    # Validate results
    results = query(sels, limit=7)
    assert len(results) == 4
    assert {
        # All columns automatically selected for `foo`
        "foo_pk",
        "foo_a",
        "foo_b",
        # Only one column selected for `foo2`
        "foo2_c",
        # The id always comes back
        "id",
    } == set(results.columns)

    assert dict(query_route.calls[-2].request.url.params) == {
        "full_name": sels[0].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[0].source.address.warehouse_hash),
        "resolution_name": DEFAULT_RESOLUTION,
        "limit": "4",
    }
    assert dict(query_route.calls[-1].request.url.params) == {
        "full_name": sels[1].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[1].source.address.warehouse_hash),
        "resolution_name": DEFAULT_RESOLUTION,
        "limit": "3",
    }

    # It also works with the selectors specified separately
    query([sels[0]], [sels[1]], limit=7)


@pytest.mark.parametrize(
    "combine_type",
    ["set_agg", "explode"],
)
@patch.object(Source, "to_arrow")
def test_query_combine_type(
    to_arrow: Mock, combine_type: str, matchbox_api: MockRouter
):
    """Various ways of combining multiple sources are supported."""
    # Mock API
    matchbox_api.get("/query").mock(
        side_effect=[
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            {"source_pk": "0", "id": 1},
                            {"source_pk": "1", "id": 1},
                            {"source_pk": "2", "id": 2},
                        ],
                        schema=SCHEMA_MB_IDS,
                    )
                ).read(),
            ),
            Response(
                200,
                content=table_to_buffer(
                    pa.Table.from_pylist(
                        [
                            # Creating a duplicate value for the same Matchbox ID
                            {"source_pk": "3", "id": 2},
                            {"source_pk": "3", "id": 2},
                            {"source_pk": "4", "id": 3},
                        ],
                        schema=SCHEMA_MB_IDS,
                    )
                ).read(),
            ),
        ]  # two sources to query
    )

    # Mock `Source.to_arrow`
    to_arrow.side_effect = [
        pa.Table.from_arrays(
            [
                pa.array(["0", "1", "2"], type=pa.large_string()),
                pa.array([20, 40, 60], type=pa.int64()),
            ],
            names=["foo_pk", "foo_col"],
        ),
        pa.Table.from_arrays(
            [
                pa.array(["3", "4", "5"], type=pa.large_string()),
                pa.array(["val1", "val2", "val3"], type=pa.large_string()),
            ],
            names=["bar_pk", "bar_col"],
        ),
    ]  # two sources to query

    # Well-formed select from these mocks
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(full_name="foo", warehouse_hash=b"wh"),
                db_pk="pk",
            ),
        ),
        Selector(
            source=Source(
                address=SourceAddress(full_name="bar", warehouse_hash=b"wh"),
                db_pk="pk",
            ),
        ),
    ]

    # Validate results
    results = query(sels, combine_type=combine_type)

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
        "foo_pk",
        "foo_col",
        "bar_pk",
        "bar_col",
        "id",
    } == set(results.columns)


def test_query_404_resolution(matchbox_api: MockRouter):
    # Mock API
    matchbox_api.get("/query").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendRetrievableType.RESOLUTION,
            ).model_dump(),
        )
    )

    # Well-formed selector for these mocks
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    # Test with no optional params
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        query(sels)


def test_query_404_source(matchbox_api: MockRouter):
    # Mock API
    matchbox_api.get("/query").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Source 42 not found",
                entity=BackendRetrievableType.SOURCE,
            ).model_dump(),
        )
    )

    # Well-formed selector for these mocks
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    # Test with no optional params
    with pytest.raises(MatchboxSourceNotFoundError, match="42"):
        query(sels)


def test_query_with_batches(matchbox_api: MockRouter):
    """Tests that query correctly passes batching options to to_arrow."""
    # Mock API
    _ = matchbox_api.get("/query").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"source_pk": "0", "id": 1},
                        {"source_pk": "1", "id": 2},
                    ],
                    schema=SCHEMA_MB_IDS,
                )
            ).read(),
        )
    )

    # Create mock source with a mocked to_arrow method
    source_testkit = source_factory(
        features=[
            {"name": "a", "base_generator": "random_int"},
            {"name": "b", "base_generator": "random_int"},
        ],
        full_name="foo",
    )

    schema = pa.schema(
        [
            pa.field("foo_pk", pa.large_string()),
            pa.field("foo_a", pa.int64()),
            pa.field("foo_b", pa.string()),
        ]
    )

    mock_batch1 = pa.Table.from_pylist(
        [{"foo_pk": "0", "foo_a": 1, "foo_b": "2"}], schema=schema
    )

    mock_batch2 = pa.Table.from_pylist(
        [{"foo_pk": "1", "foo_a": 10, "foo_b": "20"}], schema=schema
    )
    mock_source = source_testkit.mock
    mock_source.to_arrow.return_value = iter([mock_batch1, mock_batch2])
    mock_source.format_column.return_value = "foo_pk"

    # Well-formed selector for these mocks
    sels = [
        Selector(
            source=mock_source,
            fields=["a", "b"],
        )
    ]

    # Test with return_batches=True
    batch_iterator = query(
        sels, return_batches=True, batch_size=1000, return_type="arrow"
    )

    # Check first batch before verifying the call
    first_batch = next(batch_iterator)
    assert isinstance(first_batch, pa.Table)

    # Verify to_arrow was called with iter_batches=True
    mock_source.to_arrow.assert_called_once()
    assert mock_source.to_arrow.call_args.kwargs["iter_batches"] is True
    assert mock_source.to_arrow.call_args.kwargs["batch_size"] == 1000

    # Verify we can get the remaining batch
    remaining_batches = list(batch_iterator)
    assert len(remaining_batches) == 1


@patch("matchbox.client.helpers.index.Source")
def test_index_success(
    MockSource: Mock, matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Test successful indexing flow through the API."""
    # Mock Source
    source = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}],
        engine=sqlite_warehouse,
    )
    mock_source_instance = source.mock
    MockSource.return_value = mock_source_instance

    # Mock the initial source metadata upload
    source_route = matchbox_api.post("/sources").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )
    )

    # Mock the data upload
    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id", status="complete", entity=BackendUploadType.INDEX
            ).model_dump(),
        )
    )

    # Call the index function
    index(
        full_name=source.source.address.full_name,
        db_pk=source.source.db_pk,
        engine=sqlite_warehouse,
    )

    # Verify the API calls
    source_call = Source.model_validate_json(
        source_route.calls.last.request.content.decode("utf-8")
    )
    assert source_call == source.source
    assert "test-upload-id" in upload_route.calls.last.request.url.path
    assert b"Content-Disposition: form-data;" in upload_route.calls.last.request.content
    assert b"PAR1" in upload_route.calls.last.request.content


@patch("matchbox.client.helpers.index.Source")
@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(["name", "age"], id="string_columns"),
        pytest.param(
            [
                {"name": "name", "alias": "person_name", "type": "TEXT"},
                {"name": "age", "alias": "person_age", "type": "BIGINT"},
            ],
            id="dict_columns",
        ),
        pytest.param(None, id="default_columns"),
    ],
)
def test_index_with_columns(
    MockSource: Mock,
    matchbox_api: MockRouter,
    columns: list[str] | list[dict[str, str]],
    sqlite_warehouse: Engine,
):
    """Test indexing with different column definition formats."""
    # Create source testkit and mock
    source = source_factory(
        features=[
            {"name": "name", "base_generator": "name"},
            {"name": "age", "base_generator": "random_int"},
        ],
        engine=sqlite_warehouse,
    )
    mock_source_instance = source.mock
    MockSource.return_value = mock_source_instance

    # Mock the API endpoints
    source_route = matchbox_api.post("/sources").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )
    )

    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id", status="complete", entity=BackendUploadType.INDEX
            ).model_dump(),
        )
    )

    # Call index with column definition
    index(
        full_name=source.source.address.full_name,
        db_pk=source.source.db_pk,
        engine=sqlite_warehouse,
        columns=columns,
    )

    # Verify API calls and source creation
    assert source_route.called
    source_call = Source.model_validate_json(
        source_route.calls.last.request.content.decode("utf-8")
    )
    assert source_call == source.source
    assert "test-upload-id" in upload_route.calls.last.request.url.path
    assert b"Content-Disposition: form-data;" in upload_route.calls.last.request.content
    assert b"PAR1" in upload_route.calls.last.request.content
    mock_source_instance.set_engine.assert_called_once_with(sqlite_warehouse)
    if columns:
        mock_source_instance.default_columns.assert_not_called()
    else:
        mock_source_instance.default_columns.assert_called_once()


@patch("matchbox.client.helpers.index.Source")
def test_index_upload_failure(
    MockSource: Mock, matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Test handling of upload failures."""
    # Mock Source
    source = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}],
        engine=sqlite_warehouse,
    )
    mock_source_instance = source.mock
    MockSource.return_value = mock_source_instance

    # Mock successful source creation
    source_route = matchbox_api.post("/sources").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )
    )

    # Mock failed upload
    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            400,
            json=UploadStatus(
                id="test-upload-id",
                status="failed",
                details="Invalid schema",
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )
    )

    # Verify the error is propagated
    with pytest.raises(MatchboxServerFileError):
        index(
            full_name=source.source.address.full_name,
            db_pk=source.source.db_pk,
            engine=sqlite_warehouse,
        )

    # Verify API calls
    source_call = Source.model_validate_json(
        source_route.calls.last.request.content.decode("utf-8")
    )
    assert source_call == source.source
    assert "test-upload-id" in upload_route.calls.last.request.url.path
    assert b"Content-Disposition: form-data;" in upload_route.calls.last.request.content
    assert b"PAR1" in upload_route.calls.last.request.content


@patch("matchbox.client.helpers.index.Source")
def test_index_with_batch_size(
    MockSource: Mock, matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Test that batch_size is passed correctly to hash_data when indexing."""
    # Mock Source
    source = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}],
        engine=sqlite_warehouse,
    )
    mock_source_instance = source.mock
    # Mock hash_data to capture the batch_size parameter
    mock_source_instance.hash_data.return_value = pa.table(
        {
            "source_pk": [["1", "2"]],
            "hash": pa.array([b"hash1"], type=pa.binary()),
        }
    )
    MockSource.return_value = mock_source_instance

    # Mock the API endpoints
    matchbox_api.post("/sources").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )
    )

    matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id", status="complete", entity=BackendUploadType.INDEX
            ).model_dump(),
        )
    )

    # Call index with batch_size
    index(
        full_name=source.source.address.full_name,
        db_pk=source.source.db_pk,
        engine=sqlite_warehouse,
        batch_size=1000,
    )

    # Verify batch_size was passed to hash_data
    mock_source_instance.hash_data.assert_called_once()
    assert mock_source_instance.hash_data.call_args.kwargs["iter_batches"] is True
    assert mock_source_instance.hash_data.call_args.kwargs["batch_size"] == 1000


def test_match_ok(matchbox_api: MockRouter):
    """The client can perform the right call for matching."""
    # Set up mocks
    mock_match1 = Match(
        cluster=1,
        source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target", warehouse_hash=b"bar"),
        target_id={"b"},
    )
    mock_match2 = Match(
        cluster=1,
        source=SourceAddress(full_name="test.source", warehouse_hash=b"bar"),
        source_id={"a"},
        target=SourceAddress(full_name="test.target2", warehouse_hash=b"bar"),
        target_id={"b"},
    )
    # The standard JSON serialiser does not handle Pydantic objects
    serialised_matches = (
        f"[{mock_match1.model_dump_json()}, {mock_match2.model_dump_json()}]"
    )

    match_route = matchbox_api.get("/match").mock(
        return_value=Response(200, content=serialised_matches)
    )

    # Use match function
    source = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.source",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]
    target1 = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target1",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    target2 = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target2",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    res = match(
        target1,
        target2,
        source=source,
        source_pk="pk1",
        resolution_name="foo",
    )

    # Verify results
    assert len(res) == 2
    assert isinstance(res[0], Match)
    param_set = sorted(match_route.calls.last.request.url.params.multi_items())
    assert param_set == sorted(
        [
            ("target_full_names", "test.target1"),
            ("target_full_names", "test.target2"),
            ("target_warehouse_hashes_b64", hash_to_base64(b"bar")),
            ("target_warehouse_hashes_b64", hash_to_base64(b"bar")),
            ("source_full_name", "test.source"),
            ("source_warehouse_hash_b64", hash_to_base64(b"bar")),
            ("source_pk", "pk1"),
            ("resolution_name", "foo"),
        ]
    )


def test_match_404_resolution(matchbox_api: MockRouter):
    """The client can handle a resolution not found error."""
    # Set up mocks
    matchbox_api.get("/match").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendRetrievableType.RESOLUTION,
            ).model_dump(),
        )
    )

    # Use match function
    source = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.source",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]
    target = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target1",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        match(
            target,
            source=source,
            source_pk="pk1",
            resolution_name="foo",
        )


def test_match_404_source(matchbox_api: MockRouter):
    """The client can handle a source not found error."""
    # Set up mocks
    matchbox_api.get("/match").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Source 42 not found",
                entity=BackendRetrievableType.SOURCE,
            ).model_dump(),
        )
    )

    # Use match function
    source = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.source",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]
    target = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="test.target1",
                    warehouse_hash=b"bar",
                ),
                db_pk="pk",
            ),
            fields=["a", "b"],
        )
    ]

    with pytest.raises(MatchboxSourceNotFoundError, match="42"):
        match(
            target,
            source=source,
            source_pk="pk1",
            resolution_name="foo",
        )
