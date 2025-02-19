import logging
from os import environ
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
from dotenv import find_dotenv, load_dotenv
from httpx import Response
from pandas import DataFrame
from respx import MockRouter
from sqlalchemy import Engine, create_engine

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
from matchbox.common.factories.sources import source_factory
from matchbox.common.hash import hash_to_base64
from matchbox.common.sources import Source, SourceAddress, SourceColumn

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


def test_cleaners():
    cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
    cleaner_number = cleaner(
        function=company_number, arguments={"column": "company_number"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    assert cleaner_name_number is not None


def test_process(warehouse_data: list[Source]):
    crn = warehouse_data[0].to_arrow()

    cleaner_name = cleaner(
        function=company_name,
        arguments={"column": "test_crn_company_name"},
    )
    cleaner_number = cleaner(
        function=company_number,
        arguments={"column": "test_crn_crn"},
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    df_name_cleaned = process(data=crn, pipeline=cleaner_name_number)

    assert isinstance(df_name_cleaned, DataFrame)
    assert df_name_cleaned.shape[0] == 3000


def test_comparisons():
    comparison_name_id = comparison(
        sql_condition=(
            "l.company_name = r.company_name and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None


def test_select_default_engine(matchbox_api: MockRouter, warehouse_engine: Engine):
    """We can select without explicit engine if default is set"""
    # Set up default engine
    environ["MB__CLIENT__DEFAULT_WAREHOUSE"] = warehouse_engine.url.render_as_string(
        hide_password=False
    )

    # Set up mocks and test data
    source = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.bar"),
        db_pk="pk",
    )

    matchbox_api.get(
        f"/sources/{hash_to_base64(source.address.warehouse_hash)}/test.bar"
    ).mock(return_value=Response(200, content=source.model_dump_json()))

    df = DataFrame([{"pk": 0, "a": 1, "b": "2"}, {"pk": 1, "a": 10, "b": "20"}])
    with warehouse_engine.connect() as conn:
        df.to_sql(
            name="bar",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )

    # Select sources
    selection = select("test.bar")

    # Check they contain what we expect
    assert selection[0].source.model_dump() == source.model_dump()
    # Check the engine is set by the selector
    assert selection[0].source.engine.url == warehouse_engine.url


def test_select_mixed_style(matchbox_api: MockRouter, warehouse_engine: Engine):
    """We can select select specific columns from some of the sources"""
    # Set up mocks and test data
    source1 = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.foo"),
        db_pk="pk",
        columns=[SourceColumn(name="a", type="BIGINT")],
    )
    source2 = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.bar"),
        db_pk="pk",
    )

    matchbox_api.get(
        f"/sources/{hash_to_base64(source1.address.warehouse_hash)}/test.foo"
    ).mock(return_value=Response(200, content=source1.model_dump_json()))
    matchbox_api.get(
        f"/sources/{hash_to_base64(source2.address.warehouse_hash)}/test.bar"
    ).mock(return_value=Response(200, content=source2.model_dump_json()))

    df = DataFrame([{"pk": 0, "a": 1, "b": "2"}, {"pk": 1, "a": 10, "b": "20"}])
    with warehouse_engine.connect() as conn:
        df.to_sql(
            name="foo",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )

        df.to_sql(
            name="bar",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )

    # Select sources
    selection = select({"test.foo": ["a"]}, "test.bar", engine=warehouse_engine)

    # Check they contain what we expect
    assert selection[0].fields == ["a"]
    assert not selection[1].fields
    assert selection[0].source.model_dump() == source1.model_dump()
    assert selection[1].source.model_dump() == source2.model_dump()
    # Check the engine is set by the selector
    assert selection[0].source.engine == warehouse_engine
    assert selection[1].source.engine == warehouse_engine


def test_select_non_indexed_columns(matchbox_api: MockRouter, warehouse_engine: Engine):
    """Selecting columns not declared to backend generates warning."""
    source = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.foo"),
        db_pk="pk",
    )
    matchbox_api.get(
        f"/sources/{hash_to_base64(source.address.warehouse_hash)}/test.foo"
    ).mock(return_value=Response(200, content=source.model_dump_json()))

    df = DataFrame([{"pk": 0, "a": 1, "b": "2"}, {"pk": 1, "a": 10, "b": "20"}])
    with warehouse_engine.connect() as conn:
        df.to_sql(
            name="foo",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )

    with pytest.warns(Warning):
        select({"test.foo": ["a", "b"]}, engine=warehouse_engine)


def test_select_missing_columns(matchbox_api: MockRouter, warehouse_engine: Engine):
    """Selecting columns not in the warehouse errors."""
    source = Source(
        address=SourceAddress.compose(engine=warehouse_engine, full_name="test.foo"),
        db_pk="pk",
    )

    matchbox_api.get(
        f"/sources/{hash_to_base64(source.address.warehouse_hash)}/test.foo"
    ).mock(return_value=Response(200, content=source.model_dump_json()))

    df = DataFrame([{"pk": 0, "a": 1, "b": "2"}, {"pk": 1, "a": 10, "b": "20"}])
    with warehouse_engine.connect() as conn:
        df.to_sql(
            name="foo",
            con=conn,
            schema="test",
            if_exists="replace",
            index=False,
        )
    with pytest.raises(ValueError):
        select({"test.foo": ["a", "c"]}, engine=warehouse_engine)


def test_query_no_resolution_fail():
    """Querying with multiple selectors and no resolution is not allowed."""
    sels = [
        Selector(
            source=Source(
                address=SourceAddress(
                    full_name="foo",
                    warehouse_hash=b"bar",
                ),
                db_pk="i",
            ),
            fields=["a", "b"],
        ),
        Selector(
            source=Source(
                address=SourceAddress(full_name="foo2", warehouse_hash=b"bar2"),
                db_pk="j",
            ),
            fields=["x", "y"],
        ),
    ]

    with pytest.raises(ValueError, match="resolution name"):
        query(sels)


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
    to_arrow.return_value = pa.Table.from_pylist(
        [
            {"foo_pk": "0", "foo_a": 1, "foo_b": "2"},
            {"foo_pk": "1", "foo_a": 10, "foo_b": "20"},
        ]
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
        pa.Table.from_pylist(
            [
                {"foo_pk": "0", "foo_a": 1, "foo_b": "2"},
                {"foo_pk": "1", "foo_a": 10, "foo_b": "20"},
            ]
        ),
        pa.Table.from_pylist(
            [
                {"foo2_pk": "2", "foo2_c": "val"},
                {"foo2_pk": "3", "foo2_c": "val"},
            ]
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
    results = query(sels, resolution_name="link", limit=7)
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
        "resolution_name": "link",
        "limit": "4",
    }
    assert dict(query_route.calls[-1].request.url.params) == {
        "full_name": sels[1].source.address.full_name,
        "warehouse_hash_b64": hash_to_base64(sels[1].source.address.warehouse_hash),
        "resolution_name": "link",
        "limit": "3",
    }

    # It also works with the selectors specified separately
    query([sels[0]], [sels[1]], resolution_name="link", limit=7)


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


@patch("matchbox.client.helpers.index.Source")
def test_index_success(MockSource: Mock, matchbox_api: MockRouter):
    """Test successful indexing flow through the API."""
    engine = create_engine("sqlite:///:memory:")

    # Mock Source
    source = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}], engine=engine
    )
    mock_source_instance = source.to_mock()
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
        engine=engine,
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
):
    """Test indexing with different column definition formats."""
    engine = create_engine("sqlite:///:memory:")

    # Create source dummy and mock
    source = source_factory(
        features=[
            {"name": "name", "base_generator": "name"},
            {"name": "age", "base_generator": "random_int"},
        ],
        engine=engine,
    )
    mock_source_instance = source.to_mock()
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
        engine=engine,
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
    mock_source_instance.set_engine.assert_called_once_with(engine)
    if columns:
        mock_source_instance.default_columns.assert_not_called()
    else:
        mock_source_instance.default_columns.assert_called_once()


@patch("matchbox.client.helpers.index.Source")
def test_index_upload_failure(MockSource: Mock, matchbox_api: MockRouter):
    """Test handling of upload failures."""
    engine = create_engine("sqlite:///:memory:")

    # Mock Source
    source = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}], engine=engine
    )
    mock_source_instance = source.to_mock()
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
            engine=engine,
        )

    # Verify API calls
    source_call = Source.model_validate_json(
        source_route.calls.last.request.content.decode("utf-8")
    )
    assert source_call == source.source
    assert "test-upload-id" in upload_route.calls.last.request.url.path
    assert b"Content-Disposition: form-data;" in upload_route.calls.last.request.content
    assert b"PAR1" in upload_route.calls.last.request.content


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
