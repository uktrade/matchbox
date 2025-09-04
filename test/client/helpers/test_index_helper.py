from datetime import datetime
from unittest.mock import patch

import pytest
from httpx import Response
from respx import MockRouter
from sqlalchemy import Engine, create_engine

from matchbox.client.helpers.index import get_source, index
from matchbox.client.sources import RelationalDBLocation, Source
from matchbox.common.dtos import (
    BackendResourceType,
    BackendUploadType,
    NotFoundError,
    SourceConfig,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import source_factory, source_from_tuple


def test_index_success(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test successful indexing flow through the API."""
    # Mock Source
    source_testkit = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}],
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock the initial source metadata upload
    source_route = matchbox_api.post("/sources").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.AWAITING_UPLOAD,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    # Mock the data upload
    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.COMPLETE,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    # Call the index function
    index(source=source_testkit.source)

    # Verify the API calls
    source_call = SourceConfig.model_validate_json(
        source_route.calls.last.request.content.decode("utf-8")
    )
    assert source_call == source_testkit.source_config
    assert "test-upload-id" in upload_route.calls.last.request.url.path
    assert b"Content-Disposition: form-data;" in upload_route.calls.last.request.content
    assert b"PAR1" in upload_route.calls.last.request.content


def test_index_upload_failure(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test handling of upload failures."""
    # Mock Source
    source_testkit = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}],
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock successful source creation
    source_route = matchbox_api.post("/sources").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.AWAITING_UPLOAD,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    # Mock failed upload
    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            400,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.FAILED,
                update_timestamp=datetime.now(),
                details="Invalid schema",
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    # Verify the error is propagated
    with pytest.raises(MatchboxServerFileError):
        index(source=source_testkit.source)

    # Verify API calls
    source_call = SourceConfig.model_validate_json(
        source_route.calls.last.request.content.decode("utf-8")
    )
    assert source_call == source_testkit.source_config
    assert "test-upload-id" in upload_route.calls.last.request.url.path
    assert b"Content-Disposition: form-data;" in upload_route.calls.last.request.content
    assert b"PAR1" in upload_route.calls.last.request.content


def test_index_with_batch_size(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test that batch_size is passed correctly to hash_data when indexing."""
    # Dummy data and source
    source_testkit = source_from_tuple(
        data_tuple=({"company_name": "Company A"}, {"company_name": "Company B"}),
        data_keys=["1", "2"],
        name="test_companies",
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock the API endpoints
    source_route = matchbox_api.post("/sources").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.AWAITING_UPLOAD,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                id="test-upload-id",
                stage=UploadStage.COMPLETE,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.INDEX,
            ).model_dump_json(),
        )
    )

    # Spy on the hash_data method to verify batch_size
    with patch.object(
        Source, "hash_data", wraps=source_testkit.source.hash_data
    ) as spy_hash_data:
        # Call index with batch_size
        index(
            source=source_testkit.source,
            batch_size=1,
        )

        # Verify batch_size was passed to hash_data
        spy_hash_data.assert_called_once()
        assert spy_hash_data.call_args.kwargs["batch_size"] == 1

        # Verify endpoints were called only once, despite multiple batches
        assert source_route.call_count == 1
        assert upload_route.call_count == 1


def test_get_source_success(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test successful retrieval of source config."""
    # Create test source
    testkit = source_factory(
        engine=sqlite_warehouse, name="test_source"
    ).write_to_location()

    # Mock API response
    matchbox_api.get("/sources/test_source").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
    )

    # Call function
    result = get_source(
        "test_source",
        location=RelationalDBLocation(
            name=testkit.source_config.location_config.name, client=sqlite_warehouse
        ),
    )

    # Verify result
    assert result.name == "test_source"
    assert isinstance(result, Source)


def test_get_source_with_valid_location(
    matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Test get_source with matching location validation."""
    testkit = source_factory(
        engine=sqlite_warehouse, name="test_source"
    ).write_to_location()

    matchbox_api.get("/sources/test_source").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
    )

    # Should succeed when location matches
    location = RelationalDBLocation(
        name=testkit.source_config.location_config.name, client=sqlite_warehouse
    )
    result = get_source("test_source", location=location)
    assert result.name == "test_source"


@pytest.mark.parametrize(
    ["validation_param", "validation_value", "expected_error"],
    [
        pytest.param(
            "location",
            RelationalDBLocation(
                name="other_location", client=create_engine("sqlite:///:memory:")
            ),
            "does not match the provided location",
            id="location-mismatch",
        ),
        pytest.param(
            "extract_transform",
            "different_transform",
            "does not match the provided extract/transform",
            id="extract-transform-mismatch",
        ),
        pytest.param(
            "key_field",
            "different_key",
            "does not match the provided key field",
            id="key-field-mismatch",
        ),
        pytest.param(
            "index_fields",
            ["different_field"],
            "does not match the provided index fields",
            id="index-fields-mismatch",
        ),
    ],
)
def test_get_source_validation_mismatch(
    validation_param: str,
    validation_value,
    expected_error: str,
    matchbox_api: MockRouter,
    sqlite_warehouse: Engine,
):
    """Test get_source raises error when validation parameters don't match."""
    testkit = source_factory(
        engine=sqlite_warehouse, name="test_source"
    ).write_to_location()

    matchbox_api.get("/sources/test_source").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
    )

    kwargs = {validation_param: validation_value}
    if validation_param != "location":
        kwargs["location"] = testkit.source.location
    with pytest.raises(ValueError, match=expected_error):
        get_source("test_source", **kwargs)


def test_get_source_404_error(matchbox_api: MockRouter):
    """Test get_source handles 404 source not found error."""
    matchbox_api.get("/sources/nonexistent").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="SourceConfig nonexistent not found",
                entity=BackendResourceType.SOURCE,
            ).model_dump(),
        )
    )

    with pytest.raises(MatchboxSourceNotFoundError, match="nonexistent"):
        get_source(name="nonexistent", location=RelationalDBLocation("", None))
