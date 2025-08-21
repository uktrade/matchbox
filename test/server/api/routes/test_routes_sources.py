from time import sleep
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import BackendResourceType, UploadStage, UploadStatus
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.common.sources import SourceConfig

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_get_source(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    source = source_factory(name="foo").source_config
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_source_config = Mock(return_value=source)

    response = test_client.get("/sources/foo")
    assert response.status_code == 200
    assert SourceConfig.model_validate(response.json())


def test_get_source_404(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_source_config = Mock(side_effect=MatchboxSourceNotFoundError)

    response = test_client.get("/sources/foo")
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.SOURCE


def test_get_resolution_sources(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    source = source_factory().source_config
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution_source_configs = Mock(return_value=[source])

    response = test_client.get("/sources", params={"name": "foo"})
    assert response.status_code == 200
    for s in response.json():
        assert SourceConfig.model_validate(s)


def test_get_resolution_sources_404(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution_source_configs = Mock(
        side_effect=MatchboxResolutionNotFoundError
    )

    response = test_client.get("/sources", params={"name": "foo"})
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_add_source(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test the source addition endpoint."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.index = Mock(return_value=None)

    source_testkit = source_factory()

    # Make request
    response = test_client.post(
        "/sources",
        json=source_testkit.source_config.model_dump(mode="json"),
    )

    # Validate response
    assert UploadStatus.model_validate(response.json())
    assert response.status_code == 202, response.json()
    assert response.json()["stage"] == UploadStage.AWAITING_UPLOAD
    assert response.json().get("id") is not None
    mock_backend.index.assert_not_called()


def test_complete_source_upload_process(
    s3: S3Client,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test the complete upload process from source creation through processing."""
    # Setup the backend
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.index = Mock(return_value=None)

    # Create test bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data
    source_testkit = source_factory()

    # Step 1: Add source
    response = test_client.post(
        "/sources", json=source_testkit.source_config.model_dump(mode="json")
    )
    assert response.status_code == 202
    upload_id = response.json()["id"]
    assert response.json()["stage"] == UploadStage.AWAITING_UPLOAD

    # Step 2: Upload file with real background tasks
    response = test_client.post(
        f"/upload/{upload_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(source_testkit.data_hashes),
                "application/octet-stream",
            ),
        },
    )
    assert response.status_code == 202
    assert response.json()["stage"] == UploadStage.QUEUED

    # Step 3: Poll status until complete or timeout
    max_attempts = 10
    current_attempt = 0
    while current_attempt < max_attempts:
        response = test_client.get(f"/upload/{upload_id}/status")
        assert response.status_code == 200

        stage = response.json()["stage"]
        if stage == UploadStage.COMPLETE:
            break
        elif stage == UploadStage.FAILED:
            pytest.fail(f"Upload failed: {response.json().get('details')}")
        elif stage in [UploadStage.PROCESSING, UploadStage.QUEUED]:
            sleep(0.1)  # Small delay between polls
        else:
            pytest.fail(f"Unexpected stage: {stage}")

        current_attempt += 1

    assert current_attempt < max_attempts, (
        "Timed out waiting for processing to complete"
    )
    assert stage == UploadStage.COMPLETE
    assert response.status_code == 200

    # Verify backend.index was called with correct arguments
    mock_backend.index.assert_called_once()
    call_args = mock_backend.index.call_args
    assert (
        call_args[1]["source_config"] == source_testkit.source_config
    )  # Check source matches
    assert call_args[1]["data_hashes"].equals(source_testkit.data_hashes)  # Check data

    # Verify file is deleted from S3 after processing
    with pytest.raises(ClientError):
        s3.head_object(Bucket="test-bucket", Key=f"{upload_id}.parquet")
