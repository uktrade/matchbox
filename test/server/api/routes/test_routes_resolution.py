from time import sleep
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    NotFoundError,
    Resolution,
    ResolutionOperationStatus,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_get_source(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    source_testkit = source_factory(name="foo")
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution = Mock(
        return_value=source_testkit.source.to_resolution(),  # Second call (SOURCE)
    )

    response = test_client.get("/resolutions/foo")
    assert response.status_code == 200
    assert response.json()["name"] == "foo"
    assert response.json()["resolution_type"] == "source"


def test_get_model(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory(name="test_model", description="test description")
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution = Mock(return_value=testkit.model.to_resolution())

    response = test_client.get("/resolutions/test_model")

    assert response.status_code == 200
    assert response.json()["name"] == testkit.model.name
    assert response.json()["description"] == testkit.model.description


def test_get_resolution_404(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.get("/resolutions/nonexistent")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_insert_model(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory(name="test_model")
    test_client, mock_backend, _ = api_client_and_mocks

    response = test_client.post(
        "/resolutions", json=testkit.model.to_resolution().model_dump()
    )

    assert response.status_code == 201
    assert (
        response.json()
        == ResolutionOperationStatus(
            success=True,
            name="test_model",
            operation=CRUDOperation.CREATE,
            details=None,
        ).model_dump()
    )

    mock_backend.insert_resolution.assert_called_once_with(
        resolution=testkit.model.to_resolution()
    )


def test_insert_model_error(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.insert_resolution = Mock(side_effect=Exception("Test error"))
    testkit = model_factory()

    response = test_client.post(
        "/resolutions", json=testkit.model.to_resolution().model_dump()
    )

    assert response.status_code == 500
    assert response.json()["success"] is False
    assert response.json()["details"] == "Test error"


@pytest.mark.parametrize("model_type", ["deduper", "linker"])
def test_complete_model_upload_process(
    s3: S3Client,
    model_type: str,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test the complete upload process for models from creation through processing."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.insert_model_data = Mock(return_value=None)

    # Create test bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data with specified model type
    testkit = model_factory(model_type=model_type)

    # Set up the mock to return the actual model metadata and data
    mock_backend.get_resolution = Mock(return_value=testkit.model.to_resolution())
    mock_backend.get_model_data = Mock(return_value=testkit.probabilities)

    # Step 1: Create model
    response = test_client.post(
        "/resolutions", json=testkit.model.to_resolution().model_dump()
    )
    assert response.status_code == 201
    assert response.json()["success"] is True
    assert response.json()["name"] == testkit.model.name

    # Step 2: Initialize results upload
    response = test_client.post(f"/resolutions/{testkit.model.name}/data")
    assert response.status_code == 202
    upload_id = response.json()["id"]
    assert response.json()["stage"] == UploadStage.AWAITING_UPLOAD

    # Step 3: Upload results file with real background tasks
    response = test_client.post(
        f"/upload/{upload_id}",
        files={
            "file": (
                "results.parquet",
                table_to_buffer(testkit.probabilities),
                "application/octet-stream",
            ),
        },
    )
    assert response.status_code == 202
    assert response.json()["stage"] == UploadStage.QUEUED

    # Step 4: Poll status until complete or timeout
    max_attempts = 10
    current_attempt = 0
    stage = None

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

    # Step 5: Verify results were stored correctly
    mock_backend.insert_model_data.assert_called_once()
    call_args = mock_backend.insert_model_data.call_args
    assert (
        call_args[1]["name"] == testkit.model.name
    )  # Check model resolution name matches
    assert call_args[1]["results"].equals(
        testkit.probabilities
    )  # Check results data matches

    # Step 6: Verify we can retrieve the results
    response = test_client.get(f"/resolutions/{testkit.model.name}/data")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    # Verify the model truth can be set and retrieved
    truth_value = 85
    mock_backend.get_model_truth = Mock(return_value=truth_value)

    response = test_client.patch(
        f"/resolutions/{testkit.model.name}/truth",
        json=truth_value,
    )
    assert response.status_code == 200

    response = test_client.get(f"/resolutions/{testkit.model.name}/truth")
    assert response.status_code == 200
    assert response.json() == truth_value

    # Verify file is deleted from S3 after processing
    with pytest.raises(ClientError):
        s3.head_object(Bucket="test-bucket", Key=f"{upload_id}.parquet")


def test_set_results(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution = Mock(return_value=testkit.model.to_resolution())

    response = test_client.post(f"/resolutions/{testkit.model.name}/data")

    assert response.status_code == 202
    assert response.json()["stage"] == UploadStage.AWAITING_UPLOAD


def test_set_results_model_not_found(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test setting results for a non-existent model."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.post("/resolutions/nonexistent-model/data")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_get_results(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model_data = Mock(return_value=testkit.probabilities)

    response = test_client.get(f"/resolutions/{testkit.model.name}/data")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


def test_set_truth(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks

    response = test_client.patch(f"/resolutions/{testkit.model.name}/truth", json=95)

    assert response.status_code == 200
    assert response.json()["success"] is True
    mock_backend.set_model_truth.assert_called_once_with(
        name=testkit.model.name, truth=95
    )


def test_set_truth_invalid_value(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test setting an invalid truth value (outside 0-1 range)."""
    testkit = model_factory()
    test_client, _, _ = api_client_and_mocks

    # Test value > 1
    response = test_client.patch(f"/resolutions/{testkit.model.name}/truth", json=150)
    assert response.status_code == 422

    # Test value < 0
    response = test_client.patch(f"/resolutions/{testkit.model.name}/truth", json=-50)
    assert response.status_code == 422


def test_get_truth(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model_truth = Mock(return_value=95)

    response = test_client.get(f"/resolutions/{testkit.model.name}/truth")

    assert response.status_code == 200
    assert response.json() == 95


@pytest.mark.parametrize(
    "endpoint",
    ["data", "truth"],
)
def test_model_get_endpoints_404(
    endpoint: str,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 responses for model GET endpoints when model doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    # Map endpoint to actual backend method name
    method_name = "get_model_data" if endpoint == "data" else f"get_model_{endpoint}"
    mock_method = getattr(mock_backend, method_name)
    mock_method.side_effect = MatchboxResolutionNotFoundError()

    response = test_client.get(f"/resolutions/nonexistent-model/{endpoint}")

    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendResourceType.RESOLUTION


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("truth", 95),
    ],
)
def test_model_patch_endpoints_404(
    endpoint: str,
    payload: float | list[dict[str, Any]],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 responses for model PATCH endpoints when model doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_method = getattr(mock_backend, f"set_model_{endpoint}")
    mock_method.side_effect = MatchboxResolutionNotFoundError()

    response = test_client.patch(
        f"/resolutions/nonexistent-model/{endpoint}", json=payload
    )

    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendResourceType.RESOLUTION


def test_delete_resolution(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deletion of a resolution."""
    testkit = model_factory()
    test_client, _, _ = api_client_and_mocks
    response = test_client.delete(
        f"/resolutions/{testkit.model.name}",
        params={"certain": True},
    )

    assert response.status_code == 200
    assert (
        response.json()
        == ResolutionOperationStatus(
            success=True,
            name=testkit.model.name,
            operation=CRUDOperation.DELETE,
            details=None,
        ).model_dump()
    )


def test_delete_resolution_needs_confirmation(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test deletion of a model that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_resolution = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=["dedupe1", "dedupe2"])
    )

    testkit = model_factory()
    response = test_client.delete(f"/resolutions/{testkit.model.name}")

    assert response.status_code == 409
    assert response.json()["success"] is False
    message = response.json()["details"]
    assert "dedupe1" in message and "dedupe2" in message


@pytest.mark.parametrize("certain", [True, False])
def test_delete_resolution_404(
    certain: bool,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 response when trying to delete a non-existent resolution."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_resolution.side_effect = MatchboxResolutionNotFoundError()

    response = test_client.delete(
        "/resolutions/nonexistent-model", params={"certain": certain}
    )

    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendResourceType.RESOLUTION


def test_get_resolution_sources(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    source = source_factory().source.to_resolution()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_leaf_source_resolutions = Mock(return_value=[source])

    response = test_client.get("/resolutions/foo/sources")
    assert response.status_code == 200
    for s in response.json():
        assert Resolution.model_validate(s)


def test_get_resolution_sources_404(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_leaf_source_resolutions = Mock(
        side_effect=MatchboxResolutionNotFoundError
    )

    response = test_client.get("/resolutions/foo/sources")
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_complete_source_upload_process(
    s3: S3Client,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test the complete upload process from source creation through processing."""
    # Create test data
    source_testkit = source_factory()

    # Setup the backend
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"

    mock_backend.get_resolution = Mock(
        return_value=source_testkit.source.to_resolution()
    )
    mock_backend.insert_resolution = Mock(return_value=None)
    mock_backend.insert_source_data = Mock(return_value=None)

    # Create test bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Step 1: Add source
    response = test_client.post(
        "/resolutions",
        json=source_testkit.source.to_resolution().model_dump(mode="json"),
    )
    assert response.status_code == 201
    status = ResolutionOperationStatus.model_validate(response.json())
    assert response.status_code == 201, response.json()
    assert status.name == source_testkit.name

    # Assert backend given the config but not yet the data
    mock_backend.insert_resolution.assert_called_once_with(
        resolution=source_testkit.source.to_resolution()
    )
    mock_backend.insert_source_data.assert_not_called()

    response = test_client.post(f"/resolutions/{source_testkit.name}/data")
    upload_id = response.json()["id"]
    assert response.status_code == 202
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

    # Verify backend methods were called with correct arguments
    mock_backend.insert_source_data.assert_called_once()

    # Check resolution matches
    call_args = mock_backend.insert_source_data.call_args
    assert call_args[1]["data_hashes"].equals(source_testkit.data_hashes)  # Check data

    # Verify file is deleted from S3 after processing
    with pytest.raises(ClientError):
        s3.head_object(Bucket="test-bucket", Key=f"{upload_id}.parquet")
