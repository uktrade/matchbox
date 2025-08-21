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
    ModelAncestor,
    NotFoundError,
    ResolutionOperationStatus,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.models import model_factory

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_insert_model(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory(name="test_model")
    test_client, mock_backend, _ = api_client_and_mocks

    response = test_client.post("/models", json=testkit.model.model_config.model_dump())

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

    mock_backend.insert_model.assert_called_once_with(testkit.model.model_config)


def test_insert_model_error(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.insert_model = Mock(side_effect=Exception("Test error"))
    testkit = model_factory()

    response = test_client.post("/models", json=testkit.model.model_config.model_dump())

    assert response.status_code == 500
    assert response.json()["success"] is False
    assert response.json()["details"] == "Test error"


def test_get_model(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory(name="test_model", description="test description")
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model = Mock(return_value=testkit.model.model_config)

    response = test_client.get("/models/test_model")

    assert response.status_code == 200
    assert response.json()["name"] == testkit.model.model_config.name
    assert response.json()["description"] == testkit.model.model_config.description


def test_get_model_not_found(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.get("/models/nonexistent")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


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
    mock_backend.set_model_results = Mock(return_value=None)

    # Create test bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data with specified model type
    testkit = model_factory(model_type=model_type)

    # Set up the mock to return the actual model metadata and data
    mock_backend.get_model = Mock(return_value=testkit.model.model_config)
    mock_backend.get_model_results = Mock(return_value=testkit.probabilities)

    # Step 1: Create model
    response = test_client.post("/models", json=testkit.model.model_config.model_dump())
    assert response.status_code == 201
    assert response.json()["success"] is True
    assert response.json()["name"] == testkit.model.model_config.name

    # Step 2: Initialize results upload
    response = test_client.post(f"/models/{testkit.model.model_config.name}/results")
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
    mock_backend.set_model_results.assert_called_once()
    call_args = mock_backend.set_model_results.call_args
    assert (
        call_args[1]["name"] == testkit.model.model_config.name
    )  # Check model resolution name matches
    assert call_args[1]["results"].equals(
        testkit.probabilities
    )  # Check results data matches

    # Step 6: Verify we can retrieve the results
    response = test_client.get(f"/models/{testkit.model.model_config.name}/results")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    # Step 7: Additional model-specific verifications
    if model_type == "linker":
        # For linker models, verify left and right resolutions are set
        assert testkit.model.model_config.left_resolution is not None
        assert testkit.model.model_config.right_resolution is not None
    else:
        # For deduper models, verify only left resolution is set
        assert testkit.model.model_config.left_resolution is not None
        assert testkit.model.model_config.right_resolution is None

    # Verify the model truth can be set and retrieved
    truth_value = 85
    mock_backend.get_model_truth = Mock(return_value=truth_value)

    response = test_client.patch(
        f"/models/{testkit.model.model_config.name}/truth",
        json=truth_value,
    )
    assert response.status_code == 200

    response = test_client.get(f"/models/{testkit.model.model_config.name}/truth")
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
    mock_backend.get_model = Mock(return_value=testkit.model.model_config)

    response = test_client.post(f"/models/{testkit.model.model_config.name}/results")

    assert response.status_code == 202
    assert response.json()["stage"] == UploadStage.AWAITING_UPLOAD


def test_set_results_model_not_found(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test setting results for a non-existent model."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.post("/models/nonexistent-model/results")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_get_results(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model_results = Mock(return_value=testkit.probabilities)

    response = test_client.get(f"/models/{testkit.model.model_config.name}/results")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


def test_set_truth(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks

    response = test_client.patch(
        f"/models/{testkit.model.model_config.name}/truth", json=95
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    mock_backend.set_model_truth.assert_called_once_with(
        name=testkit.model.model_config.name, truth=95
    )


def test_set_truth_invalid_value(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test setting an invalid truth value (outside 0-1 range)."""
    testkit = model_factory()
    test_client, _, _ = api_client_and_mocks

    # Test value > 1
    response = test_client.patch(
        f"/models/{testkit.model.model_config.name}/truth", json=150
    )
    assert response.status_code == 422

    # Test value < 0
    response = test_client.patch(
        f"/models/{testkit.model.model_config.name}/truth", json=-50
    )
    assert response.status_code == 422


def test_get_truth(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model_truth = Mock(return_value=95)

    response = test_client.get(f"/models/{testkit.model.model_config.name}/truth")

    assert response.status_code == 200
    assert response.json() == 95


def test_get_ancestors(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    testkit = model_factory()
    mock_ancestors = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=97),
    ]
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model_ancestors = Mock(return_value=mock_ancestors)

    response = test_client.get(f"/models/{testkit.model.model_config.name}/ancestors")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert [ModelAncestor.model_validate(a) for a in response.json()] == mock_ancestors


def test_get_ancestors_cache(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test retrieving the ancestors cache for a model."""
    testkit = model_factory()
    mock_ancestors = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=80),
    ]
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_model_ancestors_cache = Mock(return_value=mock_ancestors)

    response = test_client.get(
        f"/models/{testkit.model.model_config.name}/ancestors_cache"
    )

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert [ModelAncestor.model_validate(a) for a in response.json()] == mock_ancestors


def test_set_ancestors_cache(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test setting the ancestors cache for a model."""
    testkit = model_factory()
    test_client, mock_backend, _ = api_client_and_mocks

    ancestors_data = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=80),
    ]

    response = test_client.patch(
        f"/models/{testkit.model.model_config.name}/ancestors_cache",
        json=[a.model_dump() for a in ancestors_data],
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == CRUDOperation.UPDATE
    mock_backend.set_model_ancestors_cache.assert_called_once_with(
        name=testkit.model.model_config.name, ancestors_cache=ancestors_data
    )


@pytest.mark.parametrize(
    "endpoint",
    ["results", "truth", "ancestors", "ancestors_cache"],
)
def test_model_get_endpoints_404(
    endpoint: str,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 responses for model GET endpoints when model doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_method = getattr(mock_backend, f"get_model_{endpoint}")
    mock_method.side_effect = MatchboxResolutionNotFoundError()

    response = test_client.get(f"/models/nonexistent-model/{endpoint}")

    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendResourceType.RESOLUTION


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("truth", 95),
        (
            "ancestors_cache",
            [
                ModelAncestor(name="parent_model", truth=70).model_dump(),
                ModelAncestor(name="grandparent_model", truth=80).model_dump(),
            ],
        ),
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

    response = test_client.patch(f"/models/nonexistent-model/{endpoint}", json=payload)

    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendResourceType.RESOLUTION


def test_delete_resolution(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deletion of a resolution."""
    testkit = model_factory()
    test_client, _, _ = api_client_and_mocks
    response = test_client.delete(
        f"/resolutions/{testkit.model.model_config.name}",
        params={"certain": True},
    )

    assert response.status_code == 200
    assert (
        response.json()
        == ResolutionOperationStatus(
            success=True,
            name=testkit.model.model_config.name,
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
    response = test_client.delete(f"/resolutions/{testkit.model.model_config.name}")

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
