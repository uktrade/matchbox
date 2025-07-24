import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

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
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.models import model_factory
from matchbox.server.api.cache import MetadataStore
from matchbox.server.api.dependencies import backend, metadata_store
from matchbox.server.api.main import app

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_insert_model(test_client: TestClient):
    testkit = model_factory(name="test_model")
    mock_backend = Mock()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

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


def test_insert_model_error(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.insert_model = Mock(side_effect=Exception("Test error"))
    testkit = model_factory()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.post("/models", json=testkit.model.model_config.model_dump())

    assert response.status_code == 500
    assert response.json()["success"] is False
    assert response.json()["details"] == "Test error"


def test_get_model(test_client: TestClient):
    testkit = model_factory(name="test_model", description="test description")
    mock_backend = Mock()
    mock_backend.get_model = Mock(return_value=testkit.model.model_config)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get("/models/test_model")

    assert response.status_code == 200
    assert response.json()["name"] == testkit.model.model_config.name
    assert response.json()["description"] == testkit.model.model_config.description


def test_get_model_not_found(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.get_model = Mock(side_effect=MatchboxResolutionNotFoundError())

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get("/models/nonexistent")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


@pytest.mark.parametrize("model_type", ["deduper", "linker"])
@patch("matchbox.server.api.main.BackgroundTasks.add_task")
def test_model_upload(
    mock_add_task: Mock,
    s3: S3Client,
    model_type: str,
    test_client: TestClient,
):
    """Test uploading different types of files."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data with specified model type
    testkit = model_factory(model_type=model_type)

    # Setup metadata store
    mock_metadata_store = Mock()
    store = MetadataStore()
    upload_id = store.cache_model(testkit.model.model_config)

    mock_metadata_store.get.side_effect = store.get
    mock_metadata_store.update_status.side_effect = store.update_status

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend
    app.dependency_overrides[metadata_store] = lambda: mock_metadata_store

    # Make request
    response = test_client.post(
        f"/upload/{upload_id}",
        files={
            "file": (
                "data.parquet",
                table_to_buffer(testkit.probabilities),
                "application/octet-stream",
            ),
        },
    )

    # Validate response
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    mock_add_task.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["deduper", "linker"])
async def test_complete_model_upload_process(
    s3: S3Client, model_type: str, test_client: TestClient
):
    """Test the complete upload process for models from creation through processing."""
    # Setup the backend
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.set_model_results = Mock(return_value=None)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

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
    assert response.json()["status"] == "awaiting_upload"

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
    assert response.json()["status"] == "queued"

    # Step 4: Poll status until complete or timeout
    max_attempts = 10
    current_attempt = 0
    status = None

    while current_attempt < max_attempts:
        response = test_client.get(f"/upload/{upload_id}/status")
        assert response.status_code == 200

        status = response.json()["status"]
        if status == "complete":
            break
        elif status == "failed":
            pytest.fail(f"Upload failed: {response.json().get('details')}")
        elif status in ["processing", "queued"]:
            await asyncio.sleep(0.1)  # Small delay between polls
        else:
            pytest.fail(f"Unexpected status: {status}")

        current_attempt += 1

    assert current_attempt < max_attempts, (
        "Timed out waiting for processing to complete"
    )
    assert status == "complete"
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


def test_set_results(test_client: TestClient):
    testkit = model_factory()
    mock_backend = Mock()
    mock_backend.get_model = Mock(return_value=testkit.model.model_config)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.post(f"/models/{testkit.model.model_config.name}/results")

    assert response.status_code == 202
    assert response.json()["status"] == "awaiting_upload"


def test_set_results_model_not_found(test_client: TestClient):
    """Test setting results for a non-existent model."""
    mock_backend = Mock()
    mock_backend.get_model = Mock(side_effect=MatchboxResolutionNotFoundError())

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.post("/models/nonexistent-model/results")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_get_results(test_client: TestClient):
    testkit = model_factory()
    mock_backend = Mock()
    mock_backend.get_model_results = Mock(return_value=testkit.probabilities)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get(f"/models/{testkit.model.model_config.name}/results")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


def test_set_truth(test_client: TestClient):
    testkit = model_factory()
    mock_backend = Mock()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.patch(
        f"/models/{testkit.model.model_config.name}/truth", json=95
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    mock_backend.set_model_truth.assert_called_once_with(
        name=testkit.model.model_config.name, truth=95
    )


def test_set_truth_invalid_value(test_client: TestClient):
    """Test setting an invalid truth value (outside 0-1 range)."""
    testkit = model_factory()

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


def test_get_truth(test_client: TestClient):
    testkit = model_factory()
    mock_backend = Mock()
    mock_backend.get_model_truth = Mock(return_value=95)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get(f"/models/{testkit.model.model_config.name}/truth")

    assert response.status_code == 200
    assert response.json() == 95


def test_get_ancestors(test_client: TestClient):
    testkit = model_factory()
    mock_ancestors = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=97),
    ]
    mock_backend = Mock()
    mock_backend.get_model_ancestors = Mock(return_value=mock_ancestors)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get(f"/models/{testkit.model.model_config.name}/ancestors")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert [ModelAncestor.model_validate(a) for a in response.json()] == mock_ancestors


def test_get_ancestors_cache(test_client: TestClient):
    """Test retrieving the ancestors cache for a model."""
    testkit = model_factory()
    mock_ancestors = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=80),
    ]
    mock_backend = Mock()
    mock_backend.get_model_ancestors_cache = Mock(return_value=mock_ancestors)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.get(
        f"/models/{testkit.model.model_config.name}/ancestors_cache"
    )

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert [ModelAncestor.model_validate(a) for a in response.json()] == mock_ancestors


def test_set_ancestors_cache(test_client: TestClient):
    """Test setting the ancestors cache for a model."""
    testkit = model_factory()
    mock_backend = Mock()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

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
def test_model_get_endpoints_404(endpoint: str, test_client: TestClient) -> None:
    """Test 404 responses for model GET endpoints when model doesn't exist."""
    # Setup backend mock
    mock_backend = Mock()
    mock_method = getattr(mock_backend, f"get_model_{endpoint}")
    mock_method.side_effect = MatchboxResolutionNotFoundError()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Make request
    response = test_client.get(f"/models/nonexistent-model/{endpoint}")

    # Verify response
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
    test_client: TestClient,
) -> None:
    """Test 404 responses for model PATCH endpoints when model doesn't exist."""
    # Setup backend mock
    mock_backend = Mock()
    mock_method = getattr(mock_backend, f"set_model_{endpoint}")
    mock_method.side_effect = MatchboxResolutionNotFoundError()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Make request
    response = test_client.patch(f"/models/nonexistent-model/{endpoint}", json=payload)

    # Verify response
    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendResourceType.RESOLUTION


def test_delete_resolution(test_client: TestClient):
    """Test deletion of a resolution."""
    testkit = model_factory()
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


def test_delete_resolution_needs_confirmation(test_client: TestClient):
    """Test deletion of a model that requires confirmation."""
    mock_backend = Mock()
    mock_backend.delete_resolution = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=["dedupe1", "dedupe2"])
    )

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    testkit = model_factory()
    response = test_client.delete(f"/resolutions/{testkit.model.model_config.name}")

    assert response.status_code == 409
    assert response.json()["success"] is False
    message = response.json()["details"]
    assert "dedupe1" in message and "dedupe2" in message


@pytest.mark.parametrize(
    "certain",
    [True, False],
)
def test_delete_resolution_404(certain: bool, test_client: TestClient) -> None:
    """Test 404 response when trying to delete a non-existent resolution."""
    # Setup backend mock
    mock_backend = Mock()
    mock_backend.delete_resolution.side_effect = MatchboxResolutionNotFoundError()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    # Make request
    response = test_client.delete(
        "/resolutions/nonexistent-model", params={"certain": certain}
    )

    # Verify response
    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendResourceType.RESOLUTION
