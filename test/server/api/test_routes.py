import asyncio
from importlib.metadata import version
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import ANY, Mock, call, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.dtos import (
    BackendRetrievableType,
    ModelAncestor,
    ModelOperationType,
    NotFoundError,
    OKMessage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import hash_to_base64
from matchbox.common.sources import Match, Source, SourceAddress
from matchbox.server.api.cache import MetadataStore, process_upload
from matchbox.server.base import MatchboxDBAdapter

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


# General


def test_healthcheck(test_client: TestClient):
    """Test the healthcheck endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    response = OKMessage.model_validate(response.json())
    assert response.status == "OK"
    assert response.version == version("matchbox-db")


@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
@patch("matchbox.server.api.routes.BackgroundTasks.add_task")
def test_upload(
    mock_add_task: Mock,
    metadata_store: Mock,
    get_backend: Mock,
    s3: S3Client,
    test_client: TestClient,
):
    """Test uploading a file, happy path."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.index = Mock(return_value=None)
    get_backend.return_value = mock_backend
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    source_testkit = source_factory()

    # Mock the metadata store
    store = MetadataStore()
    update_id = store.cache_source(source_testkit.source)
    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

    # Make request with mocked background task
    response = test_client.post(
        f"/upload/{update_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(source_testkit.data_hashes),
                "application/octet-stream",
            ),
        },
    )

    # Validate response
    assert UploadStatus.model_validate(response.json())
    assert response.status_code == 202, response.json()
    assert response.json()["status"] == "queued"  # Updated to check for queued status
    # Check both status updates were called in correct order
    assert metadata_store.update_status.call_args_list == [
        call(update_id, "queued"),
    ]
    mock_backend.index.assert_not_called()  # Index happens in background
    mock_add_task.assert_called_once()  # Verify background task was queued


@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
@patch("matchbox.server.api.routes.BackgroundTasks.add_task")
def test_upload_wrong_schema(
    mock_add_task: Mock,
    metadata_store: Mock,
    get_backend: Mock,
    s3: S3Client,
    test_client: TestClient,
):
    """Test uploading a file with wrong schema."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    get_backend.return_value = mock_backend

    # Create source with results schema instead of index
    source_testkit = source_factory()

    # Setup store
    store = MetadataStore()
    update_id = store.cache_source(source_testkit.source)
    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

    # Make request with actual data instead of the hashes -- wrong schema
    response = test_client.post(
        f"/upload/{update_id}",
        files={
            "file": (
                "hashes.parquet",
                table_to_buffer(source_testkit.data),
                "application/octet-stream",
            ),
        },
    )

    # Should fail before background task starts
    assert response.status_code == 400
    assert response.json()["status"] == "failed"
    assert "schema mismatch" in response.json()["details"].lower()
    metadata_store.update_status.assert_called_with(update_id, "failed", details=ANY)
    mock_add_task.assert_not_called()  # Background task should not be queued


@patch("matchbox.server.base.BackendManager.get_backend")  # Stops real backend call
@patch("matchbox.server.api.routes.metadata_store")
def test_upload_status_check(metadata_store: Mock, _: Mock, test_client: TestClient):
    """Test checking status of an upload using the status endpoint."""
    # Setup store with a processing entry
    store = MetadataStore()
    source_testkit = source_factory()
    update_id = store.cache_source(source_testkit.source)
    store.update_status(update_id, "processing")

    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

    # Check status using GET endpoint
    response = test_client.get(f"/upload/{update_id}/status")

    # Should return current status
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    metadata_store.update_status.assert_not_called()


@patch("matchbox.server.base.BackendManager.get_backend")  # Stops real backend call
@patch("matchbox.server.api.routes.metadata_store")
def test_upload_already_processing(
    metadata_store: Mock, _: Mock, test_client: TestClient
):
    """Test attempting to upload when status is already processing."""
    # Setup store with a processing entry
    store = MetadataStore()
    source_testkit = source_factory()
    update_id = store.cache_source(source_testkit.source)
    store.update_status(update_id, "processing")

    metadata_store.get.side_effect = store.get

    # Attempt upload
    response = test_client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["status"] == "processing"


@patch("matchbox.server.base.BackendManager.get_backend")  # Stops real backend call
@patch("matchbox.server.api.routes.metadata_store")
def test_upload_already_queued(metadata_store: Mock, _: Mock, test_client: TestClient):
    """Test attempting to upload when status is already queued."""
    # Setup store with a queued entry
    store = MetadataStore()
    source_testkit = source_factory()
    update_id = store.cache_source(source_testkit.source)
    store.update_status(update_id, "queued")

    metadata_store.get.side_effect = store.get

    # Attempt upload
    response = test_client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["status"] == "queued"


@patch("matchbox.server.api.routes.metadata_store")
def test_status_check_not_found(metadata_store: Mock, test_client: TestClient):
    """Test checking status for non-existent upload ID."""
    metadata_store.get.return_value = None

    response = test_client.get("/upload/nonexistent-id/status")

    assert response.status_code == 400
    assert response.json()["status"] == "failed"
    assert "not found or expired" in response.json()["details"].lower()


@pytest.mark.asyncio
async def test_process_upload_deletes_file_on_failure(s3: S3Client):
    """Test that files are deleted from S3 even when processing fails."""
    # Setup
    bucket = "test-bucket"
    test_key = "test-upload-id.parquet"
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.index = Mock(side_effect=ValueError("Simulated processing failure"))

    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Add parquet to S3 and verify
    source_testkit = source_factory()
    buffer = table_to_buffer(source_testkit.data_hashes)
    s3.put_object(Bucket=bucket, Key=test_key, Body=buffer)

    assert s3.head_object(Bucket=bucket, Key=test_key)

    # Setup metadata store with test data
    store = MetadataStore()
    upload_id = store.cache_source(source_testkit.source)
    store.update_status(upload_id, "awaiting_upload")

    # Run the process, expecting it to fail
    with pytest.raises(MatchboxServerFileError) as excinfo:
        await process_upload(
            backend=mock_backend,
            upload_id=upload_id,
            bucket=bucket,
            key=test_key,
            metadata_store=store,
        )

    assert "Simulated processing failure" in str(excinfo.value)

    # Check that the status was updated to failed
    status = store.get(upload_id).status
    assert status.status == "failed", f"Expected status 'failed', got '{status.status}'"

    # Verify file was deleted despite the failure
    with pytest.raises(ClientError) as excinfo:
        s3.head_object(Bucket=bucket, Key=test_key)
    assert "404" in str(excinfo.value) or "NoSuchKey" in str(excinfo.value), (
        f"File was not deleted: {str(excinfo.value)}"
    )


# Retrieval


@patch("matchbox.server.base.BackendManager.get_backend")
def test_query(get_backend: Mock, test_client: TestClient):
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(
        return_value=pa.Table.from_pylist(
            [
                {"source_pk": "a", "id": 1},
                {"source_pk": "b", "id": 2},
            ],
            schema=SCHEMA_MB_IDS,
        )
    )
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = test_client.get(
        "/query",
        params={
            "full_name": "foo",
            "warehouse_hash_b64": hash_to_base64(b"bar"),
        },
    )

    # Process response
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.schema.equals(SCHEMA_MB_IDS)


@patch("matchbox.server.base.BackendManager.get_backend")
def test_query_404_resolution(get_backend: Mock, test_client: TestClient):
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(side_effect=MatchboxResolutionNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = test_client.get(
        "/query",
        params={
            "full_name": "foo",
            "warehouse_hash_b64": hash_to_base64(b"bar"),
        },
    )

    # Check response
    assert response.status_code == 404


@patch("matchbox.server.base.BackendManager.get_backend")
def test_query_404_source(get_backend: Mock, test_client: TestClient):
    # Mock backend
    mock_backend = Mock()
    mock_backend.query = Mock(side_effect=MatchboxSourceNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = test_client.get(
        "/query",
        params={
            "full_name": "foo",
            "warehouse_hash_b64": hash_to_base64(b"bar"),
        },
    )

    # Check response
    assert response.status_code == 404


@patch("matchbox.server.base.BackendManager.get_backend")
def test_match(get_backend: Mock, test_client: TestClient):
    # Mock backend
    mock_matches = [
        Match(
            cluster=1,
            source=SourceAddress(full_name="foo", warehouse_hash=b"foo"),
            source_id={"1"},
            target=SourceAddress(full_name="bar", warehouse_hash=b"bar"),
            target_id={"a"},
        )
    ]
    mock_backend = Mock()
    mock_backend.match = Mock(return_value=mock_matches)
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = test_client.get(
        "/match",
        params={
            "target_full_names": ["foo"],
            "target_warehouse_hashes_b64": [hash_to_base64(b"foo")],
            "source_full_name": "bar",
            "source_warehouse_hash_b64": hash_to_base64(b"bar"),
            "source_pk": 1,
            "resolution_name": "res",
            "threshold": 50,
        },
    )

    # Check response
    assert response.status_code == 200
    [Match.model_validate(m) for m in response.json()]


@patch("matchbox.server.base.BackendManager.get_backend")
def test_match_404_resolution(get_backend: Mock, test_client: TestClient):
    # Mock backend
    mock_backend = Mock()
    mock_backend.match = Mock(side_effect=MatchboxResolutionNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = test_client.get(
        "/match",
        params={
            "target_full_names": ["foo"],
            "target_warehouse_hashes_b64": [hash_to_base64(b"foo")],
            "source_full_name": "bar",
            "source_warehouse_hash_b64": hash_to_base64(b"bar"),
            "source_pk": 1,
            "resolution_name": "res",
        },
    )

    # Check response
    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.RESOLUTION


@patch("matchbox.server.base.BackendManager.get_backend")
def test_match_404_source(get_backend: Mock, test_client: TestClient):
    # Mock backend
    mock_backend = Mock()
    mock_backend.match = Mock(side_effect=MatchboxSourceNotFoundError())
    get_backend.return_value = mock_backend

    # Hit endpoint
    response = test_client.get(
        "/match",
        params={
            "target_full_names": ["foo"],
            "target_warehouse_hashes_b64": [hash_to_base64(b"foo")],
            "source_full_name": "bar",
            "source_warehouse_hash_b64": hash_to_base64(b"bar"),
            "source_pk": 1,
            "resolution_name": "res",
        },
    )

    # Check response
    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.SOURCE


# Data management


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_source(get_backend, test_client: TestClient):
    source_testkit = Source(
        address=SourceAddress(full_name="foo", warehouse_hash=b"bar"), db_pk="pk"
    )
    mock_backend = Mock()
    mock_backend.get_source = Mock(return_value=source_testkit)
    get_backend.return_value = mock_backend

    response = test_client.get(f"/sources/{hash_to_base64(b'bar')}/foo")
    assert response.status_code == 200
    assert Source.model_validate(response.json())


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_source_404(get_backend, test_client: TestClient):
    mock_backend = Mock()
    mock_backend.get_source = Mock(side_effect=MatchboxSourceNotFoundError)
    get_backend.return_value = mock_backend

    response = test_client.get(f"/sources/{hash_to_base64(b'bar')}/foo")
    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.SOURCE


@patch("matchbox.server.base.BackendManager.get_backend")
def test_add_source(get_backend: Mock, test_client: TestClient):
    """Test the source addition endpoint."""
    # Setup
    mock_backend = Mock()
    mock_backend.index = Mock(return_value=None)
    get_backend.return_value = mock_backend

    source_testkit = source_factory()

    # Make request
    response = test_client.post(
        "/sources",
        json=source_testkit.source.model_dump(),
    )

    # Validate response
    assert UploadStatus.model_validate(response.json())
    assert response.status_code == 202, response.json()
    assert response.json()["status"] == "awaiting_upload"
    assert response.json().get("id") is not None
    mock_backend.index.assert_not_called()


@pytest.mark.asyncio
@patch("matchbox.server.base.BackendManager.get_backend")
async def test_complete_source_upload_process(
    get_backend: Mock, s3: S3Client, test_client: TestClient
):
    """Test the complete upload process from source creation through processing."""
    # Setup the backend
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.index = Mock(return_value=None)
    get_backend.return_value = mock_backend

    # Create test bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data
    source_testkit = source_factory()

    # Step 1: Add source
    response = test_client.post("/sources", json=source_testkit.source.model_dump())
    assert response.status_code == 202
    upload_id = response.json()["id"]
    assert response.json()["status"] == "awaiting_upload"

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
    assert response.json()["status"] == "queued"

    # Step 3: Poll status until complete or timeout
    max_attempts = 10
    current_attempt = 0
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

    # Verify backend.index was called with correct arguments
    mock_backend.index.assert_called_once()
    call_args = mock_backend.index.call_args
    assert call_args[1]["source"] == source_testkit.source  # Check source matches
    assert call_args[1]["data_hashes"].equals(source_testkit.data_hashes)  # Check data

    # Verify file is deleted from S3 after processing
    with pytest.raises(ClientError):
        s3.head_object(Bucket="test-bucket", Key=f"{upload_id}.parquet")


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_resolution_graph(
    get_backend: MatchboxDBAdapter,
    resolution_graph: ResolutionGraph,
    test_client: TestClient,
):
    """Test the resolution graph report endpoint."""
    mock_backend = Mock()
    mock_backend.get_resolution_graph = Mock(return_value=resolution_graph)
    get_backend.return_value = mock_backend

    response = test_client.get("/report/resolutions")
    assert response.status_code == 200
    assert ResolutionGraph.model_validate(response.json())


# Model management


@patch("matchbox.server.base.BackendManager.get_backend")
def test_insert_model(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    get_backend.return_value = mock_backend

    testkit = model_factory(name="test_model")
    response = test_client.post("/models", json=testkit.model.metadata.model_dump())

    assert response.status_code == 201
    assert response.json() == {
        "success": True,
        "model_name": "test_model",
        "operation": ModelOperationType.INSERT.value,
        "details": None,
    }
    mock_backend.insert_model.assert_called_once_with(testkit.model.metadata)


@patch("matchbox.server.base.BackendManager.get_backend")
def test_insert_model_error(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    mock_backend.insert_model = Mock(side_effect=Exception("Test error"))
    get_backend.return_value = mock_backend

    testkit = model_factory()
    response = test_client.post("/models", json=testkit.model.metadata.model_dump())

    assert response.status_code == 500
    assert response.json()["success"] is False
    assert response.json()["details"] == "Test error"


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_model(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    testkit = model_factory(name="test_model", description="test description")
    mock_backend.get_model = Mock(return_value=testkit.model.metadata)
    get_backend.return_value = mock_backend

    response = test_client.get("/models/test_model")

    assert response.status_code == 200
    assert response.json()["name"] == testkit.model.metadata.name
    assert response.json()["description"] == testkit.model.metadata.description


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_model_not_found(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    mock_backend.get_model = Mock(side_effect=MatchboxResolutionNotFoundError())
    get_backend.return_value = mock_backend

    response = test_client.get("/models/nonexistent")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.RESOLUTION


@pytest.mark.parametrize("model_type", ["deduper", "linker"])
@patch("matchbox.server.base.BackendManager.get_backend")
@patch("matchbox.server.api.routes.metadata_store")
@patch("matchbox.server.api.routes.BackgroundTasks.add_task")
def test_model_upload(
    mock_add_task: Mock,
    metadata_store: Mock,
    get_backend: Mock,
    s3: S3Client,
    model_type: str,
    test_client: TestClient,
):
    """Test uploading different types of files."""
    # Setup
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    get_backend.return_value = mock_backend
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data with specified model type
    testkit = model_factory(model_type=model_type)

    # Setup metadata store
    store = MetadataStore()
    upload_id = store.cache_model(testkit.model.metadata)

    metadata_store.get.side_effect = store.get
    metadata_store.update_status.side_effect = store.update_status

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
@patch("matchbox.server.base.BackendManager.get_backend")
async def test_complete_model_upload_process(
    get_backend: Mock, s3: S3Client, model_type: str, test_client: TestClient
):
    """Test the complete upload process for models from creation through processing."""
    # Setup the backend
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.set_model_results = Mock(return_value=None)
    get_backend.return_value = mock_backend

    # Create test bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Create test data with specified model type
    testkit = model_factory(model_type=model_type)

    # Set up the mock to return the actual model metadata and data
    mock_backend.get_model = Mock(return_value=testkit.model.metadata)
    mock_backend.get_model_results = Mock(return_value=testkit.probabilities)

    # Step 1: Create model
    response = test_client.post("/models", json=testkit.model.metadata.model_dump())
    assert response.status_code == 201
    assert response.json()["success"] is True
    assert response.json()["model_name"] == testkit.model.metadata.name

    # Step 2: Initialize results upload
    response = test_client.post(f"/models/{testkit.model.metadata.name}/results")
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
        call_args[1]["model"] == testkit.model.metadata.name
    )  # Check model name matches
    assert call_args[1]["results"].equals(
        testkit.probabilities
    )  # Check results data matches

    # Step 6: Verify we can retrieve the results
    response = test_client.get(f"/models/{testkit.model.metadata.name}/results")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"

    # Step 7: Additional model-specific verifications
    if model_type == "linker":
        # For linker models, verify left and right resolutions are set
        assert testkit.model.metadata.left_resolution is not None
        assert testkit.model.metadata.right_resolution is not None
    else:
        # For deduper models, verify only left resolution is set
        assert testkit.model.metadata.left_resolution is not None
        assert testkit.model.metadata.right_resolution is None

    # Verify the model truth can be set and retrieved
    truth_value = 0.85
    mock_backend.get_model_truth = Mock(return_value=truth_value)

    response = test_client.patch(
        f"/models/{testkit.model.metadata.name}/truth",
        json=truth_value,
    )
    assert response.status_code == 200

    response = test_client.get(f"/models/{testkit.model.metadata.name}/truth")
    assert response.status_code == 200
    assert response.json() == truth_value

    # Verify file is deleted from S3 after processing
    with pytest.raises(ClientError):
        s3.head_object(Bucket="test-bucket", Key=f"{upload_id}.parquet")


@patch("matchbox.server.base.BackendManager.get_backend")
def test_set_results(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    testkit = model_factory()
    mock_backend.get_model = Mock(return_value=testkit.model.metadata)
    get_backend.return_value = mock_backend

    response = test_client.post(f"/models/{testkit.model.metadata.name}/results")

    assert response.status_code == 202
    assert response.json()["status"] == "awaiting_upload"


@patch("matchbox.server.base.BackendManager.get_backend")
def test_set_results_model_not_found(get_backend: Mock, test_client: TestClient):
    """Test setting results for a non-existent model."""
    mock_backend = Mock()
    mock_backend.get_model = Mock(side_effect=MatchboxResolutionNotFoundError())
    get_backend.return_value = mock_backend

    response = test_client.post("/models/nonexistent-model/results")

    assert response.status_code == 404
    assert response.json()["entity"] == BackendRetrievableType.RESOLUTION


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_results(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    testkit = model_factory()
    mock_backend.get_model_results = Mock(return_value=testkit.probabilities)
    get_backend.return_value = mock_backend

    response = test_client.get(f"/models/{testkit.model.metadata.name}/results")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"


@patch("matchbox.server.base.BackendManager.get_backend")
def test_set_truth(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    testkit = model_factory()
    get_backend.return_value = mock_backend

    response = test_client.patch(
        f"/models/{testkit.model.metadata.name}/truth", json=0.95
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    mock_backend.set_model_truth.assert_called_once_with(
        model=testkit.model.metadata.name, truth=0.95
    )


@patch("matchbox.server.base.BackendManager.get_backend")
def test_set_truth_invalid_value(get_backend: Mock, test_client: TestClient):
    """Test setting an invalid truth value (outside 0-1 range)."""
    mock_backend = Mock()
    testkit = model_factory()
    get_backend.return_value = mock_backend

    # Test value > 1
    response = test_client.patch(
        f"/models/{testkit.model.metadata.name}/truth", json=1.5
    )
    assert response.status_code == 422

    # Test value < 0
    response = test_client.patch(
        f"/models/{testkit.model.metadata.name}/truth", json=-0.5
    )
    assert response.status_code == 422


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_truth(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    testkit = model_factory()
    mock_backend.get_model_truth = Mock(return_value=0.95)
    get_backend.return_value = mock_backend

    response = test_client.get(f"/models/{testkit.model.metadata.name}/truth")

    assert response.status_code == 200
    assert response.json() == 0.95


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_ancestors(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    testkit = model_factory()
    mock_ancestors = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=97),
    ]
    mock_backend.get_model_ancestors = Mock(return_value=mock_ancestors)
    get_backend.return_value = mock_backend

    response = test_client.get(f"/models/{testkit.model.metadata.name}/ancestors")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert [ModelAncestor.model_validate(a) for a in response.json()] == mock_ancestors


@patch("matchbox.server.base.BackendManager.get_backend")
def test_get_ancestors_cache(get_backend: Mock, test_client: TestClient):
    """Test retrieving the ancestors cache for a model."""
    mock_backend = Mock()
    testkit = model_factory()
    mock_ancestors = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=80),
    ]
    mock_backend.get_model_ancestors_cache = Mock(return_value=mock_ancestors)
    get_backend.return_value = mock_backend

    response = test_client.get(f"/models/{testkit.model.metadata.name}/ancestors_cache")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert [ModelAncestor.model_validate(a) for a in response.json()] == mock_ancestors


@patch("matchbox.server.base.BackendManager.get_backend")
def test_set_ancestors_cache(get_backend: Mock, test_client: TestClient):
    """Test setting the ancestors cache for a model."""
    mock_backend = Mock()
    testkit = model_factory()
    get_backend.return_value = mock_backend

    ancestors_data = [
        ModelAncestor(name="parent_model", truth=70),
        ModelAncestor(name="grandparent_model", truth=80),
    ]

    response = test_client.patch(
        f"/models/{testkit.model.metadata.name}/ancestors_cache",
        json=[a.model_dump() for a in ancestors_data],
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == ModelOperationType.UPDATE_ANCESTOR_CACHE
    mock_backend.set_model_ancestors_cache.assert_called_once_with(
        model=testkit.model.metadata.name, ancestors_cache=ancestors_data
    )


@pytest.mark.parametrize(
    "endpoint",
    ["results", "truth", "ancestors", "ancestors_cache"],
)
@patch("matchbox.server.base.BackendManager.get_backend")
def test_model_get_endpoints_404(
    get_backend: Mock,
    endpoint: str,
    test_client: TestClient,
) -> None:
    """Test 404 responses for model GET endpoints when model doesn't exist."""
    # Setup backend mock
    mock_backend = Mock()
    mock_method = getattr(mock_backend, f"get_model_{endpoint}")
    mock_method.side_effect = MatchboxResolutionNotFoundError()
    get_backend.return_value = mock_backend

    # Make request
    response = test_client.get(f"/models/nonexistent-model/{endpoint}")

    # Verify response
    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendRetrievableType.RESOLUTION


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("truth", 0.95),
        (
            "ancestors_cache",
            [
                ModelAncestor(name="parent_model", truth=70).model_dump(),
                ModelAncestor(name="grandparent_model", truth=80).model_dump(),
            ],
        ),
    ],
)
@patch("matchbox.server.base.BackendManager.get_backend")
def test_model_patch_endpoints_404(
    get_backend: Mock,
    endpoint: str,
    payload: float | list[dict[str, Any]],
    test_client: TestClient,
) -> None:
    """Test 404 responses for model PATCH endpoints when model doesn't exist."""
    # Setup backend mock
    mock_backend = Mock()
    mock_method = getattr(mock_backend, f"set_model_{endpoint}")
    mock_method.side_effect = MatchboxResolutionNotFoundError()
    get_backend.return_value = mock_backend

    # Make request
    response = test_client.patch(f"/models/nonexistent-model/{endpoint}", json=payload)

    # Verify response
    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendRetrievableType.RESOLUTION


@patch("matchbox.server.base.BackendManager.get_backend")
def test_delete_model(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    get_backend.return_value = mock_backend

    testkit = model_factory()
    response = test_client.delete(
        f"/models/{testkit.model.metadata.name}",
        params={"certain": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "model_name": testkit.model.metadata.name,
        "operation": ModelOperationType.DELETE,
        "details": None,
    }


@patch("matchbox.server.base.BackendManager.get_backend")
def test_delete_model_needs_confirmation(get_backend: Mock, test_client: TestClient):
    mock_backend = Mock()
    mock_backend.delete_model = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=["dedupe1", "dedupe2"])
    )
    get_backend.return_value = mock_backend

    testkit = model_factory()
    response = test_client.delete(f"/models/{testkit.model.metadata.name}")

    assert response.status_code == 409
    assert response.json()["success"] is False
    message = response.json()["details"]
    assert "dedupe1" in message and "dedupe2" in message


@pytest.mark.parametrize(
    "certain",
    [True, False],
)
@patch("matchbox.server.base.BackendManager.get_backend")
def test_delete_model_404(
    get_backend: Mock, certain: bool, test_client: TestClient
) -> None:
    """Test 404 response when trying to delete a non-existent model."""
    # Setup backend mock
    mock_backend = Mock()
    mock_backend.delete_model.side_effect = MatchboxResolutionNotFoundError()
    get_backend.return_value = mock_backend

    # Make request
    response = test_client.delete(
        "/models/nonexistent-model", params={"certain": certain}
    )

    # Verify response
    assert response.status_code == 404
    error = NotFoundError.model_validate(response.json())
    assert error.entity == BackendRetrievableType.RESOLUTION


# Admin


@patch("matchbox.server.base.BackendManager.get_backend")
def test_count_all_backend_items(get_backend, test_client: TestClient):
    """Test the unparameterised entity counting endpoint."""
    entity_counts = {
        "datasets": 1,
        "models": 2,
        "data": 3,
        "clusters": 4,
        "creates": 5,
        "merges": 6,
        "proposes": 7,
    }
    mock_backend = Mock()
    for e, c in entity_counts.items():
        mock_e = Mock()
        mock_e.count = Mock(return_value=c)
        setattr(mock_backend, e, mock_e)
    get_backend.return_value = mock_backend

    response = test_client.get("/database/count")
    assert response.status_code == 200
    assert response.json() == {"entities": entity_counts}


@patch("matchbox.server.base.BackendManager.get_backend")
def test_count_backend_item(get_backend: MatchboxDBAdapter, test_client: TestClient):
    """Test the parameterised entity counting endpoint."""
    mock_backend = Mock()
    mock_backend.models.count = Mock(return_value=20)
    get_backend.return_value = mock_backend

    response = test_client.get("/database/count", params={"entity": "models"})
    assert response.status_code == 200
    assert response.json() == {"entities": {"models": 20}}


@patch("matchbox.server.base.BackendManager.get_backend")
def test_clear_backend_ok(get_backend: MatchboxDBAdapter, test_client: TestClient):
    mock_backend = Mock()
    mock_backend.clear = Mock()
    get_backend.return_value = mock_backend

    response = test_client.delete("/database", params={"certain": "true"})
    assert response.status_code == 200
    OKMessage.model_validate(response.json())


@patch("matchbox.server.base.BackendManager.get_backend")
def test_clear_backend_errors(get_backend: MatchboxDBAdapter, test_client: TestClient):
    mock_backend = Mock()
    mock_backend.clear = Mock(side_effect=MatchboxDeletionNotConfirmed)
    get_backend.return_value = mock_backend

    response = test_client.delete("/database")
    assert response.status_code == 409
    # We send some explanatory message
    assert response.content


@patch("matchbox.server.base.BackendManager.get_backend")
def test_api_key_authorisation(get_backend: MatchboxDBAdapter, test_client: TestClient):
    # Incorrect API Key Value
    test_client.headers["X-API-Key"] = "incorrect-api-key"

    response = test_client.post("/upload/upload_id")
    assert response.status_code == 401
    assert response.content == b'"API Key missing or invalid."'

    response = test_client.post("/sources")
    assert response.status_code == 401
    assert response.content == b'"API Key missing or invalid."'

    response = test_client.post("/models")
    assert response.status_code == 401
    assert response.content == b'"API Key missing or invalid."'

    response = response = test_client.patch("/models/model_name/truth")
    assert response.status_code == 401
    assert response.content == b'"API Key missing or invalid."'

    response = test_client.delete("/models/model_name")
    assert response.status_code == 401
    assert response.content == b'"API Key missing or invalid."'

    response = test_client.delete("/database")
    assert response.status_code == 401
    assert response.content == b'"API Key missing or invalid."'

    # Missing API Key Value
    test_client.headers.pop("X-API-Key")

    response = test_client.post("/upload/upload_id")
    assert response.status_code == 403
    assert response.content == b'"Not authenticated"'

    response = test_client.post("/sources")
    assert response.status_code == 403
    assert response.content == b'"Not authenticated"'

    response = test_client.post("/models")
    assert response.status_code == 403
    assert response.content == b'"Not authenticated"'

    response = response = test_client.patch("/models/model_name/truth")
    assert response.status_code == 403
    assert response.content == b'"Not authenticated"'

    response = test_client.delete("/models/model_name")
    assert response.status_code == 403
    assert response.content == b'"Not authenticated"'

    response = test_client.delete("/database")
    assert response.status_code == 403
    assert response.content == b'"Not authenticated"'
