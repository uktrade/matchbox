from importlib.metadata import version
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, call, patch

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from matchbox.client.authorisation import (
    generate_json_web_token,
)
from matchbox.common.arrow import SCHEMA_QUERY, table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    LoginAttempt,
    LoginResult,
    OKMessage,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.common.graph import ResolutionGraph
from matchbox.common.sources import Match

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


# General


def test_healthcheck(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test the healthcheck endpoint."""
    test_client, _, _ = api_client_and_mocks
    response = test_client.get("/health")
    assert response.status_code == 200
    response = OKMessage.model_validate(response.json())
    assert response.status == "OK"
    assert response.version == version("matchbox-db")


def test_login(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test the login endpoint."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.login = Mock(return_value=1)

    response = test_client.post(
        "/login", json=LoginAttempt(user_name="alice").model_dump()
    )

    assert response.status_code == 200
    response = LoginResult.model_validate(response.json())
    assert response.user_id == 1


# We can patch BackgroundTasks as the api_client_and_mocks fixture
# ensures the API runs the task (not Celery)
@patch("matchbox.server.api.main.BackgroundTasks.add_task")
def test_upload(
    mock_add_task: Mock,
    s3: S3Client,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test uploading a file, happy path."""
    # Setup
    test_client, mock_backend, mock_tracker = api_client_and_mocks

    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"
    mock_backend.index = Mock(return_value=None)
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    source_testkit = source_factory()

    update_id = mock_tracker.add_source(source_testkit.source_config)

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
    assert (
        response.json()["stage"] == UploadStage.QUEUED
    )  # Updated to check for queued status
    # Check both status updates were called in correct order
    assert mock_tracker.update.call_args_list == [
        call(update_id, UploadStage.QUEUED),
    ]
    mock_backend.index.assert_not_called()  # Index happens in background
    mock_add_task.assert_called_once()  # Verify task was queued


# We can patch BackgroundTasks as the api_client_and_mocks fixture
# ensures the API runs the task (not Celery)
@patch("matchbox.server.api.main.BackgroundTasks.add_task")
def test_upload_wrong_schema(
    mock_add_task: Mock,
    s3: S3Client,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test uploading a file with wrong schema."""
    test_client, mock_backend, mock_tracker = api_client_and_mocks
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.settings.datastore.cache_bucket_name = "test-bucket"

    # Create source with results schema instead of index
    source_testkit = source_factory()
    update_id = mock_tracker.add_source(source_testkit.source_config)

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

    # Should fail before task starts
    assert response.status_code == 400
    assert response.json()["stage"] == UploadStage.FAILED
    assert "schema mismatch" in response.json()["details"].lower()
    mock_add_task.assert_not_called()


def test_upload_status_check(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test checking status of an upload using the status endpoint."""
    test_client, _, mock_tracker = api_client_and_mocks
    source_testkit = source_factory()
    update_id = mock_tracker.add_source(source_testkit.source_config)
    mock_tracker.update(update_id, UploadStage.PROCESSING)
    mock_tracker.reset_mock()

    response = test_client.get(f"/upload/{update_id}/status")

    # Should return current status
    assert response.status_code == 200
    assert response.json()["stage"] == UploadStage.PROCESSING
    mock_tracker.update.assert_not_called()


def test_upload_already_processing(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test attempting to upload when status is already processing."""
    test_client, _, mock_tracker = api_client_and_mocks
    source_testkit = source_factory()
    update_id = mock_tracker.add_source(source_testkit.source_config)
    mock_tracker.update(update_id, UploadStage.PROCESSING)

    response = test_client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["stage"] == UploadStage.PROCESSING


def test_upload_already_queued(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test attempting to upload when status is already queued."""
    test_client, _, mock_tracker = api_client_and_mocks
    source_testkit = source_factory()
    update_id = mock_tracker.add_source(source_testkit.source_config)
    mock_tracker.update(update_id, UploadStage.QUEUED)

    response = test_client.post(
        f"/upload/{update_id}",
        files={"file": ("test.parquet", b"dummy data", "application/octet-stream")},
    )

    # Should return 400 with current status
    assert response.status_code == 400
    assert response.json()["stage"] == UploadStage.QUEUED


def test_status_check_not_found(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test checking status for non-existent upload ID."""
    test_client, _, mock_tracker = api_client_and_mocks
    mock_tracker.get.return_value = None

    response = test_client.get("/upload/nonexistent-id/status")

    assert response.status_code == 400
    assert response.json()["stage"] == "unknown"
    assert "not found or expired" in response.json()["details"].lower()


# Retrieval


def test_query(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.query = Mock(
        return_value=pa.Table.from_pylist(
            [
                {"keys": "a", "id": 1},
                {"keys": "b", "id": 2},
            ],
            schema=SCHEMA_QUERY,
        )
    )

    response = test_client.get(
        "/query",
        params={"source": "foo", "return_leaf_id": False},
    )

    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    assert response.status_code == 200
    assert table.schema.equals(SCHEMA_QUERY)


def test_query_404_resolution(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.query = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.get(
        "/query",
        params={"source": "foo", "resolution": "bar", "return_leaf_id": True},
    )

    assert response.status_code == 404


def test_query_404_source(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.query = Mock(side_effect=MatchboxSourceNotFoundError())

    response = test_client.get(
        "/query",
        params={"source": "foo", "return_leaf_id": True},
    )

    assert response.status_code == 404


def test_match(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_matches = [
        Match(
            cluster=1,
            source="foo",
            source_id={"1"},
            target="bar",
            target_id={"a"},
        )
    ]
    mock_backend.match = Mock(return_value=mock_matches)

    response = test_client.get(
        "/match",
        params={
            "targets": "foo",
            "source": "bar",
            "key": 1,
            "resolution": "res",
            "threshold": 50,
        },
    )

    assert response.status_code == 200
    [Match.model_validate(m) for m in response.json()]


def test_match_404_resolution(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.match = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.get(
        "/match",
        params={
            "targets": ["foo"],
            "source": "bar",
            "key": 1,
            "resolution": "res",
        },
    )

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION


def test_match_404_source(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.match = Mock(side_effect=MatchboxSourceNotFoundError())

    response = test_client.get(
        "/match",
        params={
            "targets": ["foo"],
            "source": "bar",
            "key": 1,
            "resolution": "res",
        },
    )

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.SOURCE


# Admin


def test_count_all_backend_items(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test the unparameterised entity counting endpoint."""
    test_client, mock_backend, _ = api_client_and_mocks
    entity_counts = {
        "sources": 1,
        "models": 2,
        "data": 3,
        "clusters": 4,
        "creates": 5,
        "merges": 6,
        "proposes": 7,
    }
    for e, c in entity_counts.items():
        mock_e = Mock()
        mock_e.count = Mock(return_value=c)
        setattr(mock_backend, e, mock_e)

    response = test_client.get("/database/count")
    assert response.status_code == 200
    assert response.json() == {"entities": entity_counts}


def test_count_backend_item(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test the parameterised entity counting endpoint."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.models.count = Mock(return_value=20)

    response = test_client.get("/database/count", params={"entity": "models"})
    assert response.status_code == 200
    assert response.json() == {"entities": {"models": 20}}


def test_clear_backend_ok(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.clear = Mock()

    response = test_client.delete("/database", params={"certain": "true"})
    assert response.status_code == 200
    OKMessage.model_validate(response.json())


def test_clear_backend_errors(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.clear = Mock(side_effect=MatchboxDeletionNotConfirmed)

    response = test_client.delete("/database")
    assert response.status_code == 409
    # We send some explanatory message
    assert response.content


def test_api_key_authorisation(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    test_client, _, _ = api_client_and_mocks
    routes = [
        (test_client.post, "/upload/upload_id"),
        (test_client.post, "/sources"),
        (test_client.post, "/models"),
        (test_client.patch, "/models/name/truth"),
        (test_client.delete, "/resolutions/name"),
        (test_client.delete, "/database"),
    ]

    # Incorrect signature
    _, _, signature_b64 = test_client.headers["Authorization"].encode().split(b".")
    header_b64, payload_64, _ = (
        generate_json_web_token(sub="incorrect.user@email.com").encode().split(b".")
    )
    test_client.headers["Authorization"] = b".".join(
        [header_b64, payload_64, signature_b64]
    ).decode()

    for method, url in routes:
        response = method(url)
        assert response.status_code == 401
        assert response.content == b'"JWT invalid."'

    # Expired JWT
    with patch("matchbox.client.authorisation.EXPIRY_AFTER_X_HOURS", -2):
        test_client.headers["Authorization"] = generate_json_web_token(
            sub="test.user@email.com"
        )
        for method, url in routes:
            response = method(url)
            assert response.status_code == 401
            assert response.content == b'"JWT expired."'

    # Missing Authorization header
    test_client.headers.pop("Authorization")
    for method, url in routes:
        response = method(url)
        assert response.status_code == 401
        assert response.content == b'"JWT required but not provided."'


def test_get_resolution_graph(
    resolution_graph: ResolutionGraph,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test the resolution graph report endpoint."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_resolution_graph = Mock(return_value=resolution_graph)

    response = test_client.get("/report/resolutions")
    assert response.status_code == 200
    assert ResolutionGraph.model_validate(response.json())
