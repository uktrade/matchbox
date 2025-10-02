"""Unit tests for collection and version management endpoints."""

from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from matchbox.common.dtos import (
    BackendResourceType,
    Collection,
    Version,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxVersionAlreadyExists,
    MatchboxVersionNotFoundError,
)
from matchbox.common.factories.sources import source_factory

# Collection management tests


def test_list_collections(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test listing all collections."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.list_collections = Mock(return_value=["default", "test_collection"])

    response = test_client.get("/collections")

    assert response.status_code == 200
    assert response.json() == ["default", "test_collection"]
    mock_backend.list_collections.assert_called_once()


def test_get_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test retrieving a specific collection."""
    test_client, mock_backend, _ = api_client_and_mocks

    # Mock collection data with versions and resolutions
    collection = Collection(default_version="v1", versions=["v1", "v2"])

    mock_backend.get_collection = Mock(return_value=collection)

    response = test_client.get("/collections/test_collection")

    assert response.status_code == 200
    assert Collection.model_validate(response.json())
    mock_backend.get_collection.assert_called_once_with(name="test_collection")


@pytest.mark.parametrize(
    ["method", "endpoint", "backend_method", "extra_params"],
    [
        pytest.param(
            "GET", "/collections/nonexistent", "get_collection", {}, id="get_collection"
        ),
        pytest.param(
            "DELETE",
            "/collections/nonexistent",
            "delete_collection",
            {"certain": True},
            id="delete_collection",
        ),
    ],
)
def test_collection_not_found_404(
    method: str,
    endpoint: str,
    backend_method: str,
    extra_params: dict[str, bool],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test 404 responses when collection doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_method = getattr(mock_backend, backend_method)
    mock_method.side_effect = MatchboxCollectionNotFoundError()

    response = test_client.request(method, endpoint, params=extra_params)

    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.COLLECTION


def test_create_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test creating a new collection."""
    test_client, mock_backend, _ = api_client_and_mocks

    mock_backend.create_collection = Mock()

    response = test_client.post("/collections/new_collection")

    assert response.status_code == 201
    assert response.json()["success"] is True
    mock_backend.create_collection.assert_called_once_with(name="new_collection")


def test_create_collection_already_exists(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test creating a collection that already exists."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.create_collection = Mock(side_effect=MatchboxCollectionAlreadyExists())

    response = test_client.post("/collections/existing_collection")

    assert response.status_code == 409
    assert response.json()["success"] is False
    assert response.json()["operation"] == "create"
    assert response.json()["details"] == "Collection already exists."


def test_delete_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deleting a collection with confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_collection = Mock()

    response = test_client.delete(
        "/collections/test_collection", params={"certain": True}
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == "delete"
    mock_backend.delete_collection.assert_called_once_with(
        name="test_collection", certain=True
    )


def test_delete_collection_needs_confirmation(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test deleting a collection that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_collection = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=["v1", "v2"])
    )

    response = test_client.delete("/collections/test_collection")

    assert response.status_code == 409
    assert response.json()["success"] is False
    assert response.json()["operation"] == "delete"
    assert "v1" in response.json()["details"]
    assert "v2" in response.json()["details"]


# Version management tests


def test_get_version(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test retrieving a specific version."""
    test_client, mock_backend, _ = api_client_and_mocks

    source = source_factory().source
    version = Version(
        name="v1",
        resolutions={source.name: source.to_resolution()},
        is_default=True,
        is_mutable=False,
    )

    mock_backend.get_version = Mock(return_value=version)

    response = test_client.get("/collections/test_collection/versions/v1")

    assert response.status_code == 200
    assert Version.model_validate(response.json())
    mock_backend.get_version.assert_called_once_with(
        collection="test_collection", name="v1"
    )


@pytest.mark.parametrize(
    [
        "exception_type",
        "expected_entity",
        "method",
        "endpoint",
        "backend_method",
        "extra_params",
    ],
    [
        pytest.param(
            MatchboxCollectionNotFoundError,
            BackendResourceType.COLLECTION,
            "GET",
            "/collections/nonexistent/versions/v1",
            "get_version",
            {},
        ),
        pytest.param(
            MatchboxVersionNotFoundError,
            BackendResourceType.VERSION,
            "GET",
            "/collections/test_collection/versions/nonexistent",
            "get_version",
            {},
        ),
        pytest.param(
            MatchboxCollectionNotFoundError,
            BackendResourceType.COLLECTION,
            "DELETE",
            "/collections/nonexistent/versions/v1",
            "delete_version",
            {"certain": True},
        ),
        pytest.param(
            MatchboxVersionNotFoundError,
            BackendResourceType.VERSION,
            "DELETE",
            "/collections/test_collection/versions/nonexistent",
            "delete_version",
            {"certain": True},
        ),
    ],
)
def test_version_endpoints_404(
    exception_type: type[Exception],
    expected_entity: BackendResourceType,
    method: str,
    endpoint: str,
    backend_method: str,
    extra_params: dict[str, Any],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test 404 responses for version endpoints when resources don't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_method = getattr(mock_backend, backend_method)
    mock_method.side_effect = exception_type()

    json_data = {"name": "v1"} if method == "POST" else None
    response = test_client.request(
        method, endpoint, params=extra_params, json=json_data
    )

    assert response.status_code == 404
    assert response.json()["entity"] == expected_entity


def test_create_version(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test creating a new version."""
    test_client, mock_backend, _ = api_client_and_mocks

    mock_backend.create_version = Mock()

    response = test_client.post("/collections/test_collection/versions/v2")

    assert response.status_code == 201
    assert response.json()["success"] is True
    mock_backend.create_version.assert_called_once_with(
        collection="test_collection", name="v2"
    )


def test_create_version_already_exists(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test creating a version that already exists."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.create_version = Mock(side_effect=MatchboxVersionAlreadyExists())

    response = test_client.post("/collections/test_collection/versions/v1")

    assert response.status_code == 409
    assert response.json()["success"] is False
    assert response.json()["operation"] == "create"
    assert response.json()["details"] == "Version already exists."


def test_set_version_mutable(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test setting version mutability."""
    test_client, mock_backend, _ = api_client_and_mocks

    updated_version = Version(
        name="v1", is_default=False, is_mutable=False, resolutions={}
    )
    mock_backend.set_version_mutable = Mock(return_value=updated_version)

    response = test_client.patch(
        "/collections/test_collection/versions/v1/mutable", json=False
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == "update"
    mock_backend.set_version_mutable.assert_called_once_with(
        collection="test_collection", name="v1", mutable=False
    )


def test_set_version_mutable_missing_flag(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test setting version mutability without providing the flag."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.patch(
        "/collections/test_collection/versions/v1/mutable", json={}
    )

    assert response.status_code == 422


def test_set_version_default(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test setting version as default."""
    test_client, mock_backend, _ = api_client_and_mocks

    updated_version = Version(
        name="v1", is_default=True, is_mutable=False, resolutions={}
    )
    mock_backend.set_version_default = Mock(return_value=updated_version)

    response = test_client.patch(
        "/collections/test_collection/versions/v1/default", json=True
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == "update"
    mock_backend.set_version_default.assert_called_once_with(
        collection="test_collection", name="v1", default=True
    )


def test_set_version_default_missing_flag(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test setting version default without providing the flag."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.patch(
        "/collections/test_collection/versions/v1/default", json={}
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ["endpoint", "payload", "exception_type", "expected_entity"],
    [
        pytest.param(
            "mutable",
            True,
            MatchboxCollectionNotFoundError,
            BackendResourceType.COLLECTION,
        ),
        pytest.param(
            "mutable", True, MatchboxVersionNotFoundError, BackendResourceType.VERSION
        ),
        pytest.param(
            "default",
            False,
            MatchboxCollectionNotFoundError,
            BackendResourceType.COLLECTION,
        ),
        pytest.param(
            "default", False, MatchboxVersionNotFoundError, BackendResourceType.VERSION
        ),
    ],
)
def test_version_patch_endpoints_404(
    endpoint: str,
    payload: bool,
    exception_type: type[Exception],
    expected_entity: BackendResourceType,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test 404 responses for version PATCH endpoints when resource doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    method_name = f"set_version_{endpoint}"
    mock_method = getattr(mock_backend, method_name)
    mock_method.side_effect = exception_type()

    response = test_client.patch(
        f"/collections/test_collection/versions/v1/{endpoint}", json=payload
    )

    assert response.status_code == 404
    assert response.json()["entity"] == expected_entity


def test_delete_version(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test deleting a version with confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_version = Mock()

    response = test_client.delete(
        "/collections/test_collection/versions/v1", params={"certain": True}
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == "delete"
    mock_backend.delete_version.assert_called_once_with(
        collection="test_collection", name="v1", certain=True
    )


def test_delete_version_needs_confirmation(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test deleting a version that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_version = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=["v1", "v2"])
    )

    response = test_client.delete("/collections/test_collection/versions/v1")

    assert response.status_code == 409
    assert response.json()["success"] is False
    assert response.json()["operation"] == "delete"
    assert "v1" in response.json()["details"]
    assert "v2" in response.json()["details"]
