"""Unit tests for collection and run management endpoints."""

from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from matchbox.common.dtos import (
    Collection,
    GroupName,
    PermissionGrant,
    PermissionType,
    Run,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxRunNotFoundError,
)
from matchbox.common.factories.sources import source_factory

# Authorisation


@pytest.mark.parametrize(
    ["method", "endpoint"],
    [
        # Collection management (admin)
        pytest.param("DELETE", "/collections/col1", id="delete_collection"),
        pytest.param("GET", "/collections/col1/permissions", id="get_perms"),
        pytest.param("POST", "/collections/col1/permissions", id="grant_perm"),
        pytest.param(
            "DELETE", "/collections/col1/permissions/g/read", id="revoke_perm"
        ),
        # Run and resolution management (write)
        pytest.param("POST", "/collections/col1/runs", id="create_run"),
        pytest.param("DELETE", "/collections/col1/runs/1", id="delete_run"),
        pytest.param("PATCH", "/collections/col1/runs/1/mutable", id="set_mutable"),
        pytest.param(
            "DELETE", "/collections/col1/runs/1/resolutions/res1", id="delete_res"
        ),
    ],
)
def test_collection_routes_forbidden(
    method: str,
    endpoint: str,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Verify protected routes return 403 when permission check fails."""
    test_client, mock_backend, _ = api_client_and_mocks

    # Force permission check to fail
    mock_backend.check_permission.return_value = False

    response = test_client.request(method, endpoint)

    assert response.status_code == 403
    assert "Permission denied" in response.json()[0]
    mock_backend.check_permission.assert_called()


# Collection management tests


def test_list_collections(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test listing all collections."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.list_collections = Mock(return_value=["default", "test_collection"])

    response = test_client.get("/collections")

    assert response.status_code == 200
    assert response.json() == ["default", "test_collection"]
    mock_backend.list_collections.assert_called_once()


def test_get_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test retrieving a specific collection."""
    test_client, mock_backend, _ = api_client_and_mocks

    # Mock collection data with runs and resolutions
    collection = Collection(default_run=1, runs=[1, 2])

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
) -> None:
    """Test 404 responses when collection doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_method = getattr(mock_backend, backend_method)
    mock_method.side_effect = MatchboxCollectionNotFoundError()

    response = test_client.request(method, endpoint, params=extra_params)

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxCollectionNotFoundError"


def test_create_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test creating a new collection."""
    test_client, mock_backend, _ = api_client_and_mocks

    mock_backend.create_collection = Mock()

    permissions_payload = [
        {"group_name": "public", "permission": "read"},
        {"group_name": "public", "permission": "write"},
    ]

    response = test_client.post("/collections/new_collection", json=permissions_payload)

    assert response.status_code == 201
    assert response.json()["success"] is True
    mock_backend.create_collection.assert_called_once_with(
        name="new_collection",
        permissions=[
            PermissionGrant(
                group_name=GroupName("public"), permission=PermissionType.READ
            ),
            PermissionGrant(
                group_name=GroupName("public"), permission=PermissionType.WRITE
            ),
        ],
    )


def test_create_collection_with_custom_permissions(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test creating a collection with custom permissions."""
    test_client, mock_backend, _ = api_client_and_mocks

    mock_backend.create_collection = Mock()

    permissions_payload = [
        {"group_name": "admins", "permission": "admin"},
        {"group_name": "viewers", "permission": "read"},
    ]

    response = test_client.post(
        "/collections/custom_collection", json=permissions_payload
    )

    assert response.status_code == 201
    assert response.json()["success"] is True
    mock_backend.create_collection.assert_called_once_with(
        name="custom_collection",
        permissions=[
            PermissionGrant(
                group_name=GroupName("admins"), permission=PermissionType.ADMIN
            ),
            PermissionGrant(
                group_name=GroupName("viewers"), permission=PermissionType.READ
            ),
        ],
    )


def test_create_collection_with_empty_permissions(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test creating a collection with no permissions."""
    test_client, mock_backend, _ = api_client_and_mocks

    mock_backend.create_collection = Mock()

    response = test_client.post("/collections/no_perms_collection", json=[])

    assert response.status_code == 201
    assert response.json()["success"] is True
    mock_backend.create_collection.assert_called_once_with(
        name="no_perms_collection",
        permissions=[],
    )


def test_create_collection_already_exists(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test creating a collection that already exists."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.create_collection = Mock(side_effect=MatchboxCollectionAlreadyExists())

    permissions_payload = [
        {"group_name": "public", "permission": "read"},
    ]

    response = test_client.post(
        "/collections/existing_collection", json=permissions_payload
    )

    assert response.status_code == 409
    assert response.json()["exception_type"] == "MatchboxCollectionAlreadyExists"
    assert response.json()["message"] == "Collection already exists."


def test_delete_collection(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
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
) -> None:
    """Test deleting a collection that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_collection = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=[1, 2])
    )

    response = test_client.delete("/collections/test_collection")

    assert response.status_code == 409
    assert response.json()["exception_type"] == "MatchboxDeletionNotConfirmed"
    assert "1" in response.json()["message"]
    assert "2" in response.json()["message"]


# Collection permissions


def test_get_collection_permissions(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test getting permissions for a specific collection."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True
    mock_backend.get_permissions.return_value = [
        PermissionGrant(group_name="viewers", permission=PermissionType.READ)
    ]

    response = test_client.get("/collections/col1/permissions")

    assert response.status_code == 200
    assert response.json()[0]["permission"] == "read"
    mock_backend.get_permissions.assert_called_with("col1")


def test_grant_collection_permission(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test granting permission on a collection."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True

    payload = {"group_name": "g", "permission": "write"}
    response = test_client.post("/collections/col1/permissions", json=payload)

    assert response.status_code == 200
    mock_backend.grant_permission.assert_called_with("g", PermissionType.WRITE, "col1")


def test_revoke_collection_permission(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test revoking permission from a collection."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True

    response = test_client.delete("/collections/col1/permissions/write/g")

    assert response.status_code == 200
    mock_backend.revoke_permission.assert_called_with("g", PermissionType.WRITE, "col1")


def test_permission_collection_not_found(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 when operating on missing collection."""
    test_client, mock_backend, _ = api_client_and_mocks
    # Auth passes, but resource check fails inside service
    mock_backend.check_permission.return_value = True
    mock_backend.get_permissions.side_effect = MatchboxCollectionNotFoundError()

    response = test_client.get("/collections/missing/permissions")

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxCollectionNotFoundError"


# Run management tests


def test_get_run(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test retrieving a specific run."""
    test_client, mock_backend, _ = api_client_and_mocks

    source = source_factory().fake_run().source
    run = Run(
        run_id=1,
        resolutions={source.name: source.to_resolution()},
        is_default=True,
        is_mutable=False,
    )

    mock_backend.get_run = Mock(return_value=run)

    response = test_client.get("/collections/test_collection/runs/1")

    assert response.status_code == 200
    assert Run.model_validate(response.json())
    mock_backend.get_run.assert_called_once_with(collection="test_collection", run_id=1)


@pytest.mark.parametrize(
    [
        "exception_type",
        "expected_exception_name",
        "method",
        "endpoint",
        "backend_method",
        "extra_params",
    ],
    [
        pytest.param(
            MatchboxCollectionNotFoundError,
            "MatchboxCollectionNotFoundError",
            "GET",
            "/collections/nonexistent/runs/1",
            "get_run",
            {},
            id="get_run_collection_not_found",
        ),
        pytest.param(
            MatchboxRunNotFoundError,
            "MatchboxRunNotFoundError",
            "GET",
            "/collections/test_collection/runs/13",
            "get_run",
            {},
            id="get_run_run_not_found",
        ),
        pytest.param(
            MatchboxCollectionNotFoundError,
            "MatchboxCollectionNotFoundError",
            "DELETE",
            "/collections/nonexistent/runs/1",
            "delete_run",
            {"certain": True},
            id="delete_run_collection_not_found",
        ),
        pytest.param(
            MatchboxRunNotFoundError,
            "MatchboxRunNotFoundError",
            "DELETE",
            "/collections/test_collection/runs/13",
            "delete_run",
            {"certain": True},
            id="delete_run_run_not_found",
        ),
    ],
)
def test_run_endpoints_404(
    exception_type: type[Exception],
    expected_exception_name: str,
    method: str,
    endpoint: str,
    backend_method: str,
    extra_params: dict[str, Any],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 responses for run endpoints when resources don't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_method = getattr(mock_backend, backend_method)
    mock_method.side_effect = exception_type()

    json_data = {"run_id": 1} if method == "POST" else None
    response = test_client.request(
        method, endpoint, params=extra_params, json=json_data
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == expected_exception_name


def test_create_run(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test creating a new run."""
    test_client, mock_backend, _ = api_client_and_mocks

    source = source_factory().fake_run().source
    run = Run(
        run_id=1,
        resolutions={source.name: source.to_resolution()},
        is_default=False,
        is_mutable=False,
    )

    mock_backend.create_run = Mock(return_value=run)

    response = test_client.post("/collections/test_collection/runs")

    assert response.status_code == 201
    assert Run.model_validate(response.json())
    mock_backend.create_run.assert_called_once_with(collection="test_collection")


def test_set_run_mutable(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test setting run mutability."""
    test_client, mock_backend, _ = api_client_and_mocks

    updated_run = Run(run_id=1, is_default=False, is_mutable=False, resolutions={})
    mock_backend.set_run_mutable = Mock(return_value=updated_run)

    response = test_client.patch(
        "/collections/test_collection/runs/1/mutable", json=False
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == "update"
    mock_backend.set_run_mutable.assert_called_once_with(
        collection="test_collection", run_id=1, mutable=False
    )


def test_set_run_mutable_missing_flag(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test setting run mutability without providing the flag."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.patch("/collections/test_collection/runs/1/mutable", json={})

    assert response.status_code == 422


def test_set_run_default(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test setting run as default."""
    test_client, mock_backend, _ = api_client_and_mocks

    updated_run = Run(run_id=1, is_default=True, is_mutable=False, resolutions={})
    mock_backend.set_run_default = Mock(return_value=updated_run)

    response = test_client.patch(
        "/collections/test_collection/runs/1/default", json=True
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == "update"
    mock_backend.set_run_default.assert_called_once_with(
        collection="test_collection", run_id=1, default=True
    )


def test_set_run_default_missing_flag(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test setting run default without providing the flag."""
    test_client, _, _ = api_client_and_mocks

    response = test_client.patch("/collections/test_collection/runs/1/default", json={})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ["endpoint", "payload", "exception_type", "expected_exception_name"],
    [
        pytest.param(
            "mutable",
            True,
            MatchboxCollectionNotFoundError,
            "MatchboxCollectionNotFoundError",
        ),
        pytest.param(
            "mutable", True, MatchboxRunNotFoundError, "MatchboxRunNotFoundError"
        ),
        pytest.param(
            "default",
            False,
            MatchboxCollectionNotFoundError,
            "MatchboxCollectionNotFoundError",
        ),
        pytest.param(
            "default", False, MatchboxRunNotFoundError, "MatchboxRunNotFoundError"
        ),
    ],
)
def test_run_patch_endpoints_404(
    endpoint: str,
    payload: bool,
    exception_type: type[Exception],
    expected_exception_name: str,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 responses for run PATCH endpoints when resource doesn't exist."""
    test_client, mock_backend, _ = api_client_and_mocks
    method_name = f"set_run_{endpoint}"
    mock_method = getattr(mock_backend, method_name)
    mock_method.side_effect = exception_type()

    response = test_client.patch(
        f"/collections/test_collection/runs/1/{endpoint}", json=payload
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == expected_exception_name


def test_delete_run(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test deleting a run with confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_run = Mock()

    response = test_client.delete(
        "/collections/test_collection/runs/1", params={"certain": True}
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["operation"] == "delete"
    mock_backend.delete_run.assert_called_once_with(
        collection="test_collection", run_id=1, certain=True
    )


def test_delete_run_needs_confirmation(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test deleting a run that requires confirmation."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_run = Mock(
        side_effect=MatchboxDeletionNotConfirmed(children=[1, 2])
    )

    response = test_client.delete("/collections/test_collection/runs/1")

    assert response.status_code == 409
    assert response.json()["exception_type"] == "MatchboxDeletionNotConfirmed"
    assert "1" in response.json()["message"]
    assert "2" in response.json()["message"]
