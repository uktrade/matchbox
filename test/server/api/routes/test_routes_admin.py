"""Unit tests for admin management endpoints."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from matchbox.common.dtos import (
    CRUDOperation,
    ErrorResponse,
    Group,
)
from matchbox.common.exceptions import (
    MatchboxGroupAlreadyExistsError,
    MatchboxGroupNotFoundError,
)

# Group management


def test_list_groups(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test listing groups."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True
    mock_backend.list_groups.return_value = [
        Group(name="admins", is_system=True),
        Group(name="analysts"),
    ]

    response = test_client.get("/admin/groups")

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["name"] == "admins"


def test_create_group(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test creating a group."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True

    payload = Group(name="new_group", description="desc").model_dump()
    response = test_client.post("/admin/groups", json=payload)

    assert response.status_code == 201
    assert response.json()["operation"] == CRUDOperation.CREATE
    mock_backend.create_group.assert_called_once()


def test_create_group_conflict(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 409 when group already exists."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True
    mock_backend.create_group.side_effect = MatchboxGroupAlreadyExistsError("Exists")

    response = test_client.post("/admin/groups", json={"name": "exists"})

    assert response.status_code == 409
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxGroupAlreadyExistsError"
    assert "Exists" in error.message


def test_get_group(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test getting a single group."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True
    mock_backend.get_group.return_value = Group(name="g")

    response = test_client.get("/admin/groups/g")

    assert response.status_code == 200
    assert response.json()["name"] == "g"


def test_get_group_not_found(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 when group not found."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True
    mock_backend.get_group.side_effect = MatchboxGroupNotFoundError("Missing")

    response = test_client.get("/admin/groups/missing")

    assert response.status_code == 404


def test_delete_group(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test deleting a group."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True

    response = test_client.delete("/admin/groups/g", params={"certain": True})

    assert response.status_code == 200
    mock_backend.delete_group.assert_called_with("g", certain=True)


# Membership


def test_add_member(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test adding a member."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True

    response = test_client.post("/admin/groups/g/members", json={"sub": "user1"})

    assert response.status_code == 201
    mock_backend.add_user_to_group.assert_called_with("user1", "g")


def test_add_member_group_not_found(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404 when adding member to missing group."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True
    mock_backend.add_user_to_group.side_effect = MatchboxGroupNotFoundError()

    response = test_client.post("/admin/groups/missing/members", json={"sub": "user1"})

    assert response.status_code == 404


def test_remove_member(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test removing a member."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True

    response = test_client.delete("/admin/groups/g/members/user1")

    assert response.status_code == 200
    mock_backend.remove_user_from_group.assert_called_with("user1", "g")


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(MatchboxGroupNotFoundError(), id="group404"),
        pytest.param(MatchboxGroupNotFoundError(), id="user404"),
    ],
)
def test_remove_member_errors(
    exception: Exception,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test 404s for removing members."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.check_permission.return_value = True
    mock_backend.remove_user_from_group.side_effect = exception

    response = test_client.delete("/admin/groups/g/members/u")

    assert response.status_code == 404
