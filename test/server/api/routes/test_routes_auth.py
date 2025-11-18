"""Tests for authentication routes."""

from unittest.mock import Mock

from fastapi.testclient import TestClient

from matchbox.client._settings import settings
from matchbox.common.dtos import AuthStatusResponse, LoginAttempt, LoginResult
from test.scripts.authorisation import generate_json_web_token


def test_login(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test the login endpoint at /auth/login."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.login = Mock(return_value=42)

    response = test_client.post(
        "/auth/login", json=LoginAttempt(user_name="alice").model_dump()
    )

    assert response.status_code == 200
    result = LoginResult.model_validate(response.json())
    assert result.user_id == 42


def test_auth_status_valid_token(
    api_EdDSA_key_pair: tuple[bytes, bytes],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test auth status with valid JWT token."""
    test_client, _, mock_settings = api_client_and_mocks
    private_key, _ = api_EdDSA_key_pair
    mock_settings.authorisation = True

    username = "test.user@email.com"
    token = generate_json_web_token(
        private_key_bytes=private_key,
        sub=username,
        api_root=settings.api_root,
    )
    test_client.headers["Authorization"] = token

    response = test_client.get("/auth/status")

    assert response.status_code == 200
    result = AuthStatusResponse.model_validate(response.json())
    assert result.authenticated is True
    assert result.username == username


def test_auth_status_no_token(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test auth status when no token is provided."""
    test_client, _, mock_settings = api_client_and_mocks
    mock_settings.authorisation = True

    test_client.headers.pop("Authorization", None)

    response = test_client.get("/auth/status")

    assert response.status_code == 200
    result = AuthStatusResponse.model_validate(response.json())
    assert result.authenticated is False
