"""Tests for authentication routes."""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from matchbox.client._settings import settings
from matchbox.common.dtos import (
    AuthStatusResponse,
    User,
)
from test.scripts.authorisation import generate_json_web_token


def test_login(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test the login endpoint at /auth/login."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.login = Mock(return_value=User(user_name="alice"))

    response = test_client.post(
        "/auth/login", json=User(user_name="alice").model_dump()
    )

    assert response.status_code == 200
    assert User.model_validate(response.json())


@pytest.mark.parametrize(
    ("has_token", "expected_authenticated"),
    [
        pytest.param(True, True, id="valid_token"),
        pytest.param(False, False, id="no_token"),
    ],
)
def test_auth_status(
    api_EdDSA_key_pair: tuple[bytes, bytes],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
    has_token: bool,
    expected_authenticated: bool,
) -> None:
    """Test auth status with and without valid JWT token."""
    test_client, _, mock_settings = api_client_and_mocks
    private_key, _ = api_EdDSA_key_pair
    mock_settings.authorisation = True

    username = "test.user@email.com"

    if has_token:
        token = generate_json_web_token(
            private_key_bytes=private_key,
            sub=username,
            api_root=settings.api_root,
        )
        test_client.headers["Authorization"] = token
    else:
        test_client.headers.pop("Authorization", None)

    response = test_client.get("/auth/status")

    assert response.status_code == 200
    result = AuthStatusResponse.model_validate(response.json())
    assert result.authenticated is expected_authenticated
    if expected_authenticated:
        assert result.username == username


PROTECTED_ROUTES = [
    pytest.param(
        "post",
        "/collections/default/runs/1/resolutions/name/data",
        id="post_resolution_data",
    ),
    pytest.param(
        "post", "/collections/default/runs/1/resolutions/name", id="post_resolution"
    ),
    pytest.param(
        "put", "/collections/default/runs/1/resolutions/name", id="put_resolution"
    ),
    pytest.param(
        "delete", "/collections/default/runs/1/resolutions/name", id="delete_resolution"
    ),
    pytest.param("delete", "/database", id="delete_database"),
]


@pytest.mark.parametrize(("method_name", "url"), PROTECTED_ROUTES)
def test_incorrect_signature(
    api_EdDSA_key_pair: tuple[bytes, bytes],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
    method_name: str,
    url: str,
) -> None:
    """Test that routes reject tokens with incorrect signatures."""
    test_client, _, _ = api_client_and_mocks
    private_key, _ = api_EdDSA_key_pair

    # Create token with wrong user but keep original signature
    _, _, signature_b64 = test_client.headers["Authorization"].encode().split(b".")
    header_b64, payload_64, _ = (
        generate_json_web_token(
            private_key_bytes=private_key,
            api_root=settings.api_root,
            sub="incorrect.user@email.com",
        )
        .encode()
        .split(b".")
    )
    test_client.headers["Authorization"] = b".".join(
        [header_b64, payload_64, signature_b64]
    ).decode()

    method: Callable[..., Any] = getattr(test_client, method_name)
    response = method(url)

    assert response.status_code == 401
    assert response.content == b'"JWT invalid."'


@pytest.mark.parametrize(("method_name", "url"), PROTECTED_ROUTES)
def test_expired_token(
    api_EdDSA_key_pair: tuple[bytes, bytes],
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
    method_name: str,
    url: str,
) -> None:
    """Test that routes reject expired tokens."""
    test_client, _, _ = api_client_and_mocks
    private_key, _ = api_EdDSA_key_pair

    test_client.headers["Authorization"] = generate_json_web_token(
        private_key_bytes=private_key,
        sub="test.user@email.com",
        api_root=settings.api_root,
        expiry_hours=-2,
    )

    method: Callable[..., Any] = getattr(test_client, method_name)
    response = method(url)

    assert response.status_code == 401
    assert response.content == b'"JWT expired."'


@pytest.mark.parametrize(("method_name", "url"), PROTECTED_ROUTES)
def test_missing_authorisation_header(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
    method_name: str,
    url: str,
) -> None:
    """Test that routes reject requests without authorisation headers."""
    test_client, _, _ = api_client_and_mocks

    test_client.headers.pop("Authorization", None)

    method: Callable[..., Any] = getattr(test_client, method_name)
    response = method(url)

    assert response.status_code == 401
    assert response.content == b'"JWT required but not provided."'
