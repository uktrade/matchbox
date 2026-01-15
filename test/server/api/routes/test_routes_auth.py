"""Tests for authentication routes."""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest
from _pytest.mark.structures import ParameterSet
from fastapi.testclient import TestClient

from matchbox.client._settings import settings
from matchbox.common.dtos import (
    AuthStatusResponse,
    ErrorResponse,
    LoginResponse,
    User,
)
from test.scripts.authorisation import generate_json_web_token


def test_login_first_user_setup_mode(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test that the first user login returns setup_mode_admin=True."""
    test_client, mock_backend, _ = api_client_and_mocks

    # First user should get setup_mode_admin=True
    mock_backend.login = Mock(
        return_value=LoginResponse(
            user=User(user_name="alice"),
            setup_mode_admin=True,
        )
    )

    response = test_client.post(
        "/auth/login",
        json=User(user_name="alice").model_dump(),
    )

    assert response.status_code == 200
    alice_admin = LoginResponse.model_validate(response.json())
    assert alice_admin.setup_mode_admin is True
    assert alice_admin.user.user_name == "alice"


def test_login_subsequent_users_normal_mode(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test that subsequent user logins return setup_mode_admin=False."""
    test_client, mock_backend, _ = api_client_and_mocks

    # Second and subsequent users should get setup_mode_admin=False
    mock_backend.login = Mock(
        return_value=LoginResponse(
            user=User(user_name="bob"),
            setup_mode_admin=False,
        )
    )

    response = test_client.post(
        "/auth/login",
        json=User(user_name="bob").model_dump(),
    )

    assert response.status_code == 200
    bob_user = LoginResponse.model_validate(response.json())
    assert bob_user.setup_mode_admin is False
    assert bob_user.user.user_name == "bob"


def test_login_with_email(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test login with email address."""
    test_client, mock_backend, _ = api_client_and_mocks

    user = User(user_name="alice", email="alice@example.com")
    mock_backend.login = Mock(
        return_value=LoginResponse(
            user=user,
            setup_mode_admin=False,
        )
    )

    response = test_client.post(
        "/auth/login",
        json=user.model_dump(),
    )

    assert response.status_code == 200
    result = LoginResponse.model_validate(response.json())
    assert result.user.user_name == "alice"
    assert result.user.email == "alice@example.com"


def test_login_existing_user(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test that logging in with an existing user returns the same user."""
    test_client, mock_backend, _ = api_client_and_mocks

    user = User(user_name="alice", email="alice@example.com")
    mock_backend.login = Mock(
        return_value=LoginResponse(
            user=user,
            setup_mode_admin=False,
        )
    )

    # First login
    response1 = test_client.post(
        "/auth/login",
        json=user.model_dump(),
    )
    assert response1.status_code == 200

    # Second login (same user)
    response2 = test_client.post(
        "/auth/login",
        json=user.model_dump(),
    )
    assert response2.status_code == 200

    result1 = LoginResponse.model_validate(response1.json())
    result2 = LoginResponse.model_validate(response2.json())
    assert result1.user.user_name == result2.user.user_name


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


PROTECTED_ROUTES: list[ParameterSet] = [
    pytest.param("get", "/admin/groups", id="router"),
    pytest.param("post", "/collections/default/runs", id="endpoint"),
]
"""Tests parameters for 'exemplar' authentication methods.

We already cover the various endpoints that are protected. These parameters are
to cover different families of authentication implementations we use, and are used 
to check different high level authentication methods, such as JWTs.

Examples of other routes we might add here would be:

* Routes using the Security class
* Routes that authenticate inline, for some reason
"""


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
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxAuthenticationError"
    assert error.message == "JWT invalid."


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
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxAuthenticationError"
    assert error.message == "JWT expired."


@pytest.mark.parametrize(("method_name", "url"), PROTECTED_ROUTES)
def test_missing_authorisation_header(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
    method_name: str,
    url: str,
) -> None:
    """Test that routes reject requests without authorisation headers.

    We assume the public user doesn't have permissions to see these routes.
    """
    test_client, mock_backend, _ = api_client_and_mocks

    # Public user not authorised
    mock_backend.check_permission.return_value = False

    # Authorisation headers gone
    test_client.headers.pop("Authorization", None)

    method: Callable[..., Any] = getattr(test_client, method_name)
    response = method(url)

    assert response.status_code == 403
    error = ErrorResponse.model_validate(response.json())
    assert error.exception_type == "MatchboxPermissionDenied"
    assert "Permission denied: requires" in error.message
