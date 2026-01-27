"""Tests for authentication routes and dependencies."""

from collections.abc import Callable
from http import HTTPMethod
from typing import Any
from unittest.mock import Mock

import pytest
from _pytest.mark.structures import ParameterSet
from fastapi import HTTPException, Request
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader
from fastapi.testclient import TestClient

from matchbox.client._settings import settings
from matchbox.common.dtos import (
    AuthStatusResponse,
    BackendResourceType,
    DefaultUser,
    ErrorResponse,
    LoginResponse,
    PermissionType,
    User,
)
from matchbox.common.exceptions import (
    MatchboxAuthenticationError,
    MatchboxPermissionDenied,
)
from matchbox.server.api.dependencies import (
    RequiresPermission,
    get_current_user,
    validate_jwt,
)
from matchbox.server.api.main import app
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
    test_client, mock_backend, mock_settings = api_client_and_mocks
    private_key, _ = api_EdDSA_key_pair
    mock_settings.authorisation = True

    user = User(user_name="alice", email="alice@example.com")
    mock_backend.login = Mock(
        return_value=LoginResponse(
            user=user,
            setup_mode_admin=False,
        )
    )

    if has_token:
        token = generate_json_web_token(
            private_key_bytes=private_key,
            sub=user.user_name,
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
        assert result.user is not None
        assert result.user.user_name == user.user_name
    else:
        assert result.user is not None
        assert result.user.user_name == DefaultUser.PUBLIC


PROTECTED_ROUTES: list[ParameterSet] = [
    pytest.param("get", "/admin/groups", id="router"),
    pytest.param("post", "/collections/default/runs", id="endpoint"),
]
"""Tests parameters for 'exemplar' authentication methods."""


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
    """Test that routes reject requests without authorisation headers."""
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


# RequiresPermission dependency


class TestRequiresPermission:
    """Unit tests for the RequiresPermission dependency."""

    def test_permission_granted_static_resource(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test permission granted for a static resource."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = True
        mock_backend.check_permission.return_value = True
        user = User(user_name="alice")

        dependency = RequiresPermission(
            PermissionType.READ, resource=BackendResourceType.SYSTEM
        )

        result = dependency(Mock(spec=Request), mock_backend, mock_settings, user)

        assert result == user
        mock_backend.check_permission.assert_called_with(
            "alice", PermissionType.READ, BackendResourceType.SYSTEM
        )

    def test_permission_granted_dynamic_path_param(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test permission granted using a resource from path parameters."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = True
        mock_backend.check_permission.return_value = True
        user = User(user_name="alice")

        dependency = RequiresPermission(
            PermissionType.WRITE, resource_from_param=BackendResourceType.COLLECTION
        )

        # Mock request with path params
        request = Mock(spec=Request)
        request.path_params = {"collection": "my_collection"}
        request.query_params = {}

        result = dependency(request, mock_backend, mock_settings, user)

        assert result == user
        mock_backend.check_permission.assert_called_with(
            "alice", PermissionType.WRITE, "my_collection"
        )

    def test_permission_granted_dynamic_query_param(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test permission granted using a resource from query parameters."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = True
        mock_backend.check_permission.return_value = True
        user = User(user_name="alice")

        dependency = RequiresPermission(
            PermissionType.WRITE, resource_from_param=BackendResourceType.COLLECTION
        )

        # Mock request with query params
        request = Mock(spec=Request)
        request.path_params = {}
        request.query_params = {"collection": "my_collection"}

        result = dependency(request, mock_backend, mock_settings, user)

        assert result == user
        mock_backend.check_permission.assert_called_with(
            "alice", PermissionType.WRITE, "my_collection"
        )

    def test_permission_denied(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test that MatchboxPermissionDenied is raised when check fails."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = True
        mock_backend.check_permission.return_value = False
        user = User(user_name="alice")

        dependency = RequiresPermission(
            PermissionType.READ, resource=BackendResourceType.SYSTEM
        )

        with pytest.raises(MatchboxPermissionDenied) as exc:
            dependency(Mock(spec=Request), mock_backend, mock_settings, user)

        assert exc.value.permission == PermissionType.READ
        assert exc.value.resource_type == BackendResourceType.SYSTEM

    def test_missing_resource_param_raises_500(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test that missing dynamic resource parameter raises 500."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = True
        user = User(user_name="alice")

        dependency = RequiresPermission(
            PermissionType.READ, resource_from_param=BackendResourceType.COLLECTION
        )

        # Request missing the 'collection' param
        request = Mock(spec=Request)
        request.path_params = {}
        request.query_params = {}

        with pytest.raises(HTTPException) as exc:
            dependency(request, mock_backend, mock_settings, user)

        assert exc.value.status_code == 500
        assert "Authorisation failed: Resource parameter" in exc.value.detail[0]

    def test_auth_disabled_short_circuit(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test that permission check is skipped if authorisation is disabled."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = False
        user = User(user_name="alice")

        dependency = RequiresPermission(
            PermissionType.READ, resource=BackendResourceType.SYSTEM
        )

        result = dependency(Mock(spec=Request), mock_backend, mock_settings, user)

        assert result == user
        mock_backend.check_permission.assert_not_called()

    def test_public_forbidden_when_auth_disabled(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test error when allow_public=False but server auth is disabled."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = False
        user = User(user_name="alice")

        dependency = RequiresPermission(
            PermissionType.READ,
            resource=BackendResourceType.SYSTEM,
            allow_public=False,
        )

        with pytest.raises(MatchboxAuthenticationError) as exc:
            dependency(Mock(spec=Request), mock_backend, mock_settings, user)

        assert "authentication is disabled" in str(exc.value)

    def test_public_forbidden_for_unauthenticated_user(
        self, api_client_and_mocks: tuple[TestClient, Mock, Mock]
    ) -> None:
        """Test error when allow_public=False and user is not authenticated."""
        _, mock_backend, mock_settings = api_client_and_mocks
        mock_settings.authorisation = True
        public_user = User(user_name=DefaultUser.PUBLIC)

        dependency = RequiresPermission(
            PermissionType.READ,
            resource=BackendResourceType.SYSTEM,
            allow_public=False,
        )

        with pytest.raises(MatchboxAuthenticationError):
            dependency(Mock(spec=Request), mock_backend, mock_settings, public_user)


# Route auth coveraage

# Routes that are explicitly allowed to be public (unauthenticated).
# Any route NOT in this list MUST have an authentication dependency.
PUBLIC_ROUTES: set[tuple[HTTPMethod, str]] = {
    (HTTPMethod.GET, "/health"),
    (HTTPMethod.GET, "/docs"),
    (HTTPMethod.GET, "/docs/oauth2-redirect"),
    (HTTPMethod.GET, "/openapi.json"),
    (HTTPMethod.POST, "/auth/login"),
    (HTTPMethod.GET, "/collections"),
    (HTTPMethod.POST, "/collections/{collection}"),
    (HTTPMethod.GET, "/eval/judgements"),
}


def is_auth_mechanism(
    call: RequiresPermission | APIKeyHeader | Callable[..., object],
) -> bool:
    """Check if a dependency callable is a recognised authentication mechanism."""
    # 1. Class-based permission checks (e.g. RequireSysAdmin, RequireCollectionRead)
    if isinstance(call, RequiresPermission):
        return True

    # 2. Function-based user retrieval (e.g. CurrentUserDependency)
    if call is get_current_user:
        return True

    # 3. Low-level JWT validation helpers
    if call is validate_jwt:
        return True

    # 4. Security schemes (e.g. JWT_HEADER used in auth/status)
    return bool(isinstance(call, APIKeyHeader))


def has_auth_defence(route: APIRoute) -> bool:
    """Recursively check if a route has any authentication dependency."""

    def check_dependant(dependant: Dependant) -> bool:
        # Check the callable for this specific dependency
        if dependant.call is not None and is_auth_mechanism(dependant.call):
            return True

        # Recursively check sub-dependencies
        # (e.g. a dependency that depends on get_current_user)
        return any(check_dependant(sub_dep) for sub_dep in dependant.dependencies)

    return check_dependant(route.dependant)


def test_all_endpoints_are_classified_correctly() -> None:
    """Safety net: Ensure every endpoint is either explicitly public or defended.

    This test prevents accidental exposure of new endpoints. If you add a new
    endpoint, you must either:

    1. Add an authentication dependency (e.g. RequireSysAdmin), OR
    2. Explicitly add it to PUBLIC_ROUTES if it is intended to be open.
    """
    routes_found: int = 0

    for route in app.routes:
        # Skip static file mounts and other non-API routes
        if not isinstance(route, APIRoute):
            continue

        for method in route.methods:
            routes_found += 1
            route_id: tuple[str, str] = (method, route.path)

            is_defended: bool = has_auth_defence(route)
            is_marked_public: bool = route_id in PUBLIC_ROUTES

            if is_marked_public:
                # If it's public, it shouldn't enforce auth
                pass
            else:
                # If not marked public, it MUST have a defense.
                assert is_defended, (
                    f"Security Alert: Endpoint '{method} {route.path}' is unprotected! "
                    "It is not in PUBLIC_ROUTES and lacks an authentication dependency "
                    "(RequiresPermission, get_current_user, etc)."
                )

    assert routes_found > 0, "No routes found to audit."
