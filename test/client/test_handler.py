from importlib.metadata import version

import httpx
from httpx import Response
from respx import MockRouter

from matchbox.client import _handler
from matchbox.client._handler.main import create_client
from matchbox.client._settings import ClientSettings
from matchbox.common.dtos import AuthStatusResponse, LoginResponse


def test_create_client() -> None:
    mock_settings = ClientSettings(api_root="http://example.com", timeout=20)
    client = create_client(mock_settings)

    assert client.headers.get("X-Matchbox-Client-Version") == version("matchbox_db")
    assert client.base_url == mock_settings.api_root
    assert client.timeout.connect == mock_settings.timeout
    assert client.timeout.pool == mock_settings.timeout
    assert client.timeout.read == 60 * 30
    assert client.timeout.write == 60 * 30


def test_retry_decorator_applied(matchbox_api: MockRouter) -> None:
    """Test that retry decorator works by mocking API errors."""

    # Check that the function has retry attributes (indicating decorator was applied)
    assert hasattr(_handler.login, "retry")
    assert hasattr(_handler.login, "retry_with")

    # Verify retry configuration
    retry_state = _handler.login.retry
    assert retry_state.stop.max_attempt_number == 5

    # Mock the API to fail twice with network errors, then succeed
    matchbox_api.post("/auth/login").mock(
        side_effect=[
            httpx.ConnectError("Connection failed"),  # First call fails
            httpx.ConnectError("Connection failed"),  # Second call fails
            Response(
                200, json={"user": {"user_id": 123, "user_name": "test_user"}}
            ),  # Third call succeeds
        ]
    )

    # Call the function - it should retry and eventually succeed
    result: LoginResponse = _handler.login()

    # Verify it succeeded after retries
    assert result.user.user_name == "test_user"

    # Verify the API was called 3 times (2 failures + 1 success)
    assert len(matchbox_api.calls) == 3


def test_healthcheck(matchbox_api: MockRouter) -> None:
    """Test the healthcheck endpoint works."""
    matchbox_api.get("/health").mock(
        side_effect=[
            Response(200, json={"status": "OK", "version": "0.0.0.dev0"}),
        ]
    )
    result = _handler.healthcheck()

    assert result.status == "OK"
    assert result.version == "0.0.0.dev0"


def test_auth_status(matchbox_api: MockRouter) -> None:
    """Test the auth status endpoint works."""
    matchbox_api.get("/auth/status").mock(
        side_effect=[
            Response(
                200,
                json={
                    "authenticated": True,
                    "user": {
                        "user_name": "test_user",
                        "email": "test@example.com",
                    },
                },
            ),
        ]
    )
    result: AuthStatusResponse = _handler.auth_status()

    assert result.authenticated is True
    assert result.user is not None
    assert result.user.user_name == "test_user"
    assert result.user.email == "test@example.com"
