from importlib.metadata import version

import httpx
from httpx import Response
from respx import MockRouter

from matchbox.client._handler import create_client, login
from matchbox.client._settings import ClientSettings


def test_create_client():
    mock_settings = ClientSettings(api_root="http://example.com", timeout=20)
    client = create_client(mock_settings)

    assert client.headers.get("X-Matchbox-Client-Version") == version("matchbox_db")
    assert client.base_url == mock_settings.api_root
    assert client.timeout.connect == mock_settings.timeout
    assert client.timeout.pool == mock_settings.timeout
    assert client.timeout.read == 60 * 30
    assert client.timeout.write == 60 * 30


def test_retry_decorator_applied(matchbox_api: MockRouter):
    """Test that retry decorator works by mocking API errors."""

    # Check that the function has retry attributes (indicating decorator was applied)
    assert hasattr(login, "retry")
    assert hasattr(login, "retry_with")

    # Verify retry configuration
    retry_state = login.retry
    assert retry_state.stop.max_attempt_number == 5

    # Mock the API to fail twice with network errors, then succeed
    matchbox_api.post("/login").mock(
        side_effect=[
            httpx.ConnectError("Connection failed"),  # First call fails
            httpx.ConnectError("Connection failed"),  # Second call fails
            Response(200, json={"user_id": 123}),  # Third call succeeds
        ]
    )

    # Call the function - it should retry and eventually succeed
    result = login("test_user")

    # Verify it succeeded after retries
    assert result == 123

    # Verify the API was called 3 times (2 failures + 1 success)
    assert len(matchbox_api.calls) == 3
