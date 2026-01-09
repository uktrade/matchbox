from collections.abc import Callable, Generator
from os import environ
from unittest.mock import Mock

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Client
from pydantic import SecretStr
from respx import MockRouter

from matchbox.client._handler import create_client
from matchbox.client._settings import ClientSettings
from matchbox.client._settings import settings as client_settings
from matchbox.server.api import app, dependencies
from matchbox.server.base import MatchboxBackends, MatchboxServerSettings
from matchbox.server.uploads import InMemoryUploadTracker
from test.scripts.authorisation import (
    generate_EdDSA_key_pair,
    generate_json_web_token,
)


@pytest.fixture(scope="function")
def env_setter() -> Generator[Callable[[str, str], None], None, None]:
    """Set temporary env variable and refresh client settings."""
    original_values = {}

    def setter(var_name: str, var_value: str) -> None:
        original_values[var_name] = environ.get(var_name)

        environ[var_name] = var_value
        client_settings.__init__()

    yield setter

    for var_name, original_value in original_values.items():
        if original_value is None:
            del environ[var_name]
        else:
            environ[var_name] = original_value
    client_settings.__init__()


@pytest.fixture(scope="function")
def api_EdDSA_key_pair() -> tuple[bytes, bytes]:
    """Return EdDSA key pair for testing."""
    return generate_EdDSA_key_pair()


@pytest.fixture(scope="function")
def api_client_and_mocks(
    api_EdDSA_key_pair: tuple[bytes, bytes],
    env_setter: Callable[[str, str], None],
) -> Generator[tuple[TestClient, Mock, Mock], None, None]:
    """Return client to testable API and associated mocks."""
    # 1) Prepare keys for authentication
    private_key, public_key = api_EdDSA_key_pair
    user = "test.user@email.com"
    token = generate_json_web_token(
        private_key_bytes=private_key,
        sub=user,
        api_root=client_settings.api_root,
    )
    env_setter("MB__CLIENT__USER", user)
    env_setter("MB__CLIENT__JWT", token)
    auth_headers = {"Authorization": token}

    # 2) Override backend with mock
    # Backend has no functionality and must be adapted for each test
    mock_backend = Mock()
    app.dependency_overrides[dependencies.backend] = lambda: mock_backend

    # 3) Override upload tracker with fully functioning mock
    # Note that we don't need to patch the tracker used by the delayed task,
    # as later we set the API as the task runner, and we assume that in that setting
    # the tracker is passed to the background task via dependency injection
    tracker = InMemoryUploadTracker()
    mock_tracker = Mock()
    mock_tracker.get.side_effect = tracker.get
    mock_tracker.set.side_effect = tracker.set
    app.dependency_overrides[dependencies.upload_tracker] = lambda: mock_tracker

    # 4) Override server settings used by API
    test_settings = MatchboxServerSettings(
        backend_type=MatchboxBackends.POSTGRES,
        task_runner="api",
        authorisation=True,
        public_key=SecretStr(public_key.decode()),
    )
    app.dependency_overrides[dependencies.settings] = lambda: test_settings

    # 5) Yield authenticated test client and the server-side mocks
    # raise_server_exceptions=False ensures exception handlers return responses instead
    # of raising the exception to the test
    yield (
        TestClient(app, headers=auth_headers, raise_server_exceptions=False),
        mock_backend,
        mock_tracker,
    )

    # 6) Restore API's dependencies
    # The client settings should be reset by env_setter once we exit this context
    app.dependency_overrides = {}


@pytest.fixture(scope="function")
def matchbox_api() -> Generator[MockRouter, None, None]:
    """Client for the mocked Matchbox API."""
    with respx.mock(
        base_url=client_settings.api_root, assert_all_called=True
    ) as respx_mock:
        yield respx_mock


@pytest.fixture(scope="session")
def matchbox_client_settings() -> ClientSettings:
    """Client settings for the Matchbox API running in Docker."""
    return client_settings


@pytest.fixture(scope="session")
def matchbox_client(matchbox_client_settings: ClientSettings) -> Client:
    """Client for the Matchbox API running in Docker."""
    return create_client(settings=matchbox_client_settings)
