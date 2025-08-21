from os import environ
from typing import Callable, Generator
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
from matchbox.client.authorisation import (
    generate_EdDSA_key_pair,
    generate_json_web_token,
)
from matchbox.server.api import app, dependencies
from matchbox.server.base import MatchboxBackends, MatchboxServerSettings
from matchbox.server.uploads import InMemoryUploadTracker


@pytest.fixture(scope="function")
def env_setter() -> Generator[Callable[[str, str], None], None, None]:
    """Set temporary env variable and refresh client settings."""
    original_values = {}

    def setter(var_name: str, var_value: str):
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
def api_client_and_mocks(
    env_setter: Callable[[str, str], None],
) -> Generator[tuple[TestClient, Mock, Mock], None, None]:
    """Return client to testable API and associated mocks."""
    # 1) Prepare keys for authentication
    private_key, public_key = generate_EdDSA_key_pair()
    env_setter("MB__CLIENT__PRIVATE_KEY", private_key.decode())
    token = generate_json_web_token(sub="test.user@email.com")
    auth_headers = {"Authorization": token}

    # 2) Override backend with mock
    # Backend has no functionality and must be adapted for each test
    mock_backend = Mock()
    app.dependency_overrides[dependencies.backend] = lambda: mock_backend

    # 3) Override upload tracker with fully functioning mock
    # Note that we don't need to patch the tracker used by the task, as later
    # we set the API as the task runner, and we assume that in that setting
    # the tracker is passed to the background task via dependency injection
    tracker = InMemoryUploadTracker()
    mock_tracker = Mock()
    mock_tracker.get.side_effect = tracker.get
    mock_tracker.update.side_effect = tracker.update
    mock_tracker.add_model.side_effect = tracker.add_model
    mock_tracker.add_source.side_effect = tracker.add_source
    mock_tracker._tracker = tracker
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
    yield TestClient(app, headers=auth_headers), mock_backend, mock_tracker

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
