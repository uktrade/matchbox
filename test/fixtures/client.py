from os import environ
from typing import Callable, Generator
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from matchbox.client._settings import settings as client_settings
from matchbox.server import app


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
@patch("matchbox.server.api.routes.backend")
def test_client(
    _: Mock, env_setter: Callable[[str, str], None]
) -> Generator[TestClient, None, None]:
    env_setter("MB__SERVER__API_KEY", "test-api-key")
    return TestClient(app, headers={"X-API-Key": "test-api-key"})
