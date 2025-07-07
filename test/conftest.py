import logging
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.progress import Progress

pytest_plugins = [
    "test.fixtures.db",
    "test.fixtures.graph",
    "test.fixtures.client",
]

TEST_ROOT = Path(__file__).resolve().parent


def pytest_configure():
    """Configure pytest settings."""

    # Quieten down the logging for specific loggers
    logging.getLogger("faker").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_root_dir() -> Path:
    return TEST_ROOT


@pytest.fixture(scope="session", autouse=True)
def patch_rich_console() -> Iterator[None]:
    """Patch Rich console for quiet output in tests."""
    quiet_console = Console(quiet=True)

    console_patch = patch("matchbox.common.logging.console", new=quiet_console)
    progress_patch = patch(
        "matchbox.common.logging.build_progress_bar",
        return_value=Progress(console=quiet_console),
    )

    with console_patch, progress_patch:
        yield
