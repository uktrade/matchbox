"""Matchbox server.

Includes the API, and database adapters for various backends.
"""

import os

from matchbox.server.api.routes import app
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxSettings,
    initialise_matchbox,
)

__all__ = ["app", "MatchboxDBAdapter", "MatchboxSettings"]

if "PYTEST_CURRENT_TEST" not in os.environ:
    initialise_matchbox()
