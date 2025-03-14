"""Matchbox server.

Includes the API, and database adapters for various backends.
"""

from matchbox.server.api.routes import app
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxSettings,
    initialise_matchbox,
)

__all__ = ["app", "MatchboxDBAdapter", "MatchboxSettings"]

initialise_matchbox()
