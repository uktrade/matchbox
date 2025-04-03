"""PostgreSQL adapter for Matchbox server."""

from matchbox.server.postgresql._settings import MatchboxPostgresSettings
from matchbox.server.postgresql.adapter import (
    MatchboxPostgres,
)

__all__ = ["MatchboxPostgres", "MatchboxPostgresSettings"]
