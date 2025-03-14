"""PostgreSQL adapter for Matchbox server."""

from matchbox.server.postgresql.adapter import (
    MatchboxPostgres,
    MatchboxPostgresSettings,
)

__all__ = ["MatchboxPostgres", "MatchboxPostgresSettings"]
