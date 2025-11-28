"""Composed PostgreSQL adapter for Matchbox server."""

from matchbox.server.postgresql.adapter.main import (
    MatchboxPostgres,
    MatchboxPostgresSettings,
)

__all__ = ("MatchboxPostgres", "MatchboxPostgresSettings")
