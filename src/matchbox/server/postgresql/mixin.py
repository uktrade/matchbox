"""A module for defining mixins for the PostgreSQL backend ORM."""

from typing import TypeVar

from sqlalchemy import func

from matchbox.server.postgresql.db import MBDB

T = TypeVar("T")


class CountMixin:
    """A mixin for counting the number of rows in a table."""

    @classmethod
    def count(cls: type[T]) -> int:
        """Counts the number of rows in the table."""
        with MBDB.get_session() as session:
            return session.query(func.count()).select_from(cls).scalar()
