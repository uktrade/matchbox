"""Alembic post-generation hook that rewrites a migration file to be schema-agnostic.

Replaces hard-coded schema references with dynamic equivalents driven by a
schema variable, so the same migration works against any target schema at runtime.
"""

from __future__ import annotations

import re
import sys

from matchbox.server.postgresql.db import MBDB


def make_schema_agnostic(path: str) -> None:
    """Rewrite migration in-place, replacing hard-coded schema literals.

    Args:
        path: Path to the Alembic migration file to patch.
    """
    schema_name = MBDB.settings.postgres.db_schema

    with open(path) as fh:
        content = fh.read()

    quote = r"""['"]"""  # either quote style

    content = re.sub(
        rf"schema={quote}{re.escape(schema_name)}{quote}", "schema=schema", content
    )
    content = re.sub(
        rf'{quote}{re.escape(schema_name)}\.([^"\']+){quote}',
        r'f"{schema}.\1"',
        content,
    )

    with open(path, "w") as fh:
        fh.write(content)


if __name__ == "__main__":
    make_schema_agnostic(sys.argv[1])
