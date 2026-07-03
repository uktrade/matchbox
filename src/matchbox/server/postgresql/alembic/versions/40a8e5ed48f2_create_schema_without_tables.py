"""Create schema without tables.

Revision ID: 40a8e5ed48f2
Revises: This is the first migration
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "40a8e5ed48f2"
# down_revision=None indicates that this is the first migration file
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

REQUIRED_EXTENSIONS = ("uuid-ossp", "pgcrypto")


def upgrade() -> None:
    """Verify the operator has provisioned the schema and required extensions.

    See see docs/server/install.md for more information.
    """
    schema = op.get_context().config.get_main_option("db_schema")
    connection = op.get_bind()

    schema_exists = connection.execute(
        sa.text(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = :schema"
        ),
        {"schema": schema},
    ).scalar()
    if not schema_exists:
        raise RuntimeError(
            f'Schema "{schema}" does not exist. It must be created by the '
            "operator before running migrations (see docs/server/install.md)."
        )

    for extension in REQUIRED_EXTENSIONS:
        extension_installed = connection.execute(
            sa.text("SELECT 1 FROM pg_extension WHERE extname = :extension"),
            {"extension": extension},
        ).scalar()
        if not extension_installed:
            raise RuntimeError(
                f'Extension "{extension}" is not installed. It must be created '
                "by the operator before running migrations "
                "(see docs/server/install.md)."
            )


def downgrade() -> None:
    """Downgrade schema."""
    # No-op: schema and extension lifecycle is managed by the operator, not migrations.
