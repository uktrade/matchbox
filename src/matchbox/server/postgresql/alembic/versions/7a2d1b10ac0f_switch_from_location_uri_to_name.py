"""Switch from location URI to name.

Revision ID: 7a2d1b10ac0f
Revises: dd0c3a9ecdf9
Create Date: 2025-07-29 07:52:36.312694

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7a2d1b10ac0f"
down_revision: str | None = "dd0c3a9ecdf9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Add default value for rows without location names
    op.add_column(
        "source_configs",
        sa.Column(
            "location_name",
            sa.TEXT(),
            nullable=False,
            server_default="legacy",
        ),
        schema=schema,
    )
    # Remove default value for location names
    op.alter_column(
        "source_configs", "location_name", server_default=None, schema=schema
    )
    op.drop_column("source_configs", "location_uri", schema=schema)


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Add default value for rows without location URIs
    op.add_column(
        "source_configs",
        sa.Column(
            "location_uri",
            sa.TEXT(),
            autoincrement=False,
            nullable=False,
            server_default="sqlite:///:memory:",
        ),
        schema=schema,
    )
    # Remove default value for location URIs
    op.alter_column(
        "source_configs", "location_uri", server_default=None, schema=schema
    )
    op.drop_column("source_configs", "location_name", schema=schema)
