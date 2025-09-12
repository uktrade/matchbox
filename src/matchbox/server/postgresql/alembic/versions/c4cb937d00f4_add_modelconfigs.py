"""Add ModelConfigs.

Revision ID: c4cb937d00f4
Revises: b38d61ab11cc
Create Date: 2025-09-11 12:30:44.246293

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c4cb937d00f4"
down_revision: str | None = "b38d61ab11cc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "model_configs",
        sa.Column(
            "model_config_id",
            sa.BIGINT(),
            sa.Identity(always=False, start=1),
            nullable=False,
        ),
        sa.Column("resolution_id", sa.BIGINT(), nullable=False),
        sa.Column("model_class", sa.TEXT(), nullable=False),
        sa.Column(
            "model_settings", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "left_query", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "right_query", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["resolution_id"], ["mb.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("model_config_id"),
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("model_configs", schema="mb")
