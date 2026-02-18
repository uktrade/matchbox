"""Replace resolver strategy/threshold columns with class/settings payloads.

Revision ID: a43de8fd3981
Revises: 5eaaa8446fb8
Create Date: 2026-02-18 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a43de8fd3981"
down_revision: str | None = "5eaaa8446fb8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "resolver_configs",
        "strategy",
        new_column_name="resolver_class",
        schema="mb",
    )
    op.alter_column(
        "resolver_configs",
        "thresholds",
        new_column_name="resolver_settings",
        schema="mb",
    )
    op.alter_column(
        "resolver_configs",
        "resolver_settings",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.TEXT(),
        postgresql_using="resolver_settings::text",
        schema="mb",
    )

    op.execute(
        "UPDATE mb.resolver_configs "
        "SET resolver_class = 'Components' "
        "WHERE resolver_class = 'union'"
    )
    op.execute(
        "UPDATE mb.resolver_configs "
        "SET resolver_settings = "
        "jsonb_build_object('thresholds', resolver_settings::jsonb)::text"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "UPDATE mb.resolver_configs "
        "SET resolver_settings = "
        "coalesce((resolver_settings::jsonb -> 'thresholds')::text, '{}'::jsonb::text)"
    )
    op.alter_column(
        "resolver_configs",
        "resolver_settings",
        existing_type=sa.TEXT(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        postgresql_using="resolver_settings::jsonb",
        schema="mb",
    )
    op.alter_column(
        "resolver_configs",
        "resolver_settings",
        new_column_name="thresholds",
        schema="mb",
    )
    op.alter_column(
        "resolver_configs",
        "resolver_class",
        new_column_name="strategy",
        schema="mb",
    )
    op.execute(
        "UPDATE mb.resolver_configs "
        "SET strategy = 'union' "
        "WHERE strategy = 'Components'"
    )
