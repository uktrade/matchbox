"""Add level to StepFrom primary key.

Revision ID: 65dd8d1fe119
Revises: e06333c2daaf
Create Date: 2026-03-25 12:32:45.780827

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "65dd8d1fe119"
down_revision: str | None = "e06333c2daaf"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_constraint("step_from_pkey", "step_from", schema=schema, type_="primary")
    op.create_primary_key(
        "step_from_pkey", "step_from", ["parent", "child", "level"], schema=schema
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_constraint("step_from_pkey", "step_from", schema=schema, type_="primary")

    # Remove duplicate (parent, child) pairs if any exist
    op.execute("""
        DELETE FROM {schema}.step_from sf1
        WHERE EXISTS (
            SELECT 1 FROM {schema}.step_from sf2
            WHERE sf1.parent = sf2.parent
            AND sf1.child = sf2.child
            AND sf1.level > sf2.level
        )
    """)

    op.create_primary_key(
        "step_from_pkey", "step_from", ["parent", "child"], schema=schema
    )