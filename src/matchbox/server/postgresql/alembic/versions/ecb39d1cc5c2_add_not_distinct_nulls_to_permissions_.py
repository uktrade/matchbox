"""Add not distinct nulls to Permissions to trigger unique constraints.

Revision ID: ecb39d1cc5c2
Revises: 3dfe29cecf2f
Create Date: 2026-01-19 12:41:36.447563

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ecb39d1cc5c2"
down_revision: str | None = "3dfe29cecf2f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_constraint(
        op.f("unique_permission_grant"), "permissions", schema=schema, type_="unique"
    )
    op.create_unique_constraint(
        "unique_permission_grant",
        "permissions",
        ["permission", "group_id", "collection_id", "is_system"],
        schema=schema,
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_constraint(
        "unique_permission_grant", "permissions", schema=schema, type_="unique"
    )
    op.create_unique_constraint(
        op.f("unique_permission_grant"),
        "permissions",
        ["permission", "group_id", "collection_id", "is_system"],
        schema=schema,
        postgresql_nulls_not_distinct=False,
    )
