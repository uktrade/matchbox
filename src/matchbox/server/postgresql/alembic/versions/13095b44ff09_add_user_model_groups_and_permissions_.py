"""Add user model, groups and permissions to ORM.

Revision ID: 13095b44ff09
Revises: c774fd4b69f8
Create Date: 2025-12-02 12:48:16.444413

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "13095b44ff09"
down_revision: str | None = "c774fd4b69f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "groups",
        sa.Column("group_id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("name", sa.TEXT(), nullable=False),
        sa.Column("description", sa.TEXT(), nullable=True),
        sa.Column("is_system", sa.BOOLEAN(), nullable=False),
        sa.PrimaryKeyConstraint("group_id"),
        sa.UniqueConstraint("name", name="groups_name_key"),
        schema="mb",
    )
    op.create_table(
        "permissions",
        sa.Column("permission_id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("permission", sa.TEXT(), nullable=False),
        sa.Column("group_id", sa.BIGINT(), nullable=False),
        sa.Column("collection_id", sa.BIGINT(), nullable=True),
        sa.Column("is_system", sa.BOOLEAN(), nullable=True),
        sa.CheckConstraint(
            "permission IN ('read', 'write', 'admin')", name="valid_permission"
        ),
        sa.CheckConstraint(
            "(collection_id IS NOT NULL AND is_system IS NULL) "
            "OR (collection_id IS NULL AND is_system = true)",
            name="exactly_one_resource",
        ),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["mb.collections.collection_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["group_id"], ["mb.groups.group_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("permission_id"),
        sa.UniqueConstraint(
            "permission",
            "group_id",
            "collection_id",
            "is_system",
            name="unique_permission_grant",
        ),
        schema="mb",
    )
    op.create_table(
        "user_groups",
        sa.Column("user_id", sa.BIGINT(), nullable=False),
        sa.Column("group_id", sa.BIGINT(), nullable=False),
        sa.ForeignKeyConstraint(
            ["group_id"], ["mb.groups.group_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["mb.users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "group_id"),
        schema="mb",
    )
    op.add_column("users", sa.Column("email", sa.TEXT(), nullable=True), schema="mb")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "email", schema="mb")
    op.drop_table("user_groups", schema="mb")
    op.drop_table("permissions", schema="mb")
    op.drop_table("groups", schema="mb")
