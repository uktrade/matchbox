"""Move ORM to root-leaf contains structure.

Revision ID: 3754ae042254
Revises: 4a7c35f86405
Create Date: 2025-05-22 05:48:36.049641

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3754ae042254"
down_revision: str | None = "4a7c35f86405"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "contains", sa.Column("root", sa.BIGINT(), nullable=False), schema="mb"
    )
    op.add_column(
        "contains", sa.Column("leaf", sa.BIGINT(), nullable=False), schema="mb"
    )
    op.drop_index("ix_contains_child_parent", table_name="contains", schema="mb")
    op.drop_index("ix_contains_parent_child", table_name="contains", schema="mb")
    op.create_index(
        "ix_contains_leaf_root", "contains", ["leaf", "root"], unique=False, schema="mb"
    )
    op.create_index(
        "ix_contains_root_leaf", "contains", ["root", "leaf"], unique=False, schema="mb"
    )
    op.drop_constraint(
        "contains_child_fkey", "contains", schema="mb", type_="foreignkey"
    )
    op.drop_constraint(
        "contains_parent_fkey", "contains", schema="mb", type_="foreignkey"
    )
    op.create_foreign_key(
        None,
        "contains",
        "clusters",
        ["root"],
        ["cluster_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        None,
        "contains",
        "clusters",
        ["leaf"],
        ["cluster_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.drop_column("contains", "parent", schema="mb")
    op.drop_column("contains", "child", schema="mb")
    op.add_column(
        "probabilities",
        sa.Column("role_flag", sa.SMALLINT(), nullable=False),
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("probabilities", "role_flag", schema="mb")
    op.add_column(
        "contains",
        sa.Column("child", sa.BIGINT(), autoincrement=False, nullable=False),
        schema="mb",
    )
    op.add_column(
        "contains",
        sa.Column("parent", sa.BIGINT(), autoincrement=False, nullable=False),
        schema="mb",
    )
    op.drop_constraint(None, "contains", schema="mb", type_="foreignkey")
    op.drop_constraint(None, "contains", schema="mb", type_="foreignkey")
    op.create_foreign_key(
        "contains_parent_fkey",
        "contains",
        "clusters",
        ["parent"],
        ["cluster_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "contains_child_fkey",
        "contains",
        "clusters",
        ["child"],
        ["cluster_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.drop_index("ix_contains_root_leaf", table_name="contains", schema="mb")
    op.drop_index("ix_contains_leaf_root", table_name="contains", schema="mb")
    op.create_index(
        "ix_contains_parent_child",
        "contains",
        ["parent", "child"],
        unique=False,
        schema="mb",
    )
    op.create_index(
        "ix_contains_child_parent",
        "contains",
        ["child", "parent"],
        unique=False,
        schema="mb",
    )
    op.drop_column("contains", "leaf", schema="mb")
    op.drop_column("contains", "root", schema="mb")
