"""Move ORM to root-leaf contains structure.

Note that this migration is destructive and will clear the data subgraph.

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
    """Upgrade schema - DESTRUCTIVE: clears the data subgraph."""
    # Clear all data first since this is a destructive migration
    op.execute("TRUNCATE TABLE mb.probabilities CASCADE")
    op.execute("TRUNCATE TABLE mb.cluster_keys CASCADE")
    op.execute("TRUNCATE TABLE mb.contains CASCADE")
    op.execute("TRUNCATE TABLE mb.clusters CASCADE")

    # Update contains table structure (parent/child -> root/leaf)
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
        "contains_root_fkey",
        "contains",
        "clusters",
        ["root"],
        ["cluster_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "contains_leaf_fkey",
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

    # Update probabilities table column names
    # (resolution -> resolution_id, cluster -> cluster_id)
    # First drop constraints and indexes
    op.drop_constraint(
        "probabilities_pkey", "probabilities", schema="mb", type_="primary"
    )
    op.drop_constraint(
        "probabilities_resolution_fkey",
        "probabilities",
        schema="mb",
        type_="foreignkey",
    )
    op.drop_constraint(
        "probabilities_cluster_fkey", "probabilities", schema="mb", type_="foreignkey"
    )
    op.drop_index(
        "ix_probabilities_resolution", table_name="probabilities", schema="mb"
    )

    # Rename columns
    op.alter_column(
        "probabilities", "resolution", new_column_name="resolution_id", schema="mb"
    )
    op.alter_column(
        "probabilities", "cluster", new_column_name="cluster_id", schema="mb"
    )

    # Recreate constraints and indexes with new column names
    op.create_primary_key(
        "probabilities_pkey",
        "probabilities",
        ["resolution_id", "cluster_id"],
        schema="mb",
    )
    op.create_foreign_key(
        "probabilities_resolution_id_fkey",
        "probabilities",
        "resolutions",
        ["resolution_id"],
        ["resolution_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "probabilities_cluster_id_fkey",
        "probabilities",
        "clusters",
        ["cluster_id"],
        ["cluster_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_probabilities_resolution",
        "probabilities",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )

    # Create results table with surrogate primary key and unique constraint
    op.create_table(
        "results",
        sa.Column("result_id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("resolution_id", sa.BIGINT(), nullable=False),
        sa.Column("left_id", sa.BIGINT(), nullable=False),
        sa.Column("right_id", sa.BIGINT(), nullable=False),
        sa.Column("probability", sa.SMALLINT(), nullable=False),
        sa.CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
        sa.ForeignKeyConstraint(
            ["resolution_id"], ["mb.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["left_id"], ["mb.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["right_id"], ["mb.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("result_id"),
        sa.UniqueConstraint("resolution_id", "left_id", "right_id"),
        schema="mb",
    )
    op.create_index(
        "ix_results_resolution",
        "results",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema - DESTRUCTIVE: clears the data subgraph."""
    # Clear all data first since this is a destructive migration
    op.execute("TRUNCATE TABLE mb.probabilities CASCADE")
    op.execute("TRUNCATE TABLE mb.cluster_keys CASCADE")
    op.execute("TRUNCATE TABLE mb.contains CASCADE")
    op.execute("TRUNCATE TABLE mb.clusters CASCADE")
    op.execute("TRUNCATE TABLE mb.results CASCADE")

    # Drop results table
    op.drop_table("results", schema="mb")

    # Revert probabilities table column names
    # (resolution_id -> resolution, cluster_id -> cluster)
    # First drop constraints and indexes
    op.drop_constraint(
        "probabilities_pkey", "probabilities", schema="mb", type_="primary"
    )
    op.drop_constraint(
        "probabilities_resolution_id_fkey",
        "probabilities",
        schema="mb",
        type_="foreignkey",
    )
    op.drop_constraint(
        "probabilities_cluster_id_fkey",
        "probabilities",
        schema="mb",
        type_="foreignkey",
    )
    op.drop_index(
        "ix_probabilities_resolution", table_name="probabilities", schema="mb"
    )

    # Rename columns back
    op.alter_column(
        "probabilities", "resolution_id", new_column_name="resolution", schema="mb"
    )
    op.alter_column(
        "probabilities", "cluster_id", new_column_name="cluster", schema="mb"
    )

    # Recreate constraints and indexes with old column names
    op.create_primary_key(
        "probabilities_pkey", "probabilities", ["resolution", "cluster"], schema="mb"
    )
    op.create_foreign_key(
        "probabilities_resolution_fkey",
        "probabilities",
        "resolutions",
        ["resolution"],
        ["resolution_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "probabilities_cluster_fkey",
        "probabilities",
        "clusters",
        ["cluster"],
        ["cluster_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_probabilities_resolution",
        "probabilities",
        ["resolution"],
        unique=False,
        schema="mb",
    )

    # Revert contains table structure (root/leaf -> parent/child)
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
    op.drop_constraint(
        "contains_root_fkey", "contains", schema="mb", type_="foreignkey"
    )
    op.drop_constraint(
        "contains_leaf_fkey", "contains", schema="mb", type_="foreignkey"
    )
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
