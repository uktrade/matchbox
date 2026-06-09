"""Create tables given schema.

Revision ID: 1907c34cfa1f
Revises: 40a8e5ed48f2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "1907c34cfa1f"
down_revision: str | None = "40a8e5ed48f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.create_table(
        "clusters",
        sa.Column("cluster_id", sa.BIGINT(), nullable=False),
        sa.Column("cluster_hash", postgresql.BYTEA(), nullable=False),
        sa.PrimaryKeyConstraint("cluster_id"),
        sa.UniqueConstraint("cluster_hash", name="clusters_hash_key"),
        schema=schema,
    )
    op.create_table(
        "resolutions",
        sa.Column("resolution_id", sa.BIGINT(), nullable=False),
        sa.Column("resolution_hash", postgresql.BYTEA(), nullable=False),
        sa.Column("type", sa.TEXT(), nullable=False),
        sa.Column("name", sa.TEXT(), nullable=False),
        sa.Column("description", sa.TEXT(), nullable=True),
        sa.Column("truth", sa.SMALLINT(), nullable=True),
        sa.CheckConstraint(
            "type IN ('model', 'dataset', 'human')", name="resolution_type_constraints"
        ),
        sa.PrimaryKeyConstraint("resolution_id"),
        sa.UniqueConstraint("name", name="resolutions_name_key"),
        sa.UniqueConstraint("resolution_hash", name="resolutions_hash_key"),
        schema=schema,
    )
    op.create_table(
        "contains",
        sa.Column("parent", sa.BIGINT(), nullable=False),
        sa.Column("child", sa.BIGINT(), nullable=False),
        sa.CheckConstraint("parent != child", name="no_self_containment"),
        sa.ForeignKeyConstraint(
            ["child"], [f"{schema}.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["parent"], [f"{schema}.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("parent", "child"),
        schema=schema,
    )
    op.create_index(
        "ix_contains_child_parent",
        "contains",
        ["child", "parent"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        "ix_contains_parent_child",
        "contains",
        ["parent", "child"],
        unique=False,
        schema=schema,
    )
    op.create_table(
        "probabilities",
        sa.Column("resolution", sa.BIGINT(), nullable=False),
        sa.Column("cluster", sa.BIGINT(), nullable=False),
        sa.Column("probability", sa.SMALLINT(), nullable=False),
        sa.CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
        sa.ForeignKeyConstraint(
            ["cluster"], [f"{schema}.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["resolution"], [f"{schema}.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("resolution", "cluster"),
        schema=schema,
    )
    op.create_table(
        "resolution_from",
        sa.Column("parent", sa.BIGINT(), nullable=False),
        sa.Column("child", sa.BIGINT(), nullable=False),
        sa.Column("level", sa.INTEGER(), nullable=False),
        sa.Column("truth_cache", sa.SMALLINT(), nullable=True),
        sa.CheckConstraint("level > 0", name="positive_level"),
        sa.CheckConstraint("parent != child", name="no_self_reference"),
        sa.ForeignKeyConstraint(
            ["child"], [f"{schema}.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["parent"], [f"{schema}.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("parent", "child"),
        schema=schema,
    )
    op.create_table(
        "sources",
        sa.Column(
            "source_id", sa.BIGINT(), sa.Identity(always=False, start=1), nullable=False
        ),
        sa.Column("resolution_id", sa.BIGINT(), nullable=True),
        sa.Column("resolution_name", sa.TEXT(), nullable=False),
        sa.Column("full_name", sa.TEXT(), nullable=False),
        sa.Column("warehouse_hash", postgresql.BYTEA(), nullable=False),
        sa.Column("db_pk", sa.TEXT(), nullable=False),
        sa.ForeignKeyConstraint(
            ["resolution_id"],
            [f"{schema}.resolutions.resolution_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("source_id"),
        sa.UniqueConstraint(
            "full_name", "warehouse_hash", name="unique_source_address"
        ),
        schema=schema,
    )
    op.create_table(
        "cluster_source_pks",
        sa.Column("pk_id", sa.BIGINT(), nullable=False),
        sa.Column("cluster_id", sa.BIGINT(), nullable=False),
        sa.Column("source_id", sa.BIGINT(), nullable=False),
        sa.Column("source_pk", sa.TEXT(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cluster_id"], [f"{schema}.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_id"], [f"{schema}.sources.source_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("pk_id"),
        sa.UniqueConstraint("pk_id", "source_id", name="unique_pk_source"),
        schema=schema,
    )
    op.create_index(
        "ix_cluster_source_pks_cluster_id",
        "cluster_source_pks",
        ["cluster_id"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        "ix_cluster_source_pks_source_pk",
        "cluster_source_pks",
        ["source_pk"],
        unique=False,
        schema=schema,
    )
    op.create_table(
        "source_columns",
        sa.Column("column_id", sa.BIGINT(), nullable=False),
        sa.Column("source_id", sa.BIGINT(), nullable=False),
        sa.Column("column_index", sa.INTEGER(), nullable=False),
        sa.Column("column_name", sa.TEXT(), nullable=False),
        sa.Column("column_type", sa.TEXT(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"], [f"{schema}.sources.source_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("column_id"),
        sa.UniqueConstraint("source_id", "column_index", name="unique_column_index"),
        schema=schema,
    )
    op.create_index(
        "ix_source_columns_source_id",
        "source_columns",
        ["source_id"],
        unique=False,
        schema=schema,
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_index(
        "ix_source_columns_source_id", table_name="source_columns", schema=schema
    )
    op.drop_table("source_columns", schema=schema)
    op.drop_index(
        "ix_cluster_source_pks_source_pk",
        table_name="cluster_source_pks",
        schema=schema,
    )
    op.drop_index(
        "ix_cluster_source_pks_cluster_id",
        table_name="cluster_source_pks",
        schema=schema,
    )
    op.drop_table("cluster_source_pks", schema=schema)
    op.drop_table("sources", schema=schema)
    op.drop_table("resolution_from", schema=schema)
    op.drop_table("probabilities", schema=schema)
    op.drop_index("ix_contains_parent_child", table_name="contains", schema=schema)
    op.drop_index("ix_contains_child_parent", table_name="contains", schema=schema)
    op.drop_table("contains", schema=schema)
    op.drop_table("resolutions", schema=schema)
    op.drop_table("clusters", schema=schema)
