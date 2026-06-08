"""Renamed primary keys to just keys.

Revision ID: e4122bdf9b0d
Revises: 83b134a86713
Create Date: 2025-05-13 13:10:12.346864

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e4122bdf9b0d"
down_revision: str | None = "83b134a86713"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Rename column in pk_space table
    op.alter_column(
        "pk_space",
        "next_cluster_source_pk_id",
        new_column_name="next_cluster_keys_id",
        schema=schema,
    )

    # Rename cluster_source_pks table to cluster_keys
    op.rename_table("cluster_source_pks", "cluster_keys", schema=schema)

    # Rename columns in cluster_keys table
    op.alter_column("cluster_keys", "pk_id", new_column_name="key_id", schema=schema)
    op.alter_column("cluster_keys", "source_pk", new_column_name="key", schema=schema)

    # Rename constraint in cluster_keys table
    op.drop_constraint("unique_pk_source", "cluster_keys", schema=schema)
    op.create_unique_constraint(
        "unique_keys_source",
        "cluster_keys",
        ["key_id", "source_config_id"],
        schema=schema,
    )

    # Rename indexes (they need to be dropped and recreated with new names)
    op.drop_index(
        "ix_cluster_source_pks_cluster_id", table_name="cluster_keys", schema=schema
    )
    op.drop_index(
        "ix_cluster_source_pks_source_pk", table_name="cluster_keys", schema=schema
    )

    op.create_index(
        "ix_cluster_keys_cluster_id",
        "cluster_keys",
        ["cluster_id"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        "ix_cluster_keys_keys",
        "cluster_keys",
        ["key"],
        unique=False,
        schema=schema,
    )

    # Rename column in source_configs table
    op.alter_column("source_configs", "db_pk", new_column_name="key_field", schema=schema)


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Rename column in source_configs table back
    op.alter_column("source_configs", "key_field", new_column_name="db_pk", schema=schema)

    # Rename indexes back
    op.drop_index("ix_cluster_keys_cluster_id", table_name="cluster_keys", schema=schema)
    op.drop_index("ix_cluster_keys_keys", table_name="cluster_keys", schema=schema)

    op.create_index(
        "ix_cluster_source_pks_cluster_id",
        "cluster_keys",
        ["cluster_id"],
        unique=False,
        schema=schema,
    )
    op.create_index(
        "ix_cluster_source_pks_source_pk",
        "cluster_keys",
        ["key"],  # Note: column is still named 'key' at this point
        unique=False,
        schema=schema,
    )

    # Rename constraint back
    op.drop_constraint("unique_keys_source", "cluster_keys", schema=schema)
    op.create_unique_constraint(
        "unique_pk_source", "cluster_keys", ["key_id", "source_config_id"], schema=schema
    )

    # Rename columns back in cluster_keys table
    op.alter_column("cluster_keys", "key", new_column_name="source_pk", schema=schema)
    op.alter_column("cluster_keys", "key_id", new_column_name="pk_id", schema=schema)

    # Rename cluster_keys table back to cluster_source_pks
    op.rename_table("cluster_keys", "cluster_source_pks", schema=schema)

    # Rename column in pk_space table back to next_cluster_source_pk_id
    op.alter_column(
        "pk_space",
        "next_cluster_keys_id",
        new_column_name="next_cluster_source_pk_id",
        schema=schema,
    )
