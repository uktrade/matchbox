"""Removed source key reference and added it to the source field table.

Revision ID: 05cc4181a0ad
Revises: ae63f79f6b39
Create Date: 2025-05-14 16:33:08.656616

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision: str = "05cc4181a0ad"
down_revision: str | None = "ae63f79f6b39"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema: remove key_field column and add is_key flag to source_fields."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Shift all existing source_fields indexes up by 1 to make room for key field
    # at index 0. This is done in two steps to avoid index collisions.

    # Step 1: Move all indexes to negative values
    op.execute(
        text("""
        UPDATE {schema}.source_fields 
        SET index = -(index + 1)
    """)
    )

    # Step 2: Convert back to positive values (now shifted up by 1)
    op.execute(
        text("""
        UPDATE {schema}.source_fields 
        SET index = -index
    """)
    )

    # Add the is_key column as nullable first
    op.add_column(
        "source_fields", sa.Column("is_key", sa.BOOLEAN(), nullable=True), schema=schema
    )

    # Create new source_fields entries for each key_field and mark them as key fields
    connection = op.get_bind()
    source_configs = connection.execute(
        text(f"SELECT source_config_id, key_field FROM {schema}.source_configs")
    ).fetchall()

    for source_config_id, key_field_name in source_configs:
        # Insert new source_field for the key field at index 0 with is_key=True
        connection.execute(
            text("""
                INSERT INTO {schema}.source_fields
                    (source_config_id, index, name, type, is_key)
                VALUES
                    (:source_config_id, 0, :key_field_name, 'String', true)
            """),
            {"source_config_id": source_config_id, "key_field_name": key_field_name},
        )

    # Set all existing fields to is_key=False since they're not key fields
    op.execute(
        text("""
        UPDATE {schema}.source_fields 
        SET is_key = false 
        WHERE is_key IS NULL
    """)
    )

    # Make is_key non-nullable now that all rows have values
    op.alter_column("source_fields", "is_key", nullable=False, schema=schema)

    # Add partial unique index to ensure only one key field per source_config
    op.create_index(
        "ix_unique_key_field",
        "source_fields",
        ["source_config_id"],
        unique=True,
        postgresql_where=text("is_key = true"),
        schema=schema,
    )

    # Drop the old key_field column
    op.drop_column("source_configs", "key_field", schema=schema)


def downgrade() -> None:
    """Downgrade schema: restore key_field column and remove is_key flag."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Add the key_field column back as nullable first
    op.add_column(
        "source_configs",
        sa.Column("key_field", sa.TEXT(), nullable=True),
        schema=schema,
    )

    # Populate key_field with names from key source_fields (is_key=True)
    op.execute(
        text("""
        UPDATE {schema}.source_configs sc
        SET key_field = sf.name
        FROM {schema}.source_fields sf
        WHERE sc.source_config_id = sf.source_config_id
        AND sf.is_key = true
    """)
    )

    # Make key_field non-nullable
    op.alter_column("source_configs", "key_field", nullable=False, schema=schema)

    # Drop the partial unique index
    op.drop_index("ix_unique_key_field", "source_fields", schema=schema)

    # Remove the key field entries from source_fields (is_key=True)
    op.execute(
        text("""
        DELETE FROM {schema}.source_fields 
        WHERE is_key = true
    """)
    )

    # Shift remaining source_fields indexes back down by 1
    op.execute(
        text("""
        UPDATE {schema}.source_fields 
        SET index = index - 1
    """)
    )

    # Drop the is_key column
    op.drop_column("source_fields", "is_key", schema=schema)