"""Move SourceConfigs from SourceAddress to more generalised Location system.

Revision ID: 4a7c35f86405
Revises: 05cc4181a0ad
Create Date: 2025-05-16 11:46:57.932236

"""

import hashlib
import re
from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "4a7c35f86405"
down_revision: str | None = "05cc4181a0ad"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new columns
    op.add_column(
        "source_configs",
        sa.Column("location_type", sa.TEXT(), nullable=True),
        schema="mb",
    )
    op.add_column(
        "source_configs",
        sa.Column("location_uri", sa.TEXT(), nullable=True),
        schema="mb",
    )
    op.add_column(
        "source_configs",
        sa.Column("extract_transform", sa.TEXT(), nullable=True),
        schema="mb",
    )

    # Populate new columns with data from existing columns
    connection = op.get_bind()

    # Get source configs and their source fields
    result = connection.execute(
        sa.text("""
        SELECT 
            sc.source_config_id,
            sc.full_name, 
            array_agg(sf.name ORDER BY sf.index) as field_names
        FROM mb.source_configs sc
        LEFT JOIN mb.source_fields sf ON sc.source_config_id = sf.source_config_id
        GROUP BY sc.source_config_id, sc.full_name
    """)
    )

    for row in result:
        config_id = row.source_config_id
        full_name = row.full_name
        field_names = row.field_names or []

        # Set location_type to 'rdbms'
        location_type = "rdbms"

        # Set location_uri to generic URI as specified
        location_uri = "rdbms://database"

        # Create extract_transform query
        fields_str = ", ".join(field_names) if field_names else "*"
        extract_transform = f"select {fields_str} from {full_name}"

        # Update the record
        connection.execute(
            sa.text("""
            UPDATE mb.source_configs 
            SET location_type = :location_type,
                location_uri = :location_uri,
                extract_transform = :extract_transform
            WHERE source_config_id = :config_id
        """),
            {
                "location_type": location_type,
                "location_uri": location_uri,
                "extract_transform": extract_transform,
                "config_id": config_id,
            },
        )

    # Make new columns non-nullable
    op.alter_column("source_configs", "location_type", nullable=False, schema="mb")
    op.alter_column("source_configs", "location_uri", nullable=False, schema="mb")
    op.alter_column("source_configs", "extract_transform", nullable=False, schema="mb")

    # Drop old constraint and create new one
    op.drop_constraint(
        "unique_source_address", "source_configs", schema="mb", type_="unique"
    )
    op.create_unique_constraint(
        "unique_location_transform",
        "source_configs",
        ["location_type", "location_uri", "extract_transform"],
        schema="mb",
    )

    # Drop old columns
    op.drop_column("source_configs", "warehouse_hash", schema="mb")
    op.drop_column("source_configs", "full_name", schema="mb")


def downgrade() -> None:
    """Downgrade schema."""
    # Add back old columns
    op.add_column(
        "source_configs",
        sa.Column("full_name", sa.TEXT(), autoincrement=False, nullable=True),
        schema="mb",
    )
    op.add_column(
        "source_configs",
        sa.Column(
            "warehouse_hash", postgresql.BYTEA(), autoincrement=False, nullable=True
        ),
        schema="mb",
    )

    # Populate old columns from new columns
    connection = op.get_bind()

    # Extract data from new columns
    result = connection.execute(
        sa.text("""
        SELECT source_config_id, location_uri, extract_transform
        FROM mb.source_configs
    """)
    )

    for row in result:
        config_id = row.source_config_id
        location_uri = row.location_uri
        extract_transform = row.extract_transform

        # Extract full_name from extract_transform using regex
        # Pattern to match "from {table_name}" where table_name can contain dots
        match = re.search(r"from\s+([\w\.]+)", extract_transform, re.IGNORECASE)
        full_name = match.group(1) if match else "unknown_table"

        # Create warehouse_hash by hashing the location_uri
        warehouse_hash = hashlib.sha256(location_uri.encode("utf-8")).digest()

        # Update the record
        connection.execute(
            sa.text("""
            UPDATE mb.source_configs 
            SET full_name = :full_name,
                warehouse_hash = :warehouse_hash
            WHERE source_config_id = :config_id
        """),
            {
                "full_name": full_name,
                "warehouse_hash": warehouse_hash,
                "config_id": config_id,
            },
        )

    # Make old columns non-nullable
    op.alter_column("source_configs", "full_name", nullable=False, schema="mb")
    op.alter_column("source_configs", "warehouse_hash", nullable=False, schema="mb")

    # Drop new constraint and create old one
    op.drop_constraint(
        "unique_location_transform", "source_configs", schema="mb", type_="unique"
    )
    op.create_unique_constraint(
        "unique_source_address",
        "source_configs",
        ["full_name", "warehouse_hash"],
        schema="mb",
    )

    # Drop new columns
    op.drop_column("source_configs", "extract_transform", schema="mb")
    op.drop_column("source_configs", "location_uri", schema="mb")
    op.drop_column("source_configs", "location_type", schema="mb")
