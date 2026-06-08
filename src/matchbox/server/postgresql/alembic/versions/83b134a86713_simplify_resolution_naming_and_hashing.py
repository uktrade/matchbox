"""Simplify resolution naming and hashing.

Revision ID: 83b134a86713
Revises: 95c0b5c23446
Create Date: 2025-05-12 17:11:45.330677

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from matchbox.common.hash import hash_values

# revision identifiers, used by Alembic.
revision: str = "83b134a86713"
down_revision: str | None = "95c0b5c23446"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def calculate_resolution_hashes(
    resolutions: dict[int, tuple[str, str]],
    relationships: list[tuple[int, int]],
) -> list[tuple[int, bytes]]:
    """Calculate resolution_hash for each resolution based on its type and dependencies.

    Args:
        resolutions: Dict mapping resolution_id -> (name, type)
        relationships: List of (child, parent) tuples for direct parent relationships

    Returns:
        List of (resolution_id, calculated_hash) tuples
    """
    # Build parent relationships
    parents: dict[int, list[int]] = {}
    for child, parent in relationships:
        parents.setdefault(child, []).append(parent)

    # Calculate hashes recursively
    cache: dict[int, bytes] = {}

    def get_hash(rid: int) -> bytes:
        if rid not in cache:
            name, rtype = resolutions[rid]
            if rtype == "dataset":
                cache[rid] = hash_values(name)
            else:
                parent_hashes = sorted(get_hash(p) for p in parents.get(rid, []))
                cache[rid] = hash_values(*parent_hashes, bytes(name, encoding="utf-8"))
        return cache[rid]

    return [(rid, get_hash(rid)) for rid in resolutions]


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Drop the old check constraint
    op.drop_constraint(
        "resolution_type_constraints", "resolutions", schema=schema, type_="check"
    )

    # Update any 'dataset' types to 'source' types
    op.execute(
        """
        UPDATE {schema}.resolutions 
        SET type = 'source'
        WHERE type = 'dataset'
        """
    )

    # Create the new check constraint with updated type values
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source', 'human')",
        schema=schema,
    )

    # Add the new hash column
    op.add_column(
        "resolutions", sa.Column("hash", postgresql.BYTEA(), nullable=True), schema=schema
    )

    # Populate the new hash column based on type
    op.execute(
        """
        UPDATE {schema}.resolutions 
        SET hash = CASE 
            WHEN type = 'model' THEN resolution_hash
            WHEN type = 'source' THEN content_hash
            ELSE NULL
        END
        """
    )

    # Drop the unique constraint on resolution_hash
    op.drop_constraint(
        "resolutions_hash_key", "resolutions", schema=schema, type_="unique"
    )

    # Drop the old hash columns
    op.drop_column("resolutions", "resolution_hash", schema=schema)
    op.drop_column("resolutions", "content_hash", schema=schema)

    # Drop the resolution_name column from source_configs
    op.drop_column("source_configs", "resolution_name", schema=schema)


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Add back the resolution_name column to source_configs as nullable first
    op.add_column(
        "source_configs",
        sa.Column("resolution_name", sa.TEXT(), autoincrement=False, nullable=True),
        schema=schema,
    )

    # Populate source_configs.resolution_name with resolutions.name
    # Use the correct column names based on the ORM
    op.execute(
        """
        UPDATE {schema}.source_configs 
        SET resolution_name = r.name
        FROM {schema}.resolutions r
        WHERE source_configs.resolution_id = r.resolution_id
        """
    )

    # Now make the column NOT NULL after it's been populated
    op.alter_column("source_configs", "resolution_name", nullable=False, schema=schema)

    # Add back the old hash columns
    op.add_column(
        "resolutions",
        sa.Column(
            "content_hash", postgresql.BYTEA(), autoincrement=False, nullable=True
        ),
        schema=schema,
    )
    op.add_column(
        "resolutions",
        sa.Column(
            "resolution_hash", postgresql.BYTEA(), autoincrement=False, nullable=True
        ),
        schema=schema,
    )

    # Load data for hash calculation
    connection = op.get_bind()

    # Get all resolutions
    resolutions = {}
    for rid, name, rtype in connection.execute(
        sa.text(f"SELECT resolution_id, name, type FROM {schema}.resolutions")
    ):
        resolutions[rid] = (name, rtype)

    # Get parent relationships
    relationships = list(
        connection.execute(
            sa.text(f"SELECT child, parent FROM {schema}.resolution_from WHERE level = 1")
        )
    )

    # Calculate new hashes
    hash_results = calculate_resolution_hashes(resolutions, relationships)

    # Update resolution_hash with calculated values
    for resolution_id, calculated_hash in hash_results:
        connection.execute(
            sa.text(
                f"UPDATE {schema}.resolutions SET resolution_hash = :hash "
                "WHERE resolution_id = :id"
            ),
            {"hash": calculated_hash, "id": resolution_id},
        )

    # Set content_hash from hash for sources only
    op.execute(
        """
        UPDATE {schema}.resolutions 
        SET content_hash = CASE 
            WHEN type = 'source' THEN hash
            ELSE NULL
        END
        """
    )

    # Now make resolution_hash NOT NULL after it's been populated
    op.alter_column("resolutions", "resolution_hash", nullable=False, schema=schema)

    # Recreate the unique constraint
    op.create_unique_constraint(
        "resolutions_hash_key", "resolutions", ["resolution_hash"], schema=schema
    )

    # Drop the new hash column
    op.drop_column("resolutions", "hash", schema=schema)

    # Drop the new check constraint
    op.drop_constraint(
        "resolution_type_constraints", "resolutions", schema=schema, type_="check"
    )

    # Update any 'source' types back to 'dataset' types
    op.execute(
        """
        UPDATE {schema}.resolutions 
        SET type = 'dataset'
        WHERE type = 'source'
        """
    )

    # Recreate the old check constraint with original type values
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'dataset', 'human')",
        schema=schema,
    )