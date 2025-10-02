"""Adds Collections and Versions to scope Resolutions.

Revision ID: f500f7d832fe
Revises: c4cb937d00f4
Create Date: 2025-09-22 07:54:35.453669

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f500f7d832fe"
down_revision: str | None = "c4cb937d00f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create collections table
    op.create_table(
        "collections",
        sa.Column("collection_id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("name", sa.TEXT(), nullable=False),
        sa.PrimaryKeyConstraint("collection_id"),
        sa.UniqueConstraint("name", name="collections_name_key"),
        schema="mb",
    )

    # Create versions table
    op.create_table(
        "versions",
        sa.Column("version_id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("collection_id", sa.BIGINT(), nullable=False),
        sa.Column("name", sa.TEXT(), nullable=False),
        sa.Column("is_mutable", sa.BOOLEAN(), nullable=True),
        sa.Column("is_default", sa.BOOLEAN(), nullable=True),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["mb.collections.collection_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("version_id"),
        sa.UniqueConstraint("collection_id", "name", name="unique_version_name"),
        sa.UniqueConstraint("collection_id", "version_id", name="unique_version_id"),
        schema="mb",
    )

    # Create the partial index for default versions
    op.create_index(
        "ix_default_version_collection",
        "versions",
        ["collection_id"],
        unique=True,
        schema="mb",
        postgresql_where=sa.text("is_default = true"),
    )

    # Add version_id column to resolutions (initially nullable for data migration)
    op.add_column(
        "resolutions", sa.Column("version_id", sa.BIGINT(), nullable=True), schema="mb"
    )

    # Data migration: Create default collection and version for existing resolutions
    # Only do this if there are existing resolutions
    bind = op.get_bind()
    existing_resolutions = bind.execute(
        sa.text("SELECT COUNT(*) FROM mb.resolutions")
    ).scalar()

    if existing_resolutions > 0:
        # Create default collection
        collection_result = bind.execute(
            sa.text(
                "INSERT INTO mb.collections (name) VALUES ('default') "
                "RETURNING collection_id"
            )
        )
        collection_id = collection_result.scalar()

        # Create default version
        version_result = bind.execute(
            sa.text("""
            INSERT INTO mb.versions (collection_id, name, is_mutable, is_default)
            VALUES (:collection_id, 'v1', false, true)
            RETURNING version_id
            """),
            {"collection_id": collection_id},
        )
        version_id = version_result.scalar()

        # Associate all existing resolutions with the new default version
        bind.execute(
            sa.text("""
            UPDATE mb.resolutions
            SET version_id = :version_id
            WHERE version_id IS NULL
            """),
            {"version_id": version_id},
        )

    # Now make version_id non-nullable
    op.alter_column(
        "resolutions",
        "version_id",
        existing_type=sa.BIGINT(),
        nullable=False,
        schema="mb",
    )

    # Update unique constraints on resolutions
    op.drop_constraint(
        op.f("resolutions_name_key"), "resolutions", schema="mb", type_="unique"
    )
    op.create_unique_constraint(
        "resolutions_name_key", "resolutions", ["version_id", "name"], schema="mb"
    )

    # Create foreign key constraint for version_id
    op.create_foreign_key(
        "resolutions_version_fkey",
        "resolutions",
        "versions",
        ["version_id"],
        ["version_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop foreign key constraint
    op.drop_constraint(
        "resolutions_version_fkey", "resolutions", schema="mb", type_="foreignkey"
    )

    # Revert unique constraint changes
    op.drop_constraint(
        "resolutions_name_key", "resolutions", schema="mb", type_="unique"
    )
    op.create_unique_constraint(
        op.f("resolutions_name_key"),
        "resolutions",
        ["name"],
        schema="mb",
        postgresql_nulls_not_distinct=False,
    )

    # Drop the version_id column
    op.drop_column("resolutions", "version_id", schema="mb")

    # Drop the partial index
    op.drop_index(
        "ix_default_version_collection",
        table_name="versions",
        schema="mb",
        postgresql_where=sa.text("is_default = true"),
    )

    # Drop the tables
    op.drop_table("versions", schema="mb")
    op.drop_table("collections", schema="mb")
