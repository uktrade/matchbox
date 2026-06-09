"""Add upload stage to resolutions.

Revision ID: c774fd4b69f8
Revises: 8c7f757b1046
Create Date: 2025-10-31 07:42:00.669155

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c774fd4b69f8"
down_revision: str | None = "8c7f757b1046"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    upload_stages = sa.Enum(
        "READY", "PROCESSING", "COMPLETE", name="upload_stages", schema=schema
    )
    # Create upload stage
    connection = op.get_bind()
    upload_stages.create(connection)
    op.add_column(
        "resolutions",
        sa.Column("upload_stage", upload_stages, nullable=True),
        schema=schema,
    )

    connection.execute(
        sa.text(f"UPDATE {schema}.resolutions SET upload_stage = :stage"),
        {"stage": "COMPLETE"},
    )

    op.alter_column("resolutions", "upload_stage", nullable=False, schema=schema)

    # Change "content hash" to "fingerprint"
    op.execute(
        f"UPDATE {schema}.resolutions SET hash = decode('', 'hex') WHERE hash IS NULL"
    )

    op.alter_column(
        "resolutions",
        "hash",
        new_column_name="fingerprint",
        schema=schema,
        existing_type=postgresql.BYTEA(),
        nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    upload_stages = sa.Enum(
        "READY", "PROCESSING", "COMPLETE", name="upload_stages", schema=schema
    )
    # Drop upload stage
    op.drop_column("resolutions", "upload_stage", schema=schema)
    upload_stages.drop(op.get_bind())

    # Change "fingerprint" to "content hash"
    op.alter_column(
        "resolutions",
        "fingerprint",
        new_column_name="hash",
        schema=schema,
        existing_type=postgresql.BYTEA(),
        nullable=True,
    )
