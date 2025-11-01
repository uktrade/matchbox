"""Add upload stage to resolutions.

Revision ID: c774fd4b69f8
Revises: 59d47cc16734
Create Date: 2025-10-31 07:42:00.669155

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c774fd4b69f8"
down_revision: str | None = "59d47cc16734"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

upload_stages = sa.Enum("READY", "PROCESSING", "COMPLETE", name="upload_stages")


def upgrade() -> None:
    """Upgrade schema."""
    connection = op.get_bind()
    upload_stages.create(connection)
    op.add_column(
        "resolutions",
        sa.Column("upload_stage", upload_stages, nullable=True),
        schema="mb",
    )

    connection.execute(
        sa.text("UPDATE mb.resolutions SET upload_stage = :stage"),
        {"stage": "COMPLETE"},
    )

    op.alter_column("resolutions", "upload_stage", nullable=False, schema="mb")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("resolutions", "upload_stage", schema="mb")
    upload_stages.drop(op.get_bind())
