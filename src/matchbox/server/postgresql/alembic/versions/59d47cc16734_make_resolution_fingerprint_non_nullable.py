"""Make resolution fingerprint non-nullable.

Revision ID: 59d47cc16734
Revises: 8c7f757b1046
Create Date: 2025-10-28 08:13:08.238818

"""

from typing import Sequence

from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "59d47cc16734"
down_revision: str | None = "8c7f757b1046"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Fill NULL values in 'hash' with empty byte string
    op.execute("UPDATE mb.resolutions SET hash = decode('', 'hex') WHERE hash IS NULL")

    op.alter_column(
        "resolutions",
        "hash",
        new_column_name="fingerprint",
        schema="mb",
        existing_type=postgresql.BYTEA(),
        nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "resolutions",
        "fingerprint",
        new_column_name="hash",
        schema="mb",
        existing_type=postgresql.BYTEA(),
        nullable=True,
    )
