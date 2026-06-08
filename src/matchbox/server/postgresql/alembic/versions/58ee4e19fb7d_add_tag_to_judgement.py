"""Add tag to judgement.

Revision ID: 58ee4e19fb7d
Revises: 13095b44ff09
Create Date: 2025-12-05 16:59:15.778813

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "58ee4e19fb7d"
down_revision: str | None = "13095b44ff09"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.add_column(
        "eval_judgements", sa.Column("tag", sa.TEXT(), nullable=True), schema=schema
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_column("eval_judgements", "tag", schema=schema)
