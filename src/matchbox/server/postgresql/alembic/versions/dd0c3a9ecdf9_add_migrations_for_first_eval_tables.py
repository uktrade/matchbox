"""Add migrations for first eval tables.

Revision ID: dd0c3a9ecdf9
Revises: 3754ae042254
Create Date: 2025-05-14 11:33:54.343350

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dd0c3a9ecdf9"
down_revision: str | None = "3754ae042254"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.create_table(
        "users",
        sa.Column("user_id", sa.BIGINT(), nullable=False),
        sa.Column("name", sa.TEXT(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("name", name="user_name_unique"),
        schema=schema,
    )
    op.create_table(
        "eval_judgements",
        sa.Column("judgement_id", sa.BIGINT(), nullable=False),
        sa.Column("user_id", sa.BIGINT(), nullable=False),
        sa.Column("endorsed_cluster_id", sa.BIGINT(), nullable=False),
        sa.Column("shown_cluster_id", sa.BIGINT(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["endorsed_cluster_id"], [f"{schema}.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["shown_cluster_id"], [f"{schema}.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], [f"{schema}.users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("judgement_id"),
        schema=schema,
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_table("eval_judgements", schema=schema)
    op.drop_table("users", schema=schema)
