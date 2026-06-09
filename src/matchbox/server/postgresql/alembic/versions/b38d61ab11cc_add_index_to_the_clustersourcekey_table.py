"""Add index to the ClusterSourceKey table.

Revision ID: b38d61ab11cc
Revises: 7a2d1b10ac0f
Create Date: 2025-08-21 15:21:52.224388

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b38d61ab11cc"
down_revision: str | None = "7a2d1b10ac0f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.create_index(
        "ix_cluster_keys_source_config_id",
        "cluster_keys",
        ["source_config_id"],
        unique=False,
        schema=schema,
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    op.drop_index(
        "ix_cluster_keys_source_config_id", table_name="cluster_keys", schema=schema
    )
