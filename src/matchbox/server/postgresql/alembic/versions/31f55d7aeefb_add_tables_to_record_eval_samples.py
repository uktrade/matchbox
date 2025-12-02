"""Add tables to record eval samples.

Revision ID: 31f55d7aeefb
Revises: c774fd4b69f8
Create Date: 2025-11-28 17:14:13.727281

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import table

# revision identifiers, used by Alembic.
revision: str = "31f55d7aeefb"
down_revision: str | None = "c774fd4b69f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "eval_sample_sets",
        sa.Column("sample_set_id", sa.BIGINT(), nullable=False),
        sa.Column("name", sa.TEXT(), nullable=False),
        sa.Column("description", sa.TEXT(), nullable=True),
        sa.Column("collection_id", sa.BIGINT(), nullable=False),
        sa.ForeignKeyConstraint(
            ["collection_id"], ["mb.collections.collection_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("sample_set_id"),
        sa.UniqueConstraint(
            "name", "collection_id", name="unique_collection_sampleset_name"
        ),
        schema="mb",
    )
    op.create_table(
        "eval_samples",
        sa.Column("sample_set_id", sa.BIGINT(), nullable=False),
        sa.Column("cluster_id", sa.BIGINT(), nullable=False),
        sa.Column("weight", sa.SMALLINT(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cluster_id"], ["mb.clusters.cluster_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["sample_set_id"], ["mb.eval_sample_sets.sample_set_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("sample_set_id", "cluster_id"),
        schema="mb",
    )
    # BREAKING CHANGE! Deletes all existing judgements as they are not associated to
    # a sample set
    op.execute(table("eval_judgements", schema="mb").delete())
    op.add_column(
        "eval_judgements",
        sa.Column("sample_set_id", sa.BIGINT(), nullable=False),
        schema="mb",
    )
    op.create_foreign_key(
        "fk_eval_judgements_sample_sets",
        "eval_judgements",
        "eval_sample_sets",
        ["sample_set_id"],
        ["sample_set_id"],
        source_schema="mb",
        referent_schema="mb",
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "fk_eval_judgements_sample_sets",
        "eval_judgements",
        schema="mb",
        type_="foreignkey",
    )
    op.drop_column("eval_judgements", "sample_set_id", schema="mb")
    op.drop_table("eval_samples", schema="mb")
    op.drop_table("eval_sample_sets", schema="mb")
