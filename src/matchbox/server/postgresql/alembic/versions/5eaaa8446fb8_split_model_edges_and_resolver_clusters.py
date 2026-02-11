"""Split model edges and resolver clusters.

Revision ID: 5eaaa8446fb8
Revises: ecb39d1cc5c2
Create Date: 2026-02-11 17:04:26.738876

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "5eaaa8446fb8"
down_revision: str | None = "ecb39d1cc5c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(
        "resolution_type_constraints",
        "resolutions",
        schema="mb",
        type_="check",
    )
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source', 'resolver')",
        schema="mb",
    )

    # Keep existing model edge rows by renaming the results table.
    op.rename_table("results", "model_edges", schema="mb")
    op.drop_index(op.f("ix_results_resolution"), table_name="model_edges", schema="mb")
    op.create_index(
        "ix_model_edges_resolution",
        "model_edges",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )

    # Rename probabilities to an association table and drop probability payload.
    op.rename_table("probabilities", "resolution_clusters", schema="mb")
    op.drop_index(
        op.f("ix_probabilities_resolution"),
        table_name="resolution_clusters",
        schema="mb",
    )
    op.create_index(
        "ix_resolution_clusters_resolution",
        "resolution_clusters",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )
    op.create_index(
        "ix_resolution_clusters_cluster",
        "resolution_clusters",
        ["cluster_id"],
        unique=False,
        schema="mb",
    )
    op.drop_constraint(
        op.f("valid_probability"),
        "resolution_clusters",
        schema="mb",
        type_="check",
    )
    op.drop_column("resolution_clusters", "probability", schema="mb")

    op.create_table(
        "resolver_configs",
        sa.Column(
            "resolver_config_id",
            sa.BIGINT(),
            sa.Identity(always=False, start=1),
            nullable=False,
        ),
        sa.Column("resolution_id", sa.BIGINT(), nullable=False),
        sa.Column("strategy", sa.TEXT(), nullable=False),
        sa.Column("inputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "thresholds", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["resolution_id"], ["mb.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("resolver_config_id"),
        sa.UniqueConstraint("resolution_id", name="resolver_configs_resolution_key"),
        schema="mb",
    )

    # Preserve closure-table threshold semantics before removing resolutions.truth.
    op.execute(
        """
        UPDATE mb.resolution_from rf
        SET truth_cache = COALESCE(rf.truth_cache, r.truth)
        FROM mb.resolutions r
        WHERE rf.parent = r.resolution_id
        """
    )
    op.drop_column("resolutions", "truth", schema="mb")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "resolution_type_constraints",
        "resolutions",
        schema="mb",
        type_="check",
    )
    # Downgrading to a schema without resolver type, so remove resolver rows first.
    # Cascades clear resolver lineage and resolver-scoped tables.
    op.execute("DELETE FROM mb.resolutions WHERE type = 'resolver'")
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source')",
        schema="mb",
    )

    op.add_column(
        "resolutions",
        sa.Column("truth", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema="mb",
    )

    op.drop_table("resolver_configs", schema="mb")

    op.drop_index(
        "ix_resolution_clusters_cluster",
        table_name="resolution_clusters",
        schema="mb",
    )
    op.drop_index(
        "ix_resolution_clusters_resolution",
        table_name="resolution_clusters",
        schema="mb",
    )
    op.add_column(
        "resolution_clusters",
        sa.Column("probability", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema="mb",
    )
    op.execute(
        "UPDATE mb.resolution_clusters SET probability = 100 WHERE probability IS NULL"
    )
    op.alter_column(
        "resolution_clusters",
        "probability",
        existing_type=sa.SMALLINT(),
        nullable=False,
        schema="mb",
    )
    op.create_check_constraint(
        op.f("valid_probability"),
        "resolution_clusters",
        "probability >= 0 AND probability <= 100",
        schema="mb",
    )
    op.rename_table("resolution_clusters", "probabilities", schema="mb")
    op.create_index(
        op.f("ix_probabilities_resolution"),
        "probabilities",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )

    op.drop_index("ix_model_edges_resolution", table_name="model_edges", schema="mb")
    op.rename_table("model_edges", "results", schema="mb")
    op.create_index(
        op.f("ix_results_resolution"),
        "results",
        ["resolution_id"],
        unique=False,
        schema="mb",
    )
