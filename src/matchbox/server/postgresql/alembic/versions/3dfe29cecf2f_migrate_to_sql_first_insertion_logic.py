"""Migrate to SQL-first insertion logic.

Revision ID: 3dfe29cecf2f
Revises: 58ee4e19fb7d
Create Date: 2026-01-12 10:08:32.246473

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3dfe29cecf2f"
down_revision: str | None = "58ee4e19fb7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    connection = op.get_bind()
    metadata = sa.MetaData(schema="mb")
    pk_space = sa.Table("pk_space", metadata, autoload_with=connection)

    # Migrate next IDs from PKSpace to sequences
    next_ids = connection.execute(
        sa.select(pk_space.c.next_cluster_id, pk_space.c.next_cluster_keys_id)
    ).one_or_none()

    if next_ids:
        connection.execute(
            sa.select(
                sa.func.setval(
                    sa.func.pg_get_serial_sequence("mb.clusters", "cluster_id"),
                    next_ids.next_cluster_id,
                    False,
                ),
            )
        )

        connection.execute(
            sa.select(
                sa.func.setval(
                    sa.func.pg_get_serial_sequence("mb.cluster_keys", "key_id"),
                    next_ids.next_cluster_keys_id,
                    False,
                ),
            )
        )

    # Drop PKSpace
    op.drop_table("pk_space", schema="mb")

    # Create constraint on Contains
    op.create_unique_constraint(
        "contains_unique_root_leaf", "contains", ["root", "leaf"], schema="mb"
    )


def downgrade() -> None:
    """Downgrade schema."""
    connection = op.get_bind()
    metadata = sa.MetaData(schema="mb")

    # Drop constraint on Contains
    op.drop_constraint(
        "contains_unique_root_leaf", "contains", schema="mb", type_="unique"
    )

    # Create PKSpace
    op.create_table(
        "pk_space",
        sa.Column("id", sa.BIGINT(), autoincrement=True, nullable=False),
        sa.Column("next_cluster_id", sa.BIGINT(), autoincrement=False, nullable=False),
        sa.Column(
            "next_cluster_keys_id", sa.BIGINT(), autoincrement=False, nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_space_pkey")),
        schema="mb",
    )

    # Migrate next IDs from sequences to PKSpace
    pk_space = sa.Table("pk_space", metadata, autoload_with=connection)
    next_cluster_id = connection.execute(
        sa.select(
            sa.func.nextval(sa.func.pg_get_serial_sequence("mb.clusters", "cluster_id"))
        )
    ).scalar()
    next_cluster_key_id = connection.execute(
        sa.select(
            sa.func.nextval(sa.func.pg_get_serial_sequence("mb.cluster_keys", "key_id"))
        )
    ).scalar()

    connection.execute(
        sa.insert(pk_space).values(
            id=1,
            next_cluster_id=next_cluster_id,
            next_cluster_keys_id=next_cluster_key_id,
        )
    )
