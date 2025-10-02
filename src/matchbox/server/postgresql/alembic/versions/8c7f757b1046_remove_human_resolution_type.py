"""Remove human resolution type.

Revision ID: 8c7f757b1046
Revises: f500f7d832fe
Create Date: 2025-09-29 15:57:07.548653

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c7f757b1046"
down_revision: str | None = "f500f7d832fe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop the old constraint
    op.drop_constraint(
        "resolution_type_constraints", "resolutions", type_="check", schema="mb"
    )

    # Update existing rows from 'dataset' to 'source'
    op.execute(
        """
        UPDATE mb.resolutions
        SET type = 'source'
        WHERE type = 'dataset'
        """
    )

    # Create the new constraint with only 'model' and 'source'
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source')",
        schema="mb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop the new constraint
    op.drop_constraint(
        "resolution_type_constraints", "resolutions", type_="check", schema="mb"
    )

    # Revert 'source' back to 'dataset'
    op.execute(
        """
        UPDATE mb.resolutions
        SET type = 'dataset'
        WHERE type = 'source'
        """
    )
    # Recreate the old constraint with 'model', 'dataset', and 'human'
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'dataset', 'human')",
        schema="mb",
    )
