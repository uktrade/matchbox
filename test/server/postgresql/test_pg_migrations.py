"""Tests migrations for the Matchbox PostgreSQL backend."""

import pytest
from alembic.command import downgrade, upgrade
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

from matchbox.common.factories.scenarios import setup_scenario
from matchbox.common.logging import logger
from matchbox.server.postgresql import MatchboxPostgres, MatchboxPostgresSettings


@pytest.mark.docker
def test_migrations_stairway(
    matchbox_postgres_settings: MatchboxPostgresSettings,
) -> None:
    """Tests that all migrations can be applied and then rolled back in sequence.

    This test runs migrations in their natural order and shows which one fails.
    """
    # Set up revisions in reverse order
    alembic_config: Config = matchbox_postgres_settings.postgres.get_alembic_config()
    revisions_dir = ScriptDirectory.from_config(alembic_config)
    revisions = list(revisions_dir.walk_revisions("base", "heads"))
    revisions.reverse()

    logger.debug(f"Total revisions to test: {len(revisions)}")

    # Reset database to base state first
    downgrade(alembic_config, "base")

    for i, revision in enumerate(revisions):
        revision_id = revision.revision
        down_revision = revision.down_revision or "-1"

        logger.debug(f"Testing migration {i + 1}/{len(revisions)}: {revision_id}")

        try:
            # Apply the migration
            upgrade(alembic_config, revision_id)

            # Roll it back
            downgrade(alembic_config, down_revision)

            # Re-apply it to ensure it works in both directions
            upgrade(alembic_config, revision_id)

            logger.debug(f"✓ Migration {revision_id} passed")
        except Exception as e:  # noqa: BLE001
            pytest.fail(
                f"Migration {i + 1}/{len(revisions)} '{revision_id}' failed: {str(e)}"
            )


@pytest.mark.docker
def test_migrations_stairway_with_data(
    matchbox_postgres: MatchboxPostgres, sqla_sqlite_warehouse: Engine
) -> None:
    """Tests that all migrations can be applied and rolled back with data.

    This shows that the migrations are not only schema changes but also data changes.

    Will start at head, then downgrade to base, step by step.
    """
    with setup_scenario(matchbox_postgres, "link", warehouse=sqla_sqlite_warehouse):
        alembic_config: Config = (
            matchbox_postgres.settings.postgres.get_alembic_config()
        )
        revisions_dir = ScriptDirectory.from_config(alembic_config)
        revisions = list(revisions_dir.walk_revisions("base", "heads"))

        logger.debug(f"Total revisions to test: {len(revisions)}")

        # Set up revisions in reverse order
        for i, revision in enumerate(revisions):
            revision_id = revision.revision
            down_revision = revision.down_revision or "base"

            logger.debug(f"Testing migration {i + 1}/{len(revisions)}: {revision_id}")

            try:
                # Test downgrade
                downgrade(alembic_config, down_revision)

                # Test upgrade back to this revision
                upgrade(alembic_config, revision_id)

                # Downgrade again to test next step
                downgrade(alembic_config, down_revision)

                logger.debug(f"✓ Migration {revision_id} passed")
            except Exception as e:  # noqa: BLE001
                pytest.fail(
                    f"Migration {i + 1}/{len(revisions)} '{revision_id}' failed: "
                    f"{str(e)}"
                )

        # Upgrade to latest migration so scenario can tidy up without erroring
        upgrade(alembic_config, "head")
