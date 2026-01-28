"""Largely auto-generated Alembic configuration entrypoint."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from matchbox.common.datatypes import require
from matchbox.server.postgresql.db import MBDB

fileConfig(
    require(context.config.config_file_name, "Alembic config file name required"),
    disable_existing_loggers=False,
)


def run_migrations_online() -> None:
    """Run migrations based on SQLAlchemy ('offline' refers to raw SQL mode)."""
    config_section = context.config.get_section(context.config.config_ini_section) or {}
    config_section["sqlalchemy.url"] = (
        MBDB.settings.postgres.get_alembic_config().get_main_option("sqlalchemy.url")
    )

    connectable = engine_from_config(
        config_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    def _include_name(name: str, type_: str, _: dict[str, str]) -> bool:
        """Ensure only Matchbox's schema is used to generate diffs."""
        if type_ == "schema":
            return name == MBDB.settings.postgres.db_schema
        else:
            return True

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=MBDB.MatchboxBase.metadata,
            include_schemas=True,
            include_name=_include_name,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


run_migrations_online()
