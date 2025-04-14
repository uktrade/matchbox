"""Matchbox PostgreSQL database connection."""

from contextlib import contextmanager
from typing import Any, Generator

from adbc_driver_postgresql import dbapi as adbc_dbapi
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from pydantic import BaseModel, Field
from sqlalchemy import Engine, MetaData, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from matchbox.common.logging import logger
from matchbox.server.base import MatchboxBackends, MatchboxServerSettings


class MatchboxPostgresCoreSettings(BaseModel):
    """PostgreSQL-specific settings for Matchbox."""

    host: str
    port: int
    user: str
    password: str
    database: str
    db_schema: str


class MatchboxPostgresSettings(MatchboxServerSettings):
    """Settings for the Matchbox PostgreSQL backend.

    Inherits the core settings and adds the PostgreSQL-specific settings.
    """

    backend_type: MatchboxBackends = MatchboxBackends.POSTGRES

    postgres: MatchboxPostgresCoreSettings = Field(
        default_factory=MatchboxPostgresCoreSettings
    )


class MatchboxDatabase:
    """Matchbox PostgreSQL database connection."""

    def __init__(self, settings: MatchboxPostgresSettings):
        """Initialise the database connection."""
        self.settings = settings
        self._engine: Engine | None = None
        self._SessionLocal: sessionmaker | None = None
        self._adbc_pool: QueuePool | None = None
        self._source_adbc_connection: adbc_dbapi.Connection | None = None
        self.MatchboxBase = declarative_base(
            metadata=MetaData(schema=settings.postgres.db_schema)
        )

    def connection_string(self, driver: bool = True) -> str:
        """Get the connection string for PostgreSQL."""
        driver_string = ""
        if driver:
            driver_string = "+psycopg"
        return (
            f"postgresql{driver_string}://{self.settings.postgres.user}:{self.settings.postgres.password}"
            f"@{self.settings.postgres.host}:{self.settings.postgres.port}/"
            f"{self.settings.postgres.database}"
        )

    def _connect(self):
        """Connect to the database."""
        self._engine = create_engine(
            url=self.connection_string(),
            logging_name="matchbox.engine",
            echo=self.settings.log_sql,
        )
        self._SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    def _disconnect(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._SessionLocal = None

    def _connect_adbc(self) -> None:
        self._source_adbc_connection = adbc_dbapi.connect(
            self.connection_string(driver=False)
        )
        self._adbc_pool = QueuePool(
            self._source_adbc_connection.adbc_clone,
        )

    def _disconnect_adbc(self) -> None:
        if self._adbc_pool:
            self._adbc_pool.dispose()
            self._adbc_pool = None

            self._source_adbc_connection.close()
            self._source_adbc_connection = None

    def _reset_connections(self) -> None:
        """Dispose and re-initialise SQLAlchemy and ADBC connection managers."""
        self._disconnect()
        self._connect()

        self._disconnect_adbc()
        self._connect_adbc()

    def get_engine(self) -> Engine:
        """Get the database engine."""
        if not self._engine:
            self._connect()
        return self._engine

    def get_session(self):
        """Get a new session."""
        if not self._SessionLocal:
            self._connect()
        return self._SessionLocal()

    @contextmanager
    def get_adbc_connection(self) -> Generator[adbc_dbapi.Connection, Any, Any]:
        """Get a new ADBC connection.

        The connection must be used within a context manager.
        """
        if not self._adbc_pool:
            self._connect_adbc()

        conn = self._adbc_pool.connect()
        try:
            yield conn.driver_connection
        finally:
            conn.close()

    def create_database(self):
        """Create the database."""
        with self.get_engine().connect() as conn:
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {self.settings.postgres.db_schema};")
            )
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
            conn.commit()

        self.MatchboxBase.metadata.create_all(self.get_engine())

    def clear_database(self):
        """Clear the database."""
        with self.get_engine().connect() as conn:
            conn.execute(
                text(
                    f"DROP SCHEMA IF EXISTS {self.settings.postgres.db_schema} CASCADE;"
                )
            )
            conn.commit()

        self._reset_connections()

        self.create_database()

    def sync_schema(self):
        """Synchronise the database schema with the ORM.

        If any differences are detected, drop and recreate the database.
        """
        engine = self.get_engine()

        # Check if schema exists, create if not
        with engine.connect() as conn:
            schemas = conn.dialect.get_schema_names(conn)
            if self.settings.postgres.db_schema not in schemas:
                self.create_database()
                return

        # Compare schema with ORM, drop and recreate if different
        def _include_name(name: str, type_: str, _: dict[str, str]) -> bool:
            if type_ == "schema":
                return name == self.settings.postgres.db_schema
            else:
                return True

        with engine.connect() as conn:
            opts = {
                "compare_type": True,
                "compare_server_default": True,
                "include_schemas": True,
                "include_names": _include_name,
            }
            context = MigrationContext.configure(conn, opts=opts)

            diff = compare_metadata(context, self.MatchboxBase.metadata)

            if diff:
                logger.warning(
                    "Schema mismatch detected. Dropping and recreating database. \n"
                    f"Diff: {diff}"
                )
                self.clear_database()


# Global database instance -- everything should use this

MBDB = MatchboxDatabase(MatchboxPostgresSettings())
