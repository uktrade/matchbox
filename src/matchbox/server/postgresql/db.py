"""Matchbox PostgreSQL database connection."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from adbc_driver_postgresql import dbapi as adbc_dbapi
from alembic import command
from alembic.config import Config
from pydantic import BaseModel, Field
from sqlalchemy import (
    URL,
    Engine,
    MetaData,
    Table,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import PoolProxiedConnection, QueuePool

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
    alembic_config: Path = Field(
        default=Path("src/matchbox/server/postgresql/alembic.ini")
    )

    def get_alembic_config(self) -> Config:
        """Get the Alembic config."""
        config = Config(self.alembic_config)
        db_url = URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        config.set_main_option(
            "sqlalchemy.url", db_url.render_as_string(hide_password=False)
        )
        return config


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
        self.alembic_config = settings.postgres.get_alembic_config()

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
            url=self.connection_string(), logging_name="matchbox.engine", echo=False
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

    def get_session(self) -> Session:
        """Get a new session."""
        if not self._SessionLocal:
            self._connect()
        return self._SessionLocal()

    @contextmanager
    def get_adbc_connection(self) -> Generator[PoolProxiedConnection, Any, Any]:
        """Get a new ADBC connection wrapped by a SQLAlchemy pool proxy.

        The connection must be used within a context manager.
        """
        if not self._adbc_pool:
            self._connect_adbc()

        conn = self._adbc_pool.connect()
        try:
            yield conn
        finally:
            conn.close()

    def run_migrations(self):
        """Create the database and all tables expected in the schema."""
        alembic_version = self._look_for_alembic_version()
        engine = self.get_engine()
        logger.info("Determinded alembic in use so upgrading to head")
        if alembic_version is not None:
            command.upgrade(self.alembic_config, "head")
        else:
            logger.info(
                "Determinded alembic not in use so dropping schema if it "
                "exists prior to upgrading to head. "
            )
            with engine.connect() as conn:
                conn.execute(text("DROP SCHEMA IF EXISTS mb CASCADE;"))
                conn.commit()
            command.upgrade(self.alembic_config, "head")

    def clear_database(self):
        """Delete all rows in every table in the database schema.

        - TRUNCATE tables that are part of the core ORM (preserves structure)
        - DROP tables that are not in the ORM (removes temporary/test tables)
        """
        schema_name = self.MatchboxBase.metadata.schema
        engine = self.get_engine()  # Get the engine

        with self.get_session() as session:
            try:
                # Get all tables that actually exist in the database
                discovered_metadata = MetaData(schema=schema_name)
                discovered_metadata.reflect(
                    bind=engine, schema=schema_name
                )  # Use engine here

                # Get table names from the core ORM
                orm_table_names = {
                    table.name for table in self.MatchboxBase.metadata.tables.values()
                }

                # Categorise discovered tables
                tables_to_truncate: list[Table] = []
                tables_to_drop: list[Table] = []

                for table in discovered_metadata.tables.values():
                    if table.name == "alembic_version":
                        # Alembic version is in public but just in case
                        continue
                    elif table.name in orm_table_names:
                        tables_to_truncate.append(table)
                    else:
                        tables_to_drop.append(table)

                # TRUNCATE core ORM tables (preserves schema structure)
                if tables_to_truncate:
                    for table in tables_to_truncate:
                        session.execute(
                            text(f"TRUNCATE TABLE {table.fullname} CASCADE")
                        )

                # DROP temporary/test tables (removes them completely)
                if tables_to_drop:
                    for table in tables_to_drop:
                        session.execute(text(f"DROP TABLE {table.fullname} CASCADE"))

                session.commit()

            except Exception as e:
                logger.error(f"Error clearing database: {e}")
                session.rollback()
                raise

    def drop_database(self):
        """Drop all tables in the database schema and re-recreate them."""
        command.downgrade(self.alembic_config, "base")
        command.upgrade(self.alembic_config, "head")

    def _look_for_alembic_version(self) -> bool:
        engine = self.get_engine()
        inspector = inspect(engine)
        alembic_version_table = "alembic_version" in inspector.get_table_names(
            schema="public"
        )
        if alembic_version_table:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT version_num FROM public.alembic_version;")
                )
                alembic_version = result.scalar()
        else:
            alembic_version = None
        return alembic_version

    @property
    def sorted_tables(self) -> list[Table]:
        """Return a list of SQLAlchemy tables in order of creation."""
        return self.MatchboxBase.metadata.sorted_tables


# Global database instance -- everything should use this

MBDB = MatchboxDatabase(MatchboxPostgresSettings())
