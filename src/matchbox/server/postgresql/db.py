"""Matchbox PostgreSQL database connection."""

from adbc_driver_postgresql import dbapi as adbc_dbapi
from pydantic import BaseModel, Field
from sqlalchemy import Engine, MetaData, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from matchbox.server.base import MatchboxBackends, MatchboxSettings


class MatchboxPostgresCoreSettings(BaseModel):
    """PostgreSQL-specific settings for Matchbox."""

    host: str
    port: int
    user: str
    password: str
    database: str
    db_schema: str


class MatchboxPostgresSettings(MatchboxSettings):
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
        self.engine: Engine | None = None
        self.SessionLocal: sessionmaker | None = None
        self.adbc_connection: adbc_dbapi.Connection | None = None
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

    def connect(self):
        """Connect to the database."""
        if not self.engine:
            self.engine = create_engine(
                url=self.connection_string(), logging_name="mb_pg_db"
            )
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )

    def connect_adbc(self):
        """Connect to the database using ADBC."""
        if not self.adbc_connection:
            self.adbc_connection = adbc_dbapi.connect(
                self.connection_string(driver=False)
            )

    def get_engine(self) -> Engine:
        """Get the database engine."""
        if not self.engine:
            self.connect()
        return self.engine

    def get_session(self):
        """Get a new session."""
        if not self.SessionLocal:
            self.connect()
        return self.SessionLocal()

    def get_adbc_connection(self) -> adbc_dbapi.Connection:
        """Get the ADBC connection."""
        if not self.adbc_connection:
            self.connect_adbc()
        return self.adbc_connection

    def create_database(self):
        """Create the database."""
        self.connect()
        with self.engine.connect() as conn:
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {self.settings.postgres.db_schema};")
            )
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
            conn.commit()

        self.MatchboxBase.metadata.create_all(self.engine)

    def clear_database(self):
        """Clear the database."""
        self.connect()
        with self.engine.connect() as conn:
            conn.execute(
                text(
                    f"DROP SCHEMA IF EXISTS {self.settings.postgres.db_schema} CASCADE;"
                )
            )
            conn.commit()

        self.engine.dispose()

        self.create_database()


# Global database instance -- everything should use this

MBDB = MatchboxDatabase(MatchboxPostgresSettings())
