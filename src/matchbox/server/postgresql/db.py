"""Matchbox PostgreSQL database connection."""

from contextlib import contextmanager
from typing import Any, Generator, TypeVar, cast

from adbc_driver_postgresql import dbapi as adbc_dbapi
from pydantic import BaseModel, Field
from sqlalchemy import Engine, MetaData, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from matchbox.server.base import MatchboxBackends, MatchboxServerSettings

T = TypeVar("T")


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
    """Matchbox PostgreSQL database connection with connection pooling."""

    def __init__(self, settings: "MatchboxPostgresSettings") -> None:
        """Initialise the database connection.

        Args:
            settings (MatchboxPostgresSettings): The PostgreSQL settings.
        """
        self.settings = settings
        self.engine: Engine | None = None
        self.SessionLocal: sessionmaker | None = None
        self.adbc_connection: adbc_dbapi.Connection | None = None
        self.MatchboxBase: Any = declarative_base(
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

    def connect(self) -> None:
        """Connect to the database and set up SQLAlchemy connection pooling."""
        if not self.engine:
            self.engine = create_engine(
                url=self.connection_string(),
                logging_name="matchbox.engine",
                # Connection pooling settings
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
            )
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )

    def get_engine(self) -> Engine:
        """Get the database engine."""
        if not self.engine:
            self.connect()

        return cast(Engine, self.engine)

    def get_session(self) -> Session:
        """Get a new session."""
        if not self.SessionLocal:
            self.connect()

        return cast(sessionmaker, self.SessionLocal)()

    def get_session_context(self) -> Generator[Session, None, None]:
        """Get a session with automatic cleanup.

        Returns a context manager that will automatically close the session
        when the context is exited.

        Examples:
            ```python
            with MBDB.get_session_context() as session:
                results = session.query(Model).all()
                # Session will be automatically closed after this block
            ```
        """
        if not self.SessionLocal:
            self.connect()

        @contextmanager
        def session_context() -> Generator[Session, None, None]:
            session = cast(sessionmaker, self.SessionLocal)()
            try:
                yield session
            finally:
                session.close()

        return session_context()

    def setup_adbc_pool(self) -> None:
        """Initialize the ADBC source connection for cloning."""
        if self.adbc_connection is None:
            # Create a source connection that will be used to clone from
            self.adbc_connection = adbc_dbapi.connect(
                self.connection_string(driver=False)
            )

    def get_adbc_connection(self) -> "Generator[adbc_dbapi.Connection, None, None]":
        """Get an ADBC connection.

        Returns a context manager that will automatically close the connection
        when the context is exited.

        Examples:
            ```python
            with MBDB.get_adbc_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(...)
            ```
        """
        if self.adbc_connection is None:
            self.setup_adbc_pool()

        @contextmanager
        def connection_context() -> Generator[adbc_dbapi.Connection, None, None]:
            # Clone a new connection from the source
            conn = (
                self.adbc_connection.adbc_clone()
                if self.adbc_connection
                else adbc_dbapi.connect(self.connection_string(driver=False))
            )
            try:
                yield conn
            finally:
                conn.close()

        return connection_context()

    def create_database(self) -> None:
        """Create the database."""
        self.connect()
        with self.get_engine().connect() as conn:
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {self.settings.postgres.db_schema};")
            )
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
            conn.commit()

        self.MatchboxBase.metadata.create_all(self.get_engine())

    def clear_database(self) -> None:
        """Clear the database."""
        self.connect()
        with self.get_engine().connect() as conn:
            conn.execute(
                text(
                    f"DROP SCHEMA IF EXISTS {self.settings.postgres.db_schema} CASCADE;"
                )
            )
            conn.commit()

        # Dispose of all connections in the pool
        if self.engine:
            self.engine.dispose()

        # Close the source ADBC connection if it exists
        if self.adbc_connection:
            self.adbc_connection.close()
            self.adbc_connection = None

        self.create_database()

    def close(self) -> None:
        """Close all connections properly."""
        if self.engine:
            self.engine.dispose()

        if self.adbc_connection:
            self.adbc_connection.close()
            self.adbc_connection = None


# Global database instance -- everything should use this

MBDB = MatchboxDatabase(MatchboxPostgresSettings())
