import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import adbc_driver_postgresql.dbapi as adbc_postgresql
import boto3
import pytest
import redis
from _pytest.fixtures import FixtureRequest
from adbc_driver_sqlite import dbapi as adbc_sqlite
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from moto import mock_aws
from pydantic import Field, SecretBytes, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, MetaData, create_engine, text

from matchbox.server.base import MatchboxDatastoreSettings, MatchboxDBAdapter
from matchbox.server.postgresql import MatchboxPostgres, MatchboxPostgresSettings
from matchbox.server.postgresql.db import MBDB
from matchbox.server.uploads import InMemoryUploadTracker, RedisUploadTracker

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


# Warehouse database fixtures


POSTGRES_HOST = "localhost"
MATCHBOX_USER = "matchbox_user"
MATCHBOX_PASSWORD = "matchbox_password"
WAREHOUSE_USER = "warehouse_user"
WAREHOUSE_PASSWORD = "warehouse_password"


def _worker_database_name(prefix: str, worker_id: str) -> str:
    return f"{prefix}_test_{worker_id}"


def _postgres_url(
    user: str,
    password: str,
    port: int,
    database: str,
) -> str:
    return f"postgresql+psycopg://{user}:{password}@{POSTGRES_HOST}:{port}/{database}"


def _drop_database(
    user: str,
    password: str,
    port: int,
    database: str,
) -> None:
    maintenance_engine = create_engine(
        _postgres_url(user, password, port, "postgres"),
        isolation_level="AUTOCOMMIT",
    )
    quoted_database = f'"{database}"'

    try:
        with maintenance_engine.connect() as connection:
            connection.execute(
                text(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :database
                        AND pid <> pg_backend_pid()
                    """
                ),
                {"database": database},
            )
            connection.execute(
                text(f"DROP DATABASE IF EXISTS {quoted_database} WITH (FORCE)")
            )
    finally:
        maintenance_engine.dispose()


def _recreate_database(user: str, password: str, port: int, database: str) -> None:
    _drop_database(user, password, port, database)
    maintenance_engine = create_engine(
        _postgres_url(user, password, port, "postgres"),
        isolation_level="AUTOCOMMIT",
    )

    try:
        with maintenance_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database}"'))
    finally:
        maintenance_engine.dispose()


class DevelopmentSettings(BaseSettings):
    api_port: int = 8000
    datastore_console_port: int = 9003
    datastore_port: int = 9002
    warehouse_port: int = 7654
    postgres_backend_port: int = 9876
    redis_url: str = "redis://localhost:6379/0"
    private_key: SecretBytes | None = Field(default=None)

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="MB__DEV__",
        env_nested_delimiter="__",
        env_file=Path(".env"),
        env_file_encoding="utf-8",
    )

    @field_validator("private_key", mode="before")
    @classmethod
    def validate_private_key(cls, v: str | bytes | None) -> bytes | None:
        """Validate and normalise PEM private key format."""
        if v is None:
            return v

        # Convert to string if bytes
        key_str: str
        if isinstance(v, bytes):
            key_str = v.decode("ascii")
        elif isinstance(v, SecretBytes):
            key_str = v.get_secret_value().decode("ascii")
        else:
            key_str = v

        # Replace literal \n with actual newlines
        key_str = key_str.replace("\\n", "\n")
        key_bytes = key_str.encode("ascii")

        # Validate by attempting to load (assumes unencrypted key)
        _ = load_pem_private_key(key_bytes, password=None)

        return key_bytes


@pytest.fixture(scope="session")
def development_settings() -> Generator[DevelopmentSettings, None, None]:
    """Settings for the development environment."""
    settings = DevelopmentSettings()
    yield settings


# Warehouse fixtures


def drop_all_tables(engine: Engine) -> None:
    """Drop all tables from a SQLAlchemy engine."""
    metadata = MetaData()
    metadata.reflect(bind=engine)
    metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def worker_postgres_warehouse_database(
    development_settings: DevelopmentSettings,
    worker_id: str,
) -> Generator[str, None, None]:
    """Internal fixture for PostgreSQL warehouse tests.

    Creates one warehouse database per pytest worker for the whole test session, so
    parallel workers do not share warehouse state.
    """
    port = development_settings.warehouse_port
    database = _worker_database_name("warehouse", worker_id)

    _recreate_database(WAREHOUSE_USER, WAREHOUSE_PASSWORD, port, database)
    yield database
    _drop_database(WAREHOUSE_USER, WAREHOUSE_PASSWORD, port, database)


@pytest.fixture(scope="function")
def sqla_postgres_warehouse(
    development_settings: DevelopmentSettings,
    worker_postgres_warehouse_database: str,
) -> Generator[Engine, None, None]:
    """Creates an engine for the test warehouse database"""
    engine = create_engine(
        _postgres_url(
            WAREHOUSE_USER,
            WAREHOUSE_PASSWORD,
            development_settings.warehouse_port,
            worker_postgres_warehouse_database,
        )
    )
    yield engine
    drop_all_tables(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def adbc_postgres_warehouse(
    sqla_postgres_warehouse: Engine,
) -> Generator[adbc_postgresql.Connection, None, None]:
    """Creates an ADBC PostgreSQL warehouse connection.

    Uses the same database as the SQLAlchemy warehouse fixture.
    """
    url = sqla_postgres_warehouse.url
    uri = f"postgresql://{url.username}:{url.password}@{url.host}:{url.port}/{url.database}"

    conn = adbc_postgresql.connect(uri)
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def sqla_sqlite_warehouse(tmp_path: Path) -> Generator[Engine, None, None]:
    """Creates an engine for a function-scoped SQLite warehouse database.

    By using a temporary file, produces a URI that can be shared between processes.
    """
    db_path = tmp_path / "test_warehouse.db"
    engine = create_engine(f"sqlite:///{db_path}")
    yield engine
    drop_all_tables(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def adbc_sqlite_warehouse(
    sqla_sqlite_warehouse: Engine,
) -> Generator[adbc_sqlite.Connection, None, None]:
    """Creates an ADBC SQLite warehouse connection.

    Uses the same database as the SQLAlchemy warehouse fixture.
    """
    db_path = sqla_sqlite_warehouse.url.database
    conn = adbc_sqlite.connect(db_path)
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def sqlite_in_memory_warehouse() -> Generator[Engine, None, None]:
    """Creates an in-memory engine for a function-scoped SQLite warehouse database."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


WarehouseConnectionType: TypeAlias = (
    Engine | adbc_postgresql.Connection | adbc_sqlite.Connection
)


@pytest.fixture
def warehouse(
    request: FixtureRequest,
    sqla_postgres_warehouse: Engine,
    sqla_sqlite_warehouse: Engine,
    sqlite_in_memory_warehouse: Engine,
    adbc_postgres_warehouse: adbc_postgresql.Connection,
    adbc_sqlite_warehouse: adbc_sqlite.Connection,
) -> WarehouseConnectionType:
    """Parametrisable warehouse fixture.

    Use with indirect parametrisation to select which warehouse to test against.
    """
    warehouses = {
        "sqla_postgres": sqla_postgres_warehouse,
        "sqla_sqlite": sqla_sqlite_warehouse,
        "adbc_postgres": adbc_postgres_warehouse,
        "adbc_sqlite": adbc_sqlite_warehouse,
        "sqlite_in_memory": sqlite_in_memory_warehouse,
    }
    return warehouses[request.param]


# Matchbox database fixtures


@pytest.fixture(scope="session")
def matchbox_datastore(
    development_settings: DevelopmentSettings,
) -> MatchboxDatastoreSettings:
    """Settings for the Matchbox datastore."""
    return MatchboxDatastoreSettings(
        host="localhost",
        port=development_settings.datastore_port,
        access_key_id="access_key_id",
        secret_access_key="secret_access_key",
        default_region="eu-west-2",
        cache_bucket_name="cache",
    )


def _build_matchbox_postgres_settings(
    development_settings: DevelopmentSettings,
    matchbox_datastore: MatchboxDatastoreSettings,
    database: str,
) -> MatchboxPostgresSettings:
    return MatchboxPostgresSettings(
        batch_size=250_000,
        postgres={
            "host": POSTGRES_HOST,
            "port": development_settings.postgres_backend_port,
            "user": MATCHBOX_USER,
            "password": MATCHBOX_PASSWORD,
            "database": database,
            "db_schema": "mb",
            "alembic_config": "src/matchbox/server/postgresql/alembic.ini",
        },
        datastore=matchbox_datastore,
    )


@pytest.fixture(scope="session")
def worker_matchbox_postgres(
    development_settings: DevelopmentSettings,
    matchbox_datastore: MatchboxDatastoreSettings,
    worker_id: str,
) -> Generator[MatchboxPostgres, None, None]:
    """Internal fixture backing the default Matchbox PostgreSQL backend.

    Creates one MatchboxPostgres adapter per pytest worker for the whole test
    session, backed by a worker-specific database. matchbox_postgres wraps this
    fixture and clears it before each test.
    """
    port = development_settings.postgres_backend_port
    database = _worker_database_name("matchbox", worker_id)
    _recreate_database(MATCHBOX_USER, MATCHBOX_PASSWORD, port, database)

    settings = _build_matchbox_postgres_settings(
        development_settings=development_settings,
        matchbox_datastore=matchbox_datastore,
        database=database,
    )

    try:
        with MBDB.settings_scope(settings):
            yield MatchboxPostgres(settings=settings)
    finally:
        _drop_database(MATCHBOX_USER, MATCHBOX_PASSWORD, port, database)


@pytest.fixture(scope="function")
def matchbox_postgres(
    worker_matchbox_postgres: MatchboxPostgres,
) -> MatchboxPostgres:
    """Default Matchbox PostgreSQL fixture for parallel tests.

    Reuses the worker-local Matchbox database from worker_matchbox_postgres and
    clears all Matchbox state before each test, giving isolated tests that are safe
    to run in parallel.
    """
    worker_matchbox_postgres.clear(certain=True)
    return worker_matchbox_postgres


@pytest.fixture(scope="function")
def serial_matchbox_postgres(
    development_settings: DevelopmentSettings,
    matchbox_datastore: MatchboxDatastoreSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """Matchbox PostgreSQL fixture for tests marked serial.

    Connects to the fixed matchbox database name instead of a worker-specific one.
    Use this only in serial tests that must share that database name. The database
    is cleared before each test.
    """
    settings = _build_matchbox_postgres_settings(
        development_settings=development_settings,
        matchbox_datastore=matchbox_datastore,
        database="matchbox",
    )
    with MBDB.settings_scope(settings):
        adapter = MatchboxPostgres(settings=settings)
        adapter.clear(certain=True)
        yield adapter


# Mock AWS fixtures


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "eu-west-2"


@pytest.fixture(scope="function")
def s3(aws_credentials: None) -> Generator[S3Client, None, None]:
    """Return a mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3", region_name="eu-west-2")


# Upload trackers


@pytest.fixture(scope="function")
def upload_tracker_in_memory() -> Generator[InMemoryUploadTracker, None, None]:
    """In-memory upload tracker."""
    tracker = InMemoryUploadTracker()
    yield tracker


@pytest.fixture(scope="function")
def upload_tracker_redis(
    development_settings: DevelopmentSettings,
    worker_id: str,
) -> Generator[RedisUploadTracker, None, None]:
    """Redis-backed upload tracker."""
    r = redis.Redis.from_url(development_settings.redis_url)
    tracker = RedisUploadTracker(
        redis_url=development_settings.redis_url,
        expiry_minutes=100,
        key_space=f"upload:{worker_id}",
    )

    def empty_tracker() -> None:
        for key in r.scan_iter(f"{tracker.key_prefix}*"):
            r.delete(key)

    empty_tracker()
    yield tracker
    empty_tracker()


# Backends

BACKENDS = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(scope="function")
def backend_instance(request: pytest.FixtureRequest, backend: str) -> MatchboxDBAdapter:
    """Create a fresh backend instance for each test."""
    backend_obj = request.getfixturevalue(backend)
    backend_obj.clear(certain=True)
    return backend_obj
