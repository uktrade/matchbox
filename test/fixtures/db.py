import io
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import boto3
import pytest
import redis
from moto import mock_aws
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine

from matchbox.server.base import MatchboxDatastoreSettings
from matchbox.server.postgresql import MatchboxPostgres, MatchboxPostgresSettings
from matchbox.server.uploads import InMemoryUploadTracker, RedisUploadTracker

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


# Warehouse database fixtures


class DevelopmentSettings(BaseSettings):
    api_port: int = 8000
    datastore_console_port: int = 9003
    datastore_port: int = 9002
    warehouse_port: int = 7654
    postgres_backend_port: int = 9876
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="MB__DEV__",
        env_nested_delimiter="__",
        env_file=Path("environments/development.env"),
        env_file_encoding="utf-8",
    )


@pytest.fixture(scope="session")
def development_settings() -> Generator[DevelopmentSettings, None, None]:
    """Settings for the development environment."""
    settings = DevelopmentSettings()
    yield settings


# Warehouse fixtures


@pytest.fixture(scope="function")
def postgres_warehouse(
    development_settings: DevelopmentSettings,
) -> Generator[Engine, None, None]:
    """Creates an engine for the test warehouse database"""
    user = "warehouse_user"
    password = "warehouse_password"
    host = "localhost"
    database = "warehouse"
    port = development_settings.warehouse_port

    engine = create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    )
    yield engine
    engine.dispose()


@contextmanager
def named_temp_file(filename: str) -> Generator[io.BufferedWriter, None, None]:
    """
    Create a temporary file with a specific name that auto-deletes.

    Args:
        filename: Just the filename (not path) you want to use
    """
    temp_dir = Path(tempfile.gettempdir())
    full_path = temp_dir / filename
    try:
        with full_path.open(mode="wb") as f:
            yield f
    finally:
        if full_path.exists():
            full_path.unlink()


@pytest.fixture(scope="function")
def sqlite_warehouse() -> Generator[Engine, None, None]:
    """Creates an engine for a function-scoped SQLite warehouse database.

    By using a temporary file, produces a URI that can be shared between processes.
    """
    with named_temp_file("db.sqlite") as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        yield engine
        engine.dispose()


@pytest.fixture(scope="function")
def sqlite_in_memory_warehouse() -> Generator[Engine, None, None]:
    """Creates an in-memory engine for a function-scoped SQLite warehouse database."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


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


@pytest.fixture(scope="session")
def matchbox_postgres_settings(
    development_settings: DevelopmentSettings,
    matchbox_datastore: MatchboxDatastoreSettings,
) -> MatchboxPostgresSettings:
    """Settings for the Matchbox PostgreSQL database."""
    return MatchboxPostgresSettings(
        batch_size=250_000,
        postgres={
            "host": "localhost",
            "port": development_settings.postgres_backend_port,
            "user": "matchbox_user",
            "password": "matchbox_password",
            "database": "matchbox",
            "db_schema": "mb",
            "alembic_config": "src/matchbox/server/postgresql/alembic.ini",
        },
        datastore=matchbox_datastore,
    )


@pytest.fixture(scope="function")
def matchbox_postgres(
    matchbox_postgres_settings: MatchboxPostgresSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """The Matchbox PostgreSQL database, cleared."""

    adapter = MatchboxPostgres(settings=matchbox_postgres_settings)

    # Clean up the Matchbox database before each test
    adapter.clear(certain=True)

    yield adapter

    # Clean up the Matchbox database after each test
    adapter.clear(certain=True)


@pytest.fixture(scope="function")
def matchbox_postgres_dropped(
    matchbox_postgres_settings: MatchboxPostgresSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """The Matchbox PostgreSQL database, dropped and recreated."""

    adapter = MatchboxPostgres(settings=matchbox_postgres_settings)

    # Clean up the Matchbox database before each test
    adapter.drop(certain=True)

    yield adapter

    # Clean up the Matchbox database after each test
    adapter.drop(certain=True)


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
) -> Generator[RedisUploadTracker, None, None]:
    """Redis-backed upload tracker."""
    r = redis.Redis.from_url(development_settings.redis_url)
    tracker = RedisUploadTracker(
        redis_url=development_settings.redis_url, expiry_minutes=100
    )

    def empty_tracker():
        for key in r.scan_iter(f"{tracker.key_prefix}*"):
            r.delete(key)

    empty_tracker()
    yield tracker
    empty_tracker()
