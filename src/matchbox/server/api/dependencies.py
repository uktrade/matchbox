"""API dependencies for the Matchbox server."""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Generator

import jwt
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Security,
    status,
)
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from jwt.exceptions import InvalidSignatureError

from matchbox.common.jwt import generate_json_web_token
from matchbox.common.logging import ASIMFormatter
from matchbox.server.api.cache import MetadataStore
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)


class ParquetResponse(Response):
    """A response object for returning parquet data."""

    media_type = "application/octet-stream"


SETTINGS: MatchboxServerSettings | None = None
BACKEND: MatchboxDBAdapter | None = None
METADATA_STORE = MetadataStore(expiry_minutes=30)
JWT_HEADER = APIKeyHeader(name="Authorization")
ALGORITHM = "HS256"


def backend() -> Generator[MatchboxDBAdapter, None, None]:
    """Get the backend instance."""
    if BACKEND is None:
        raise ValueError("Backend not initialized.")
    yield BACKEND


def settings() -> Generator[MatchboxServerSettings, None, None]:
    """Get the settings instance."""
    if SETTINGS is None:
        raise ValueError("Settings not initialized.")
    yield SETTINGS


def metadata_store() -> Generator[MetadataStore, None, None]:
    """Get the metadata store instance."""
    if METADATA_STORE is None:
        raise ValueError("Metadata store not initialized.")
    yield METADATA_STORE


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Context manager for the FastAPI lifespan events."""
    # Set up the backend
    global SETTINGS
    global BACKEND

    SettingsClass = get_backend_settings(MatchboxServerSettings().backend_type)
    SETTINGS = SettingsClass()
    BACKEND = settings_to_backend(SETTINGS)

    # Define common formatter
    formatter = ASIMFormatter()

    # Configure loggers with the same handler and formatter
    loggers_to_configure = [
        "matchbox",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "uvicorn.asgi",
        "fastapi",
    ]

    for logger_name in loggers_to_configure:
        # Configure handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(BACKEND.settings.log_level)
        handler.setFormatter(formatter)

        logger = logging.getLogger(logger_name)
        logger.setLevel(BACKEND.settings.log_level)
        # Remove any existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        logger.addHandler(handler)

    # Set SQLAlchemy loggers
    for sql_logger in ["sqlalchemy", "sqlalchemy.engine"]:
        logging.getLogger(sql_logger).setLevel("WARNING")

    yield

    del SETTINGS
    del BACKEND


BackendDependency = Annotated[MatchboxDBAdapter, Depends(backend)]
SettingsDependency = Annotated[MatchboxServerSettings, Depends(settings)]
MetadataStoreDependency = Annotated[MetadataStore, Depends(metadata_store)]


def validate_jwt(
    settings: SettingsDependency,
    client_token: str = Security(JWT_HEADER),
) -> None:
    """Validate client JWT with server API Key."""
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing in server configuration.",
            headers={"WWW-Authenticate": "Authorization"},
        )

    if not client_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT required but not provided.",
            headers={"WWW-Authenticate": "Authorization"},
        )

    try:
        payload = jwt.decode(
            client_token, settings.api_key.get_secret_value(), algorithms=ALGORITHM
        )
    except InvalidSignatureError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT invalid.",
            headers={"WWW-Authenticate": "Authorization"},
        ) from e

    if payload["exp"] < time.time():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT expired.",
            headers={"WWW-Authenticate": "Authorization"},
        )

    if client_token != generate_json_web_token(
        sub=payload["sub"],
        private_key=settings.api_key.get_secret_value(),
        exp=payload["exp"],
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT invalid.",
            headers={"WWW-Authenticate": "Authorization"},
        )
