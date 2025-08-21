"""API dependencies for the Matchbox server."""

import json
import logging
import sys
import time
from base64 import urlsafe_b64decode
from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator, Generator

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Security,
    status,
)
from fastapi.responses import Response
from fastapi.security import APIKeyHeader

from matchbox.common.logging import ASIMFormatter
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)
from matchbox.server.uploads import UploadTracker, settings_to_upload_tracker


class ZipResponse(Response):
    """A response object for a zipped data."""

    media_type = "application/zip"


class ParquetResponse(Response):
    """A response object for returning parquet data."""

    media_type = "application/octet-stream"


SETTINGS: MatchboxServerSettings | None = None
BACKEND: MatchboxDBAdapter | None = None
UPLOAD_TRACKER: UploadTracker | None = None
JWT_HEADER = APIKeyHeader(name="Authorization", auto_error=False)


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


def upload_tracker() -> Generator[UploadTracker, None, None]:
    """Get the upload tracker instance."""
    if UPLOAD_TRACKER is None:
        raise ValueError("Upload tracker not initialized.")
    yield UPLOAD_TRACKER


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Context manager for the FastAPI lifespan events."""
    # Set up the backend
    global SETTINGS
    global BACKEND
    global UPLOAD_TRACKER

    SettingsClass = get_backend_settings(MatchboxServerSettings().backend_type)
    SETTINGS = SettingsClass()
    BACKEND = settings_to_backend(SETTINGS)
    UPLOAD_TRACKER = settings_to_upload_tracker(SETTINGS)

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
UploadTrackerDependency = Annotated[UploadTracker, Depends(upload_tracker)]


def b64_decode(b64_bytes):
    """Add padding and decode b64 bytes."""
    remainder = len(b64_bytes) % 4
    if remainder:
        b64_bytes += b"=" * (4 - remainder)
    return urlsafe_b64decode(b64_bytes)


def validate_jwt(
    settings: SettingsDependency,
    client_token: str = Security(JWT_HEADER),
) -> None:
    """Validate client JWT with server API Key."""
    if not settings.public_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Public Key missing in server configuration.",
            headers={"WWW-Authenticate": "Authorization"},
        )

    if not client_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT required but not provided.",
            headers={"WWW-Authenticate": "Authorization"},
        )

    header_b64, payload_b64, signature_b64 = client_token.encode().split(b".")
    payload = json.loads(b64_decode(payload_b64))

    # Decode to unicode-escape removes \\n encoding for
    # secrets stored in AWS secrets manager.
    public_key = load_pem_public_key(
        settings.public_key.get_secret_value()
        .encode()
        .decode("unicode-escape")
        .encode()
    )

    try:
        public_key.verify(b64_decode(signature_b64), header_b64 + b"." + payload_b64)
    except InvalidSignature as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT invalid.",
            headers={"WWW-Authenticate": "Authorization"},
        ) from e

    if payload["exp"] <= time.time():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT expired.",
            headers={"WWW-Authenticate": "Authorization"},
        )


def authorisation_dependencies(
    settings: SettingsDependency, client_token: str = Security(JWT_HEADER)
):
    """Optional authorisation."""
    if settings.authorisation:
        validate_jwt(settings, client_token)
