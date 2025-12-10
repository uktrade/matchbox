"""API dependencies for the Matchbox server."""

import json
import logging
import sys
import time
from base64 import urlsafe_b64decode
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypeAlias, overload

from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import Response
from fastapi.security import APIKeyHeader

from matchbox.common.dtos import (
    BackendResourceType,
    PermissionType,
    User,
)
from matchbox.common.logging import get_formatter, logger
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)
from matchbox.server.uploads import UploadTracker, settings_to_upload_tracker

SETTINGS: MatchboxServerSettings | None = None
BACKEND: MatchboxDBAdapter | None = None
UPLOAD_TRACKER: UploadTracker | None = None
SETUP_MODE: bool = False  # secure failsafe
JWT_HEADER = APIKeyHeader(name="Authorization", auto_error=False)


class ZipResponse(Response):
    """A response object for a zipped data."""

    media_type = "application/zip"


class ParquetResponse(Response):
    """A response object for returning parquet data."""

    media_type = "application/octet-stream"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Context manager for the FastAPI lifespan events."""
    # Set up the backend
    global SETTINGS
    global BACKEND
    global UPLOAD_TRACKER
    global SETUP_MODE

    SettingsClass = get_backend_settings(MatchboxServerSettings().backend_type)
    SETTINGS = SettingsClass()
    BACKEND = settings_to_backend(SETTINGS)
    UPLOAD_TRACKER = settings_to_upload_tracker(SETTINGS)

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
        handler.setFormatter(get_formatter())

        logger = logging.getLogger(logger_name)
        logger.setLevel(BACKEND.settings.log_level)
        # Remove any existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        logger.addHandler(handler)

    # Set SQLAlchemy loggers
    for sql_logger in ["sqlalchemy", "sqlalchemy.engine"]:
        logging.getLogger(sql_logger).setLevel("WARNING")

    # 2. Check DB state once on startup
    if BACKEND.users.count() == 0:
        logger.warning(
            "No users found. Server is in SETUP MODE. "
            "The next user to login will be an admin."
        )
        SETUP_MODE = True
    else:
        SETUP_MODE = False

    yield

    del SETTINGS
    del BACKEND


def backend() -> Generator[MatchboxDBAdapter, None, None]:
    """Get the backend instance."""
    if BACKEND is None:
        raise ValueError("Backend not initialised.")
    yield BACKEND


def settings() -> Generator[MatchboxServerSettings, None, None]:
    """Get the settings instance."""
    if SETTINGS is None:
        raise ValueError("Settings not initialised.")
    yield SETTINGS


def upload_tracker() -> Generator[UploadTracker, None, None]:
    """Get the upload tracker instance."""
    if UPLOAD_TRACKER is None:
        raise ValueError("Upload tracker not initialised.")
    yield UPLOAD_TRACKER


BackendDependency: TypeAlias = Annotated[MatchboxDBAdapter, Depends(backend)]
SettingsDependency: TypeAlias = Annotated[MatchboxServerSettings, Depends(settings)]
UploadTrackerDependency: TypeAlias = Annotated[UploadTracker, Depends(upload_tracker)]


def setup_mode(backend: BackendDependency) -> Generator[bool, None, None]:
    """Returns whether setup mode is active.

    Setup mode indicates the user table is empty, and is exited
    as soon as the first user is created.
    """
    global SETUP_MODE

    if SETUP_MODE is False:
        yield SETUP_MODE
        return

    if backend.users.count() == 0:
        yield True
    else:
        SETUP_MODE = False
        yield SETUP_MODE


SetupModeDependency: TypeAlias = Annotated[bool, Depends(setup_mode)]


def b64_decode(b64_bytes: bytes) -> bytes:
    """Add padding and decode b64 bytes."""
    remainder = len(b64_bytes) % 4
    if remainder:
        b64_bytes += b"=" * (4 - remainder)
    return urlsafe_b64decode(b64_bytes)


def validate_jwt(
    settings: SettingsDependency,
    client_token: str = Depends(JWT_HEADER),
) -> User:
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

    try:
        parts = client_token.encode().split(b".")
        if len(parts) != 3:
            raise ValueError("JWT must have 3 parts")
        header_b64, payload_b64, signature_b64 = parts

        payload: dict[str, Any] = json.loads(b64_decode(payload_b64))

        public_key = load_pem_public_key(settings.public_key.get_secret_value())

        public_key.verify(b64_decode(signature_b64), header_b64 + b"." + payload_b64)

        if payload["exp"] <= time.time():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT expired.",
                headers={"WWW-Authenticate": "Authorization"},
            )

    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.exception(f"Invalid JWT. Token: {client_token}")
        # Anything else is an invalid token -> 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT invalid.",
            headers={"WWW-Authenticate": "Authorization"},
        ) from e

    return User(user_name=payload["sub"], email=payload.get("email"))


def get_current_user(
    settings: SettingsDependency,
    backend: BackendDependency,
    client_token: str = Security(JWT_HEADER),
) -> User | None:
    """Get current user from JWT, or None if auth disabled."""
    if not settings.authorisation:
        return None

    user = validate_jwt(settings, client_token)

    # Sync user to DB (Upsert)
    return backend.login(user)


CurrentUserDependency: TypeAlias = Annotated[User | None, Depends(get_current_user)]


class RequiresPermission:
    """Dynamic dependency for checking permissions on a specific resource."""

    @overload
    def __init__(
        self,
        permission: PermissionType,
        *,
        resource: str,
        resource_from_path: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        permission: PermissionType,
        *,
        resource: None = None,
        resource_from_path: str,
    ) -> None: ...

    def __init__(
        self,
        permission: PermissionType,
        *,
        resource: str | None = None,
        resource_from_path: str | None = None,
    ) -> None:
        """Initialise the permission check."""
        if resource is None and resource_from_path is None:
            raise ValueError(
                "Either 'resource' or 'resource_from_path' must be provided"
            )
        if resource is not None and resource_from_path is not None:
            raise ValueError("Cannot specify both 'resource' and 'resource_from_path'")

        self.permission = permission
        self.static_resource = resource
        self.path_param_key = resource_from_path

    def __call__(
        self,
        request: Request,
        backend: BackendDependency,
        settings: SettingsDependency,
        user: CurrentUserDependency,
    ) -> None:
        """Authenticate and authorise the user."""
        # Short-circuit if authorisation is disabled
        if not settings.authorisation:
            return

        # 1. Determine the target resource
        if self.static_resource:
            target_resource = self.static_resource
        elif self.path_param_key:
            target_resource = request.path_params.get(self.path_param_key)
            if not target_resource:
                # Verify path param exists to prevent dev errors
                raise ValueError(
                    f"Path parameter '{self.path_param_key}' not found in request"
                )
        else:
            raise ValueError("Must provide either resource or resource_from_path")

        # 2. Check authentication
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        # 3. Check authorisation
        if not backend.check_permission(
            user.user_name, self.permission, target_resource
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Permission denied: requires {self.permission} "
                    f"access on '{target_resource}'",
                ),
            )


# Pre-configured helpers for common cases
RequireSysAdmin = RequiresPermission(
    PermissionType.ADMIN, resource=BackendResourceType.SYSTEM
)
