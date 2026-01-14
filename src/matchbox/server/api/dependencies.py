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
) -> User:
    """Get current user from JWT, or None if auth disabled."""
    if not settings.authorisation:
        return User(user_name="_public", email=None)

    # No token provided: public user
    if not client_token:
        return User(user_name="_public", email=None)

    user = validate_jwt(settings, client_token)

    # Sync user to DB (Upsert)
    return backend.login(user).user


CurrentUserDependency: TypeAlias = Annotated[User | None, Depends(get_current_user)]


class RequiresPermission:
    """Dynamic dependency for checking permissions on a specific resource."""

    @overload
    def __init__(
        self,
        permission: PermissionType,
        *,
        resource: str,
        resource_from_param: None = None,
        allow_public: bool = True,
    ) -> None: ...

    @overload
    def __init__(
        self,
        permission: PermissionType,
        *,
        resource: None = None,
        resource_from_param: str,
        allow_public: bool = True,
    ) -> None: ...

    def __init__(
        self,
        permission: PermissionType,
        *,
        resource: str | None = None,
        resource_from_param: str | None = None,
        allow_public: bool = True,
    ) -> None:
        """Initialise the permission check.

        Args:
            permission: The required permission level.
            resource: A static resource name (e.g. BackendResourceType.SYSTEM).
            resource_from_param: The name of a parameter to look up in either
                the path or query string.
            allow_public: If False, raises an error when auth is disabled or user
                is not authenticated, rather than allowing through.
        """
        if (resource is None) == (resource_from_param is None):
            raise ValueError(
                "Exactly one of 'resource' or 'resource_from_param' must be provided."
            )

        self.permission = permission
        self.static_resource = resource
        self.param_key = resource_from_param
        self.allow_public = allow_public

    def __call__(
        self,
        request: Request,
        backend: BackendDependency,
        settings: SettingsDependency,
        user: CurrentUserDependency,
    ) -> User:
        """Authenticate and authorise the user, returning the User object."""
        # Check if authentication is required
        if not self.allow_public:
            if not settings.authorisation:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=(
                        "This endpoint requires authentication, "
                        "but authentication is disabled"
                    ),
                )

            if user is None or user.user_name == "_public":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

        # Short-circuit if authorisation is disabled
        if not settings.authorisation:
            return user

        # 1. Determine the target resource
        if self.static_resource:
            target_resource = self.static_resource
        else:
            # Check path first, then query parameters
            target_resource = request.path_params.get(
                self.param_key
            ) or request.query_params.get(self.param_key)

        if not target_resource:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Authorisation failed: Resource parameter '{self.param_key}' "
                    "not found.",
                ),
            )

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

        return user


# Pre-configured helpers for common cases
RequireSysAdmin = RequiresPermission(
    PermissionType.ADMIN, resource=BackendResourceType.SYSTEM
)
RequireCollectionAdmin = RequiresPermission(
    PermissionType.ADMIN,
    resource_from_param="collection",
)
RequireCollectionWrite = RequiresPermission(
    PermissionType.WRITE,
    resource_from_param="collection",
)
RequireCollectionRead = RequiresPermission(
    PermissionType.READ,
    resource_from_param="collection",
)
