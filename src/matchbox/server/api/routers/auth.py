"""Authentication routes for the Matchbox server."""

import json
from typing import Annotated

from fastapi import APIRouter, Security
from fastapi.exceptions import HTTPException

from matchbox.common.dtos import AuthStatusResponse, User
from matchbox.server.api.dependencies import (
    JWT_HEADER,
    BackendDependency,
    SettingsDependency,
    b64_decode,
    validate_jwt,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


def get_username_from_token(token: str | None) -> str | None:
    """Extract username from JWT token payload.

    Args:
        token: The JWT token string.

    Returns:
        The username from the 'sub' claim, or None if token is invalid.
    """
    if not token:
        return None

    try:
        parts = token.encode().split(b".")
        if len(parts) != 3:
            return None

        _, payload_b64, _ = parts
        payload = json.loads(b64_decode(payload_b64))
        return payload.get("sub")
    except Exception as _:  # noqa: BLE001
        return None


@router.post("/login")
def login(
    backend: BackendDependency,
    credentials: User,
) -> User:
    """Receive a User with a username and returns it with a user ID."""
    return backend.login(credentials)


@router.get("/status")
async def authentication_status(
    settings: SettingsDependency,
    token: Annotated[str | None, Security(JWT_HEADER)] = None,
) -> AuthStatusResponse:
    """Check authentication status and return user details.

    Returns the current authentication status, including the username
    and JWT token if the user is authenticated.
    """
    if not settings.authorisation:
        return AuthStatusResponse(
            authenticated=False,
            username=None,
            token=None,
        )

    # Try to validate the JWT
    try:
        validate_jwt(settings, token)
        username = get_username_from_token(token)

        return AuthStatusResponse(
            authenticated=True,
            username=username,
            token=token,
        )
    except HTTPException:
        # JWT validation failed
        return AuthStatusResponse(
            authenticated=False,
            username=None,
            token=token,
        )
