"""Authentication routes for the Matchbox server."""

from fastapi import APIRouter, Security

from matchbox.common.dtos import AuthStatusResponse, DefaultUser, LoginResponse
from matchbox.server.api.dependencies import (
    JWT_HEADER,
    BackendDependency,
    CurrentUserDependency,
    SettingsDependency,
    validate_jwt,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login")
def login(
    backend: BackendDependency,
    settings: SettingsDependency,
    client_token: str = Security(JWT_HEADER),
) -> LoginResponse:
    """Upsert the user found in the JWT.

    If in setup mode, will add the user to the admins group.
    """
    user = validate_jwt(settings=settings, client_token=client_token)
    return backend.login(user=user)


@router.get("/status")
async def authentication_status(
    settings: SettingsDependency,
    user: CurrentUserDependency,
) -> AuthStatusResponse:
    """Check authentication status and return user details.

    Returns the current authentication status and user.
    """
    if not settings.authorisation:
        return AuthStatusResponse(authenticated=False)

    is_authenticated = user is not None and user.user_name != DefaultUser.PUBLIC

    return AuthStatusResponse(authenticated=is_authenticated, user=user)
