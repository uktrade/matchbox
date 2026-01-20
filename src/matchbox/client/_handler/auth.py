"""Authentication functions for the client handler."""

from matchbox.client._handler.main import CLIENT, http_retry
from matchbox.common.dtos import (
    AuthStatusResponse,
    LoginResponse,
    User,
)
from matchbox.common.logging import logger


@http_retry
def login(user_name: str) -> str:
    """Log into Matchbox and return the user name."""
    logger.debug(f"Log in attempt for {user_name}")
    response = CLIENT.post("/auth/login", json=User(user_name=user_name).model_dump())
    return LoginResponse.model_validate(response.json()).user.user_name


@http_retry
def auth_status() -> AuthStatusResponse:
    """Check authentication status and return user details."""
    logger.debug("Checking authentication status")
    response = CLIENT.get("/auth/status")
    return AuthStatusResponse.model_validate(response.json())
