"""Authentication functions for the client handler."""

from matchbox.client._handler.main import CLIENT, http_retry
from matchbox.common.dtos import (
    AuthStatusResponse,
    LoginResponse,
)
from matchbox.common.logging import logger


@http_retry
def login() -> LoginResponse:
    """Log into Matchbox."""
    logger.debug("Login attempt using JWT")
    response = CLIENT.post("/auth/login")
    return LoginResponse.model_validate(response.json())


@http_retry
def auth_status() -> AuthStatusResponse:
    """Check authentication status."""
    logger.debug("Checking authentication status")
    response = CLIENT.get("/auth/status")
    return AuthStatusResponse.model_validate(response.json())
