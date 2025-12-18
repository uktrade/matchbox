"""Admin functions for the client handler."""

from matchbox.client._handler.main import CLIENT, http_retry, url_params
from matchbox.common.dtos import (
    AuthStatusResponse,
    BackendCountableType,
    LoginResponse,
    ResourceOperationStatus,
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


@http_retry
def count_backend_items(
    entity: BackendCountableType | None = None,
) -> dict[str, int]:
    """Count the number of various entities in the backend."""
    if entity is not None and entity not in BackendCountableType:
        raise ValueError(
            f"Invalid entity type: {entity}. "
            f"Must be one of {list(BackendCountableType)} "
        )

    log_prefix = "Backend count"
    logger.debug("Counting", prefix=log_prefix)

    res = CLIENT.get("/database/count", params=url_params({"entity": entity}))

    counts = res.json()
    logger.debug(f"Counts: {counts}", prefix=log_prefix)

    return counts


@http_retry
def delete_orphans() -> int:
    """Delete orphaned clusters."""
    logger.debug("Deleting orphans")

    res = CLIENT.delete("/database/orphans")
    return ResourceOperationStatus.model_validate(res.json())
