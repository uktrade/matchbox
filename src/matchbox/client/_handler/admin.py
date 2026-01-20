"""Admin functions for the client handler."""

from matchbox.client._handler.main import CLIENT, http_retry, url_params
from matchbox.common.dtos import (
    BackendCountableType,
    Group,
    GroupName,
    PermissionGrant,
    ResourceOperationStatus,
)
from matchbox.common.logging import logger

# Group management


@http_retry
def list_groups() -> list[Group]:
    """List all groups."""
    response = CLIENT.get("/admin/groups")
    return [Group(**g) for g in response.json()]


@http_retry
def create_group(
    name: GroupName, description: str | None = None
) -> ResourceOperationStatus:
    """Create a new group."""
    response = CLIENT.post(
        "/admin/groups",
        json={"name": name, "description": description},
    )
    return ResourceOperationStatus(**response.json())


@http_retry
def get_group(name: GroupName) -> Group:
    """Get group details including members."""
    response = CLIENT.get(f"/admin/groups/{name}")
    return Group(**response.json())


@http_retry
def delete_group(name: GroupName, certain: bool = False) -> ResourceOperationStatus:
    """Delete a group."""
    response = CLIENT.delete(f"/admin/groups/{name}", params={"certain": certain})
    return ResourceOperationStatus(**response.json())


@http_retry
def add_user_to_group(group_name: GroupName, user_name: str) -> ResourceOperationStatus:
    """Add a user to a group."""
    response = CLIENT.post(
        f"/admin/groups/{group_name}/members",
        json={"user_name": user_name},  # Note: DTO expects user_name
    )
    return ResourceOperationStatus(**response.json())


@http_retry
def remove_user_from_group(
    group_name: GroupName, user_name: str
) -> ResourceOperationStatus:
    """Remove a user from a group."""
    response = CLIENT.delete(f"/admin/groups/{group_name}/members/{user_name}")
    return ResourceOperationStatus(**response.json())


# System permissions


@http_retry
def get_system_permissions() -> list[PermissionGrant]:
    """Get all system permissions."""
    response = CLIENT.get("/admin/system/permissions")
    return [PermissionGrant(**p) for p in response.json()]


# Miscellanious


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
