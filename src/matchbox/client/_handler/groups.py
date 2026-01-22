"""Group functions for the client handler."""

from matchbox.client._handler.main import CLIENT, http_retry
from matchbox.common.dtos import Group, GroupName, ResourceOperationStatus


@http_retry
def remove_user_from_group(
    group_name: GroupName, user_name: str
) -> ResourceOperationStatus:
    """Remove a user from a group."""
    response = CLIENT.delete(f"/admin/groups/{group_name}/members/{user_name}")
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
def delete_group(name: GroupName, certain: bool = False) -> ResourceOperationStatus:
    """Delete a group."""
    response = CLIENT.delete(f"/admin/groups/{name}", params={"certain": certain})
    return ResourceOperationStatus(**response.json())


@http_retry
def get_group(name: GroupName) -> Group:
    """Get group details including members."""
    response = CLIENT.get(f"/admin/groups/{name}")
    return Group(**response.json())


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


# Group management


@http_retry
def list_groups() -> list[Group]:
    """List all groups."""
    response = CLIENT.get("/admin/groups")
    return [Group(**g) for g in response.json()]
