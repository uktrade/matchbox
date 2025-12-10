"""Admin routes for user and group management."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    Group,
    GroupName,
    PermissionGrant,
    PermissionType,
    ResourceOperationStatus,
    User,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxGroupAlreadyExistsError,
    MatchboxSystemGroupError,
)
from matchbox.server.api.dependencies import BackendDependency, RequireSysAdmin

router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(RequireSysAdmin)]
)

# Group management endpoints


@router.get("/groups")
def list_groups(backend: BackendDependency) -> list[Group]:
    """List all groups."""
    return backend.list_groups()


@router.post("/groups", status_code=status.HTTP_201_CREATED)
def create_group(backend: BackendDependency, group: Group) -> ResourceOperationStatus:
    """Create a new group."""
    try:
        backend.create_group(group)
        return ResourceOperationStatus(
            success=True,
            target=f"Group {group.name}",
            operation=CRUDOperation.CREATE,
        )
    except MatchboxGroupAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/groups/{group_name}")
def get_group(backend: BackendDependency, group_name: GroupName) -> Group:
    """Get a group."""
    return backend.get_group(group_name)


@router.delete("/groups/{group_name}")
def delete_group(
    backend: BackendDependency,
    group_name: GroupName,
    certain: Annotated[bool, Query()] = False,
) -> ResourceOperationStatus:
    """Delete a group."""
    try:
        backend.delete_group(group_name, certain=certain)
        return ResourceOperationStatus(
            success=True, target=f"Group {group_name}", operation=CRUDOperation.DELETE
        )
    except (MatchboxSystemGroupError, MatchboxDeletionNotConfirmed) as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


# Membership management endpoints


@router.post("/groups/{group_name}/members", status_code=status.HTTP_201_CREATED)
def add_member(
    backend: BackendDependency,
    group_name: GroupName,
    member: User,
) -> ResourceOperationStatus:
    """Add a member to a group."""
    backend.add_user_to_group(member.user_name, group_name)
    return ResourceOperationStatus(
        success=True,
        target=f"User {member.user_name} in {group_name}",
        operation=CRUDOperation.CREATE,
    )


@router.delete("/groups/{group_name}/members/{user_name}")
def remove_member(
    backend: BackendDependency,
    group_name: GroupName,
    user_name: str,
) -> ResourceOperationStatus:
    """Remove a member from a group."""
    backend.remove_user_from_group(user_name, group_name)
    return ResourceOperationStatus(
        success=True,
        target=f"User {user_name} in {group_name}",
        operation=CRUDOperation.DELETE,
    )


# System permission management endpoints


@router.get("/system/permissions")
def get_permissions(
    backend: BackendDependency,
) -> list[PermissionGrant]:
    """Get permissions for the system resource."""
    return backend.get_permissions(BackendResourceType.SYSTEM)


@router.post("/system/permissions")
def grant_permission(
    backend: BackendDependency,
    grant: PermissionGrant,
) -> ResourceOperationStatus:
    """Grant a permission on the system resource."""
    backend.grant_permission(
        grant.group_name, grant.permission, BackendResourceType.SYSTEM
    )
    return ResourceOperationStatus(
        success=True,
        target=f"{grant.permission} on system for {grant.group_name}",
        operation=CRUDOperation.CREATE,
    )


@router.delete("/system/permissions/{permission}/{group_name}/")
def revoke_permission(
    backend: BackendDependency,
    permission: PermissionType,
    group_name: GroupName,
) -> ResourceOperationStatus:
    """Revoke a permission on the system resource."""
    backend.revoke_permission(group_name, permission, BackendResourceType.SYSTEM)
    return ResourceOperationStatus(
        success=True,
        target=f"{permission} on system for {group_name}",
        operation=CRUDOperation.DELETE,
    )
