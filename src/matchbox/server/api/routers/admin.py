"""Admin routes for user and group management."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    ErrorResponse,
    Group,
    GroupName,
    PermissionGrant,
    PermissionType,
    ResourceOperationStatus,
    User,
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


@router.post(
    "/groups",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
    status_code=status.HTTP_201_CREATED,
)
def create_group(backend: BackendDependency, group: Group) -> ResourceOperationStatus:
    """Create a new group."""
    backend.create_group(group)
    return ResourceOperationStatus(
        success=True,
        target=f"Group {group.name}",
        operation=CRUDOperation.CREATE,
    )


@router.get("/groups/{group_name}")
def get_group(backend: BackendDependency, group_name: GroupName) -> Group:
    """Get a group."""
    return backend.get_group(group_name)


@router.delete(
    "/groups/{group_name}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
def delete_group(
    backend: BackendDependency,
    group_name: GroupName,
    certain: Annotated[bool, Query()] = False,
) -> ResourceOperationStatus:
    """Delete a group."""
    backend.delete_group(group_name, certain=certain)
    return ResourceOperationStatus(
        success=True, target=f"Group {group_name}", operation=CRUDOperation.DELETE
    )


# Membership management endpoints


@router.post(
    "/groups/{group_name}/members",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
    status_code=status.HTTP_201_CREATED,
)
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


@router.delete(
    "/groups/{group_name}/members/{user_name}",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
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


@router.get(
    "/system/permissions",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
def get_permissions(
    backend: BackendDependency,
) -> list[PermissionGrant]:
    """Get permissions for the system resource."""
    return backend.get_permissions(BackendResourceType.SYSTEM)


@router.post(
    "/system/permissions",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
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


@router.delete(
    "/system/permissions/{permission}/{group_name}/",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
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
