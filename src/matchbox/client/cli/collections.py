"""Collection commands for Matchbox CLI."""

from typing import Annotated

import typer
from rich import print
from rich.table import Table

from matchbox.client import _handler
from matchbox.client.cli.annotations import (
    CollectionOpt,
    DeletionOpt,
    GroupOpt,
    PermissionOpt,
)
from matchbox.common.dtos import DefaultGroup, PermissionGrant, ResourceOperationStatus

app = typer.Typer(help="Manage collections")


@app.callback(invoke_without_command=True)
def list_collections(ctx: typer.Context) -> None:
    """List all collections."""
    if ctx.invoked_subcommand is not None:
        return

    collections = _handler.list_collections()
    for collection in collections:
        print(collection)


@app.command("create")
def create_collection(
    name: CollectionOpt,
    admin_group: Annotated[
        str,
        typer.Option(
            "--group",
            "-g",
            help="Group that will administer this collection",
        ),
    ] = DefaultGroup.PUBLIC,
) -> None:
    """Create a new collection."""
    response: ResourceOperationStatus = _handler.create_collection(
        name, admin_group=admin_group
    )
    if response.success:
        print(f"✓ Created collection {name}")
        if admin_group != DefaultGroup.PUBLIC:
            print(f"  Admin permission granted to group {admin_group}")
    else:
        print(f"✗ Failed to create collection {name}")
        if response.details:
            print(f"  {response.details}")
        raise typer.Exit(code=1)


@app.command("delete")
def delete_collection(
    name: CollectionOpt,
    certain: DeletionOpt,
) -> None:
    """Delete a collection."""
    response: ResourceOperationStatus = _handler.delete_collection(
        name, certain=certain
    )
    if response.success:
        print(f"✓ Deleted collection {name}")
    else:
        print(f"✗ Failed to delete collection {name}")
        if response.details:
            print(f"  {response.details}")
        raise typer.Exit(code=1)


# Collection permissions


@app.command("permissions")
def list_permissions(collection: CollectionOpt) -> None:
    """List permissions for a collection."""
    permissions: list[PermissionGrant] = _handler.get_collection_permissions(collection)

    table = Table(title=f"Permissions for {collection}")
    table.add_column("Group", style="cyan")
    table.add_column("Permission", style="green")

    for perm in permissions:
        table.add_row(perm.group_name, perm.permission)

    if not permissions:
        print("[dim]No permissions granted[/dim]")
    else:
        print(table)


@app.command("grant")
def grant_permission(
    collection: CollectionOpt,
    group: GroupOpt,
    permission: PermissionOpt,
) -> None:
    """Grant a permission to a group on a collection."""
    response: ResourceOperationStatus = _handler.grant_collection_permission(
        collection=collection,
        group_name=group,
        permission=permission,
    )
    if response.success:
        print(f"✓ Granted {permission} on {collection} to {group}")
    else:
        print(f"✗ Failed to grant {permission} on {collection} to {group}")
        if response.details:
            print(f"  {response.details}")
        raise typer.Exit(code=1)


@app.command("revoke")
def revoke_permission(
    collection: CollectionOpt,
    group: GroupOpt,
    permission: PermissionOpt,
) -> None:
    """Revoke a permission from a group on a collection."""
    response: ResourceOperationStatus = _handler.revoke_collection_permission(
        collection=collection,
        group_name=group,
        permission=permission,
    )
    if response.success:
        print(f"✓ Revoked {permission} on {collection} from {group}")
    else:
        print(f"✗ Failed to revoke {permission} on {collection} from {group}")
        if response.details:
            print(f"  {response.details}")
        raise typer.Exit(code=1)
