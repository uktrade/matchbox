"""Collection commands for Matchbox CLI."""

from typing import Annotated

import typer
from rich import print
from rich.table import Table

from matchbox.client import _handler
from matchbox.client.cli.annotations import CollectionOpt, GroupOpt, PermissionOpt
from matchbox.common.dtos import DefaultGroup

app = typer.Typer(help="Manage collections")
permissions_app = typer.Typer(help="Manage collection permissions")

app.add_typer(permissions_app, name="permissions")


@app.command("list")
def list_collections() -> None:
    """List all collections."""
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
    _ = _handler.create_collection(name, admin_group=admin_group)
    print(f"✓ Created collection '{name}'")
    if admin_group != DefaultGroup.PUBLIC:
        print(f"  Admin permission granted to group '{admin_group}'")


# Collection permissions


@permissions_app.command("list")
def list_permissions(collection: CollectionOpt) -> None:
    """List permissions for a collection."""
    permissions = _handler.get_collection_permissions(collection)

    table = Table(title=f"Permissions for '{collection}'")
    table.add_column("Group", style="cyan")
    table.add_column("Permission", style="green")

    for perm in permissions:
        table.add_row(perm.group_name, perm.permission)

    if not permissions:
        print("[dim]No permissions granted[/dim]")
    else:
        print(table)


@permissions_app.command("grant")
def grant_permission(
    collection: CollectionOpt,
    group: GroupOpt,
    permission: PermissionOpt,
) -> None:
    """Grant a permission to a group on a collection."""
    _handler.grant_collection_permission(
        collection=collection,
        group_name=group,
        permission=permission,
    )
    print(f"✓ Granted {permission} on '{collection}' to '{group}'")


@permissions_app.command("revoke")
def revoke_permission(
    collection: CollectionOpt,
    group: GroupOpt,
    permission: PermissionOpt,
) -> None:
    """Revoke a permission from a group on a collection."""
    _handler.revoke_collection_permission(
        collection=collection,
        group_name=group,
        permission=permission,
    )
    print(f"✓ Revoked {permission} on '{collection}' from '{group}'")
