"""Group management commands for Matchbox CLI."""

from typing import Annotated

import typer
from rich import print
from rich.table import Table

from matchbox.client import _handler
from matchbox.client.cli.annotations import DeletionOpt, GroupOpt, UsernameOpt
from matchbox.common.dtos import Group, ResourceOperationStatus

app = typer.Typer(help="Group management")


@app.callback(invoke_without_command=True)
def list_groups(ctx: typer.Context) -> None:
    """List all groups."""
    if ctx.invoked_subcommand is not None:
        return

    groups: list[Group] = _handler.list_groups()

    table = Table(title="Groups")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("System", justify="center")

    for group in groups:
        table.add_row(
            group.name,
            group.description or "",
            "✓" if group.is_system else "",
        )

    print(table)


@app.command("create")
def create_group(
    name: GroupOpt,
    description: Annotated[
        str | None,
        typer.Option("--description", "-d", help="Group description"),
    ] = None,
) -> None:
    """Create a new group."""
    result: ResourceOperationStatus = _handler.create_group(
        name=name, description=description
    )
    if result.success:
        print(f"✓ Created group {name}")
    else:
        print(f"✗ Failed to create group {name}")
        if result.details:
            print(f"  {result.details}")
        raise typer.Exit(code=1)


@app.command("show")
def show_group(
    name: GroupOpt,
) -> None:
    """Show group details and members."""
    group: Group = _handler.get_group(name)

    print(f"[bold cyan]{group.name}[/bold cyan]")
    if group.description:
        print(f"  {group.description}")
    if group.is_system:
        print("  [dim](system group)[/dim]")

    print("\n[bold]Members:[/bold]")
    if group.members:
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Username", style="cyan")
        table.add_column("Email", style="dim")

        for member in group.members:
            table.add_row(
                member.user_name,
                member.email or "[dim]—[/dim]",
            )
        print(table)
    else:
        print("  [dim]No members[/dim]")


@app.command("delete")
def delete_group(
    name: GroupOpt,
    certain: DeletionOpt,
) -> None:
    """Delete a group."""
    result: ResourceOperationStatus = _handler.delete_group(name=name, certain=certain)
    if result.success:
        print(f"✓ Deleted group {name}")
    else:
        print(f"✗ Failed to delete group {name}")
        if result.details:
            print(f"  {result.details}")
        raise typer.Exit(code=1)


@app.command("remove")
def remove_member(
    group: GroupOpt,
    user: UsernameOpt,
) -> None:
    """Remove a user from a group."""
    result: ResourceOperationStatus = _handler.remove_user_from_group(
        group_name=group, user_name=user
    )
    if result.success:
        print(f"✓ Removed {user} from {group}")
    else:
        print(f"✗ Failed to remove {user} from {group}")
        if result.details:
            print(f"  {result.details}")
        raise typer.Exit(code=1)


@app.command("add")
def add_member(
    group: GroupOpt,
    user: UsernameOpt,
) -> None:
    """Add a user to a group."""
    result: ResourceOperationStatus = _handler.add_user_to_group(
        group_name=group, user_name=user
    )
    if result.success:
        print(f"✓ Added {user} to {group}")
    else:
        print(f"✗ Failed to add {user} to {group}")
        if result.details:
            print(f"  {result.details}")
        raise typer.Exit(code=1)
