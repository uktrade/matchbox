"""Admin commands for Matchbox CLI."""

from typing import Annotated

import typer
from rich import print
from rich.table import Table

from matchbox.client import _handler
from matchbox.client.cli.annotations import DeletionOpt, GroupOpt, UsernameOpt

app = typer.Typer(help="Admin commands (requires admin privileges)")

groups_app = typer.Typer(help="Manage groups")
system_app = typer.Typer(help="Manage system permissions")

app.add_typer(groups_app, name="groups")
app.add_typer(system_app, name="system")


# Groups


@groups_app.command("list")
def list_groups() -> None:
    """List all groups."""
    groups = _handler.list_groups()

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


@groups_app.command("create")
def create_group(
    name: GroupOpt,
    description: Annotated[
        str | None,
        typer.Option("--description", "-d", help="Group description"),
    ] = None,
) -> None:
    """Create a new group."""
    result = _handler.create_group(name=name, description=description)
    print(f"✓ {result.target}")


@groups_app.command("delete")
def delete_group(
    name: GroupOpt,
    certain: DeletionOpt,
) -> None:
    """Delete a group."""
    result = _handler.delete_group(name=name, certain=certain)
    print(f"✓ {result.target}")


@groups_app.command("show")
def show_group(
    name: GroupOpt,
) -> None:
    """Show group details and members."""
    group = _handler.get_group(name)

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


@groups_app.command("add")
def add_member(
    group: GroupOpt,
    user: UsernameOpt,
) -> None:
    """Add a user to a group."""
    _handler.add_user_to_group(group_name=group, user_name=user)
    print(f"✓ Added {user} to {group}")


@groups_app.command("remove")
def remove_member(
    group: GroupOpt,
    user: UsernameOpt,
) -> None:
    """Remove a user from a group."""
    _handler.remove_user_from_group(group_name=group, user_name=user)
    print(f"✓ Removed {user} from {group}")


# System permissions


@system_app.command("list")
def list_system_permissions() -> None:
    """List system permissions."""
    permissions = _handler.get_system_permissions()

    table = Table(title="System permissions")
    table.add_column("Group", style="cyan")
    table.add_column("Permission", style="green")

    for perm in permissions:
        table.add_row(perm.group_name, perm.permission)

    if not permissions:
        print("[dim]No system permissions granted[/dim]")
    else:
        print(table)
