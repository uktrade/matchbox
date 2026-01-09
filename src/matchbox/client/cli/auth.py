"""Authentication management commands for Matchbox CLI."""

import typer
from rich import print

from matchbox.client import _handler

app = typer.Typer(help="Manage Matchbox authentication")


@app.command()
def status() -> None:
    """Checks the authentication status."""
    response = _handler.auth_status()
    print(response)
