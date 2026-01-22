"""Authentication management commands for Matchbox CLI."""

import typer
from rich import print

from matchbox.client import _handler
from matchbox.common.dtos import AuthStatusResponse, LoginResponse

app = typer.Typer(help="Manage Matchbox authentication")


@app.command()
def login() -> None:
    """Creates the user using the settings JWT."""
    response: LoginResponse = _handler.login()
    print(response.model_dump_json(indent=2))


@app.command()
def status() -> None:
    """Checks the authentication status."""
    response: AuthStatusResponse = _handler.auth_status()
    print(response.model_dump_json(indent=2))
