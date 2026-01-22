"""Main CLI entry point for Matchbox."""

import sys

import typer
from rich import print

from matchbox import __version__
from matchbox.client import _handler
from matchbox.client.cli import admin, auth, collections, groups
from matchbox.client.cli.eval.run import evaluate
from matchbox.common.dtos import OKMessage
from matchbox.common.exceptions import MatchboxHttpException

app = typer.Typer(
    name="matchbox", help="Matchbox: Entity resolution and data linking framework"
)


@app.command()
def version() -> None:
    """Shows the local Matchbox version."""
    print(f"Matchbox version: {__version__}")


@app.command()
def health() -> None:
    """Checks the health of the Matchbox server."""
    response: OKMessage = _handler.healthcheck()
    print(response.model_dump_json(indent=2))


# Add subcommands
app.add_typer(auth.app, name="auth")
app.add_typer(collections.app, name="collections")
app.add_typer(groups.app, name="groups")
app.add_typer(admin.app, name="admin")
app.command(name="eval")(evaluate)


def run() -> None:
    """Run the Typer app with global HTTP error handling."""
    try:
        app()
    except MatchboxHttpException as e:
        print(f"[bold red]Error:[/bold red] {e}", file=sys.stderr)
        details = e.to_details()
        if details:
            print(details, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
