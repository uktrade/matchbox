"""Main CLI entry point for Matchbox."""

import typer
from rich import print

from matchbox import __version__
from matchbox.client.cli import auth, server
from matchbox.client.cli.eval.run import evaluate

app = typer.Typer(
    name="matchbox", help="Matchbox: Entity resolution and data linking framework"
)


@app.command()
def version() -> None:
    """Shows the local Matchbox version."""
    print(f"Matchbox version: {__version__}")


# Add subcommands
app.add_typer(server.app, name="server")
app.add_typer(auth.app, name="auth")
app.command(name="eval")(evaluate)

if __name__ == "__main__":
    app()
