"""Main CLI entry point for Matchbox."""

import typer
from rich import print

from matchbox.client import _handler
from matchbox.client.cli.eval.run import evaluate

app = typer.Typer(
    name="matchbox", help="Matchbox: Entity resolution and data linking framework"
)


@app.command()
def health() -> None:
    """Checks the health of the Matchbox server."""
    response = _handler.healthcheck()
    print(response)


app.command(name="eval")(evaluate)

if __name__ == "__main__":
    app()
