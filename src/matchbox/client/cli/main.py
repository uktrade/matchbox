"""Main CLI entry point for Matchbox."""

import typer

from matchbox.client.cli.eval_commands import eval_app

app = typer.Typer(
    name="matchbox", help="Matchbox: Entity resolution and data linking framework"
)

app.add_typer(
    eval_app, name="eval", help="Entity evaluation and manual labelling tools"
)

if __name__ == "__main__":
    app()
