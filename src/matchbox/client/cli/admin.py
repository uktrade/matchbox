"""Admin commands for Matchbox CLI."""

import typer
from rich import print

from matchbox.client import _handler

app = typer.Typer(help="System administration and maintenance")

# Maintainance


@app.command("prune")
def delete_orphans() -> None:
    """Deletes orphans from Matchbox database.

    Orphan clusters are clusters that are not linked to any other table, because they
    have become isolated as a result of the change or removal of resolutions.
    This command will remove them from the database.
    """
    response = _handler.delete_orphans()
    print(response)
