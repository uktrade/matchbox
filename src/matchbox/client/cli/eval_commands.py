"""CLI commands for entity evaluation."""

import typer
from typing_extensions import Annotated

from matchbox.client.cli.eval.ui import EntityResolutionApp
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName

eval_app = typer.Typer(help="Entity evaluation commands")


@eval_app.command()
def start(
    resolution: Annotated[
        str, typer.Option("--resolution", "-r", help="Model resolution to sample from")
    ] = DEFAULT_RESOLUTION,
    samples: Annotated[
        int,
        typer.Option("--samples", "-n", help="Number of entity clusters to evaluate"),
    ] = 100,
    user: Annotated[
        str | None,
        typer.Option(
            "--user", "-u", help="Username for authentication (overrides settings)"
        ),
    ] = None,
    warehouse: Annotated[
        str | None,
        typer.Option(
            "--warehouse", "-w", help="Warehouse database URL (overrides settings)"
        ),
    ] = None,
) -> None:
    """Start the interactive entity resolution evaluation tool."""
    app = EntityResolutionApp(
        resolution=ModelResolutionName(resolution),
        num_samples=samples,
        user=user,
        warehouse=warehouse,
    )
    app.run()
