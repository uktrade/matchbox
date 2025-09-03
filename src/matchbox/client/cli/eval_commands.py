"""CLI commands for entity evaluation."""

import logging
from typing import Annotated

import typer

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
    log_file: Annotated[
        str | None,
        typer.Option(
            "--log",
            help="Log file path to redirect all logging output (keeps UI clean)",
        ),
    ] = None,
) -> None:
    """Start the interactive entity resolution evaluation tool."""
    try:
        # Set up logging redirect if --log specified
        if log_file:
            _setup_logging_redirect(log_file)

        app = EntityResolutionApp(
            resolution=ModelResolutionName(resolution),
            num_samples=samples,
            user=user,
            warehouse=warehouse,
        )
        app.run()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully - exit with success code
        pass


def _setup_logging_redirect(log_file_path: str) -> None:
    """Redirect all logging output to file to keep Textual UI clean."""
    # Get the root logger to capture ALL logging (including third-party)
    root_logger = logging.getLogger()

    # Remove any existing handlers (console handlers that print to screen)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up file handler for all logs
    file_handler = logging.FileHandler(log_file_path, mode="w")

    # Use detailed format since these are going to file, not screen
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    # Add file handler to root logger
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)  # Capture INFO and above

    # Also suppress stdout/stderr from other libraries that might bypass logging
    # This keeps the Textual UI completely clean
