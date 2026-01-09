"""CLI commands for entity evaluation."""

import logging
from typing import Annotated

import typer
from sqlalchemy import create_engine

from matchbox.client.cli.eval.app import EntityResolutionApp
from matchbox.client.dags import DAG


def evaluate(
    collection: Annotated[
        str, typer.Option("--collection", "-c", help="Collection name (required)")
    ],
    resolution: Annotated[
        str | None,
        typer.Option(
            "--resolution",
            "-r",
            help="Resolution name (defaults to collection's final_step)",
        ),
    ] = None,
    pending: Annotated[
        bool,
        typer.Option(
            "--pending",
            "-p",
            help="Whether to evaluate the pending DAG, instead of the default",
        ),
    ] = False,
    user: Annotated[
        str | None,
        typer.Option(
            "--user", "-u", help="Username for authentication (overrides settings)"
        ),
    ] = None,
    warehouse: Annotated[
        str | None,
        typer.Option(
            "--warehouse",
            "-w",
            help="Warehouse database connection string (e.g. postgresql://user:pass@host/db)",
        ),
    ] = None,
    log_file: Annotated[
        str | None,
        typer.Option(
            "--log",
            help="Log file path to redirect all logging output (keeps UI clean)",
        ),
    ] = None,
    sample_file: Annotated[
        str | None,
        typer.Option(
            "--file",
            "-f",
            help="Pre-compiled sample file. If set, ignores resolutions parameters.",
        ),
    ] = None,
    session_tag: Annotated[
        str | None,
        typer.Option(
            "--tag",
            "-t",
            help="String to use to tag judgements sent to the server.",
        ),
    ] = None,
) -> None:
    """Start the interactive entity resolution evaluation tool.

    Requires a warehouse connection to fetch source data for evaluation clusters.

    Example:
        matchbox eval --collection companies --warehouse postgresql://user:pass@localhost/warehouse
    """
    # Set up logging redirect if --log specified
    if log_file:
        _setup_logging_redirect(log_file)

    # Create warehouse engine and location (required for loading DAG)
    if not warehouse:
        raise typer.BadParameter(
            "Warehouse connection string is required. "
            "Provide via --warehouse or -w flag."
        )

    # Create engine from connection string (with password)
    warehouse_engine = create_engine(warehouse)

    # Load DAG from server
    if pending:
        dag: DAG = DAG(name=collection).load_pending()
    else:
        dag: DAG = DAG(name=collection).load_default()

    # Attach warehouse to all objects
    dag: DAG = dag.set_client(warehouse_engine)

    if sample_file:
        resolution = None
    else:
        # Get resolution name from --resolution or DAG's final_step
        model = dag.get_model(resolution) if resolution is not None else dag.final_step
        resolution = model.resolution_path

    try:
        # Create app with loaded DAG (not warehouse string)
        app = EntityResolutionApp(
            resolution=resolution,
            user=user,
            dag=dag,
            session_tag=session_tag,
            sample_file=sample_file,
            show_help=True,
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
