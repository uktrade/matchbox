#!/usr/bin/env python3
"""Script to run Textual evaluation app with scenario data via CLI."""

import logging
import tempfile
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import create_engine

from matchbox.client._settings import settings
from matchbox.client.cli.eval import run
from matchbox.common.factories import models as _  # noqa:F401
from matchbox.common.factories.scenarios import SCENARIO_REGISTRY, setup_scenario
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)

# Set up logger for this script
logger = logging.getLogger(__name__)


def _get_backend() -> MatchboxDBAdapter:
    """Instantiates a backend class based on the contents of .env."""
    SettingsClass = get_backend_settings(MatchboxServerSettings().backend_type)
    settings = SettingsClass()
    return settings_to_backend(settings)


@contextmanager
def scenario_setup(scenario_name: str) -> Generator[dict[str, object], None, None]:
    """Context manager that sets up scenario data and yields CLI parameters."""

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        warehouse_engine = create_engine(f"sqlite:///{tmp_path}")

        # Configure client to use our SQLite warehouse
        settings.default_warehouse = str(warehouse_engine.url)

        backend = _get_backend()
        backend.clear(certain=True)

        logger.info(f"Setting up {scenario_name} scenario...")

        with setup_scenario(
            backend=backend,
            scenario_type=scenario_name,
            warehouse=warehouse_engine,
            n_entities=10,
            seed=42,
        ) as dag:
            logger.info("Scenario ready! Starting Textual evaluate app via CLI...")

            # Select the appropriate resolution for each scenario
            match scenario_name:
                case "bare" | "index" | "convergent":
                    raise RuntimeError("Scenario has nothing to evaluate.")
                case "dedupe":
                    resolution = dag.dag.nodes["naive_test_crn"]
                case "probabilistic_dedupe":
                    resolution = dag.dag.nodes["probabilistic_test_crn"]
                case "link":
                    resolution = dag.dag.nodes["final_join"]
                case "alt_dedupe":
                    resolution = dag.dag.nodes["dedupe_foo_a"]
                case _:
                    raise ValueError(f"Unknown scenario: {scenario_name}")

            # Yield CLI parameters instead of app instance
            yield {
                "collection": resolution.resolution_path.collection,
                "resolution": resolution.resolution_path.name,
                "pending": True,
                "warehouse": str(warehouse_engine.url),
            }

    finally:
        # Clean up
        with suppress(Exception):
            backend.clear(certain=True)
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)


def main(
    scenario: Annotated[
        str,
        typer.Argument(
            help=f"Scenario type. Available: {', '.join(SCENARIO_REGISTRY.keys())}"
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
    """Run the scenario-based evaluation app via CLI."""
    # Set up basic logging for this script
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # If no scenario provided, list available scenarios
    if scenario is None or scenario == "":
        available_scenarios = list(SCENARIO_REGISTRY.keys())
        logger.info("Available scenarios:")
        for i, name in enumerate(available_scenarios, 1):
            logger.info(f"  {i}. {name}")
        logger.info(
            "\nUsage: uv run python test/scripts/eval.py <scenario> [--log file.log]"
        )
        raise typer.Exit(0)

    if scenario not in SCENARIO_REGISTRY:
        available = ", ".join(SCENARIO_REGISTRY.keys())
        logger.error(f"Unknown scenario: {scenario}")
        logger.info(f"Available scenarios: {available}")
        raise typer.Exit(1)

    try:
        with scenario_setup(scenario) as cli_params:
            # Call CLI evaluation command directly
            try:
                run.evaluate(
                    collection=cli_params["collection"],
                    resolution=cli_params["resolution"],
                    pending=cli_params["pending"],
                    warehouse=cli_params["warehouse"],
                    user=None,
                    log_file=log_file if log_file and log_file.strip() else None,
                )
            except KeyboardInterrupt as e:
                logger.info("\nKeyboard interrupt received, stopping...")
                raise typer.Exit(0) from e

    except KeyboardInterrupt as e:
        logger.info("\nExiting...")
        raise typer.Exit(0) from e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    typer.run(main)
