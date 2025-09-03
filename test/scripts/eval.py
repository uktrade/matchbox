#!/usr/bin/env python3
"""Script to run Textual eval app with scenario data via CLI."""

import logging
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine

from matchbox.client._settings import settings
from matchbox.common.factories.scenarios import SCENARIO_REGISTRY, setup_scenario
from matchbox.common.graph import DEFAULT_RESOLUTION
from matchbox.server.base import MatchboxDatastoreSettings
from matchbox.server.postgresql import MatchboxPostgres

# Set up logger for this script
logger = logging.getLogger(__name__)


class DevelopmentSettings(BaseSettings):
    """Duplicate of the same settings in pytest fixtures.

    Can't import from fixtures, and moving to core Matchbox makes no sense.
    """

    api_port: int = 8000
    datastore_console_port: int = 9003
    datastore_port: int = 9002
    warehouse_port: int = 7654
    postgres_backend_port: int = 9876

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="MB__DEV__",
        env_nested_delimiter="__",
        env_file=Path("environments/development.env"),
        env_file_encoding="utf-8",
    )


@contextmanager
def scenario_setup(scenario_name: str):
    """Context manager that sets up scenario data and yields CLI parameters."""

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    # Store original client settings
    original_warehouse = getattr(settings, "default_warehouse", None)

    try:
        warehouse_engine = create_engine(f"sqlite:///{tmp_path}")

        # Configure client to use our SQLite warehouse
        settings.default_warehouse = str(warehouse_engine.url)

        dev_settings = DevelopmentSettings()

        datastore_settings = MatchboxDatastoreSettings(
            host="localhost",
            port=dev_settings.datastore_port,
            access_key_id="access_key_id",
            secret_access_key="secret_access_key",
            default_region="eu-west-2",
            cache_bucket_name="cache",
        )

        from matchbox.server.postgresql import MatchboxPostgresSettings

        postgres_settings = MatchboxPostgresSettings(
            batch_size=250_000,
            postgres={
                "host": "localhost",
                "port": dev_settings.postgres_backend_port,
                "user": "matchbox_user",
                "password": "matchbox_password",
                "database": "matchbox",
                "db_schema": "mb",
                "alembic_config": "src/matchbox/server/postgresql/alembic.ini",
            },
            datastore=datastore_settings,
        )

        backend = MatchboxPostgres(settings=postgres_settings)
        backend.clear(certain=True)

        logger.info(f"Setting up {scenario_name} scenario...")

        with setup_scenario(
            backend=backend,
            scenario_type=scenario_name,
            warehouse=warehouse_engine,
            n_entities=10,
            seed=42,
        ) as dag:
            logger.info("Scenario ready! Starting Textual eval app via CLI...")

            # Determine resolution for scenario
            if scenario_name in ["bare", "index"]:
                # These scenarios don't have models, use source resolution
                resolution = (
                    list(dag.sources.keys())[0] if dag.sources else DEFAULT_RESOLUTION
                )
            else:
                # Use first model resolution
                resolution = (
                    list(dag.models.keys())[0] if dag.models else DEFAULT_RESOLUTION
                )

            # Yield CLI parameters instead of app instance
            yield {
                "resolution": resolution,
                "warehouse": str(warehouse_engine.url),
                "samples": 20,
            }

    finally:
        # Restore original client settings
        if original_warehouse is not None:
            settings.default_warehouse = original_warehouse
        elif hasattr(settings, "default_warehouse"):
            delattr(settings, "default_warehouse")

        # Clean up
        try:
            backend.clear(certain=True)
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


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
):
    """Run the scenario-based eval app via CLI."""
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
            # Build CLI command
            cmd = [
                sys.executable,
                "-m",
                "matchbox.client.cli.main",
                "eval",
                "start",
                "--resolution",
                cli_params["resolution"],
                "--warehouse",
                cli_params["warehouse"],
                "--samples",
                str(cli_params["samples"]),
            ]

            # Add log file if specified and non-empty
            if log_file and log_file.strip():
                cmd.extend(["--log", log_file])

            # Run the CLI command
            try:
                result = subprocess.run(cmd, check=False)
                raise typer.Exit(result.returncode)
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
