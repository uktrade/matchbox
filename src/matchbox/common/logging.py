"""Logging utilities."""

import datetime
import json
import logging
import os
from functools import cached_property
from importlib.metadata import version
from typing import Any, Final, Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
"""Type for all Python log levels."""


class PrefixedLoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that supports adding a prefix enclosed in square brackets.

    This adapter allows passing an optional prefix parameter to any logging call
    without modifying the underlying logger.
    """

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Process the log message, adding a prefix if provided.

        Args:
            msg: The log message
            kwargs: Additional arguments to the logging method

        Returns:
            Tuple of (modified_message, modified_kwargs)
        """
        prefix = kwargs.pop("prefix", None)

        if prefix:
            msg = f"[{prefix}] {msg}"

        return msg, kwargs


logger: Final[PrefixedLoggerAdapter]
"""Logger for Matchbox.

Used for all logging in the Matchbox library.

Allows passing a prefix to any logging call.

Examples:
    ```python
    log_prefix = f"Model {name}"
    logger.debug("Inserting metadata", prefix=log_prefix)
    logger.debug("Inserting data", prefix=log_prefix)
    logger.info("Insert successful", prefix=log_prefix)
    ```
"""
logger = PrefixedLoggerAdapter(logging.getLogger("matchbox"), {})


console: Final[Console] = Console()
"""Console for Matchbox.

Used for any CLI utilities in the Matchbox library.
"""


def build_progress_bar(console_: Console | None = None) -> Progress:
    """Create a progress bar."""
    if console_ is None:
        console_ = console

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console_,
    )


class ASIMFormatter(logging.Formatter):
    """Format logging with ASIM standard fields."""

    @cached_property
    def matchbox_version(self) -> str:
        """Cached matchbox version."""
        return version("matchbox_db")

    @cached_property
    def event_severity(self) -> dict[str, str]:
        """Event severity level lookup."""
        return {
            "DEBUG": "Informational",
            "INFO": "Informational",
            "WARNING": "Low",
            "ERROR": "Medium",
            "CRITICAL": "High",
        }

    def format(self, record) -> str:
        """Convert logs to JSON."""
        log_time = datetime.datetime.utcfromtimestamp(record.created).isoformat()
        return json.dumps(
            {
                "EventMessage": record.getMessage(),
                "EventCount": 1,
                "EventStartTime": log_time,
                "EventEndTime": log_time,
                "EventType": record.name,
                "EventResult": "NA",
                "EventSeverity": self.event_severity[record.levelname],
                "EventOriginalSeverity": record.levelname,
                "EventSchema": "ProcessEvent",
                "EventSchemaVersion": "0.1.4",
                "EventVendor": "AWS",
                "EventProduct": "VPC",
                "AdditionalFields": {
                    "dd.version": self.matchbox_version,
                    "dd.env": os.getenv("DD_ENV"),
                    "dd.service": os.getenv("DD_SERVICE"),
                    "dd.team": "matchbox",
                    "dd.application": "matchbox",
                },
            },
        )
