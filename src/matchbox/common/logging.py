"""Logging utilities."""

import importlib.metadata
import json
import logging
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Final, Literal, Protocol

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class LoggingPlugin(Protocol):
    """Protocol for logging plugins that can be registered with Matchbox."""

    def get_trace_context(self) -> tuple[str | None, str | None]:
        """Get trace context information for logging."""
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Get additional fields for logging."""
        ...


_PLUGINS = None


def get_logging_plugins():
    """Retrieve logging plugins registered in the 'matchbox.logging' entry point."""
    global _PLUGINS
    if _PLUGINS is None:
        _PLUGINS = []
        for ep in importlib.metadata.entry_points(group="matchbox.logging"):
            try:
                _PLUGINS.append(ep.load()())
            except Exception:
                pass
    return _PLUGINS


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


logger: Final[PrefixedLoggerAdapter] = PrefixedLoggerAdapter(
    logging.getLogger("matchbox"), {}
)
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

    def __init__(self, fmt=None, datefmt=None, style="%", validate=True):
        """Initialize the ASIMFormatter including any logging plugins."""
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self.plugins = get_logging_plugins()

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

    def format(self, record):
        """Convert logs to JSON including basic ASIM fields."""
        log_time = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        log_entry = {
            "EventCount": 1,
            "EventStartTime": log_time,
            "EventEndTime": log_time,
            "EventType": record.name,
            "EventSeverity": self.event_severity[record.levelname],
            "EventOriginalSeverity": record.levelname,
            "message": record.getMessage(),
        }

        for plugin in self.plugins:
            try:
                trace_id, span_id = plugin.get_trace_context()
                if trace_id:
                    log_entry.update({"trace_id": trace_id, "span_id": span_id})
                log_entry.update(plugin.get_metadata())
            except Exception:
                pass

        return json.dumps(log_entry)
