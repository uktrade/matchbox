"""Logging utilities."""

import importlib.metadata
import logging
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

_PLUGINS = None

LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
"""Type for all Python log levels."""


class PrefixedLoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that supports adding a prefix enclosed in square brackets.

    This adapter allows passing an optional prefix parameter to any logging call
    without modifying the underlying logger.
    """

    def process(
        self,
        msg: Any,  # noqa: ANN401
        kwargs: dict[str, Any],  # noqa: ANN401
    ) -> tuple[Any, dict[str, Any]]:  # noqa: ANN401
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


def get_formatter() -> logging.Formatter:
    """Retrieve plugin registered in 'matchbox.logging' entry point, or fallback."""
    global _PLUGINS
    if _PLUGINS is None:
        _PLUGINS = []
        for ep in importlib.metadata.entry_points(group="matchbox.logging"):
            try:
                _PLUGINS.append(ep.load()())
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to load logging plugin: {e}")

    if _PLUGINS:
        return _PLUGINS[0]

    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def configure_celery_logging(level: int = logging.INFO) -> None:
    """Configure root and celery loggers to use matchbox formatting.

    This should also get logging from matchbox code running in forked processes.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(get_formatter())

    # Clear handlers and add our formatted handler to root
    root.handlers.clear()
    root.addHandler(handler)

    # Propagate celery loggers to root so they use our handler/formatter
    for name in ("celery", "celery.app.trace", "celery.worker", "celery.task"):
        log = logging.getLogger(name)
        log.handlers.clear()
        log.propagate = True
        log.setLevel(level)

    # Propagate matchbox loggers to root too
    mb = logging.getLogger("matchbox")
    mb.handlers.clear()
    mb.propagate = True
    mb.setLevel(level)
