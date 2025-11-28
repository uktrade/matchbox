"""Logging utilities."""

import functools
import importlib.metadata
import logging
import time
from collections.abc import Callable
from typing import Any, Final, Literal, ParamSpec, TypeVar

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


T = TypeVar("T")
P = ParamSpec("P")


def profile(
    logger: PrefixedLoggerAdapter = logger,
    level: int = logging.INFO,
    prefix: str | None = "Profiling",
    kwarg: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to profile functions and methods using logger.

    Args:
        logger: The logger to use.
        level: The level to use to log the profiling information. It defaults to INFO.
        prefix: Prefix to pass to the logged message.
        kwarg: Argument name to extract from function call.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # If class, first argument will be self
            self = args[0] if args else None
            # If class has attribute "name", it's the node name
            node = self.name if self and hasattr(self, "name") else None
            # If kwarg, will try to get value passed to function
            if kwarg is not None:
                value = kwargs.get(kwarg)

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start

                if node:
                    msg = f"`{func.__name__}` in node `{node}` took {duration:.3f}s"
                elif kwarg:
                    msg = (
                        f"`{func.__name__}` with {kwarg} `{value}` took {duration:.3f}s"
                    )
                else:
                    msg = f"`{func.__name__}` took {duration:.3f}s"

                logger.log(level, msg, prefix=prefix)

        return wrapper

    return decorator
