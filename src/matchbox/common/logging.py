"""Logging utilities."""

import functools
import importlib.metadata
import logging
import os
import time
from collections.abc import Callable
from typing import Any, Final, Literal, ParamSpec, TypeVar

import psutil
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


def profile_time(
    logger: PrefixedLoggerAdapter = logger,
    level: int = logging.INFO,
    prefix: str | None = "Profiling",
    attr: str | None = None,
    kwarg: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to profile running time of functions and methods using logger.

    Args:
        logger: The logger to use.
        level: The level to use to log the profiling information. It defaults to INFO.
        prefix: Prefix to pass to the logged message.
        attr: Attribute name to extract from instantiated class.
        kwarg: Argument name to extract from function call.

    `attr` should be used when we want to include some class atribute in the log, e.g.
    node name in Source.
    `kwarg` should be used when we want to include the value passed to some function
    argument, e.g. path in `set_data`. This will only work if the argument is passed
    to the function as a kwarg, and not a positional argument.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # If attr, will try to get its value from class instance
            if attr is not None:
                # If class, first argument will be self
                self = args[0] if args else None
                node = getattr(self, attr) if self and hasattr(self, attr) else None
            # If kwarg, will try to get value passed to function
            if kwarg is not None:
                value = kwargs.get(kwarg)

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start

                if attr:
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


def log_mem_usage(
    logger: PrefixedLoggerAdapter = logger,
    level: int = logging.INFO,
    prefix: str | None = "Profiling",
) -> None:
    """Log current memory usage for this process."""
    proc = psutil.Process(os.getpid())
    usage = proc.memory_info().rss / (1024**2)
    msg = f"Current memory used by process (MiB): {usage}"
    logger.log(level, msg, prefix=prefix)
