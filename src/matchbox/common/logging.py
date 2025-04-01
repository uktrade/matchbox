"""Logging utilities."""

import logging

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

"""Logger for Matchbox.

Used for all logging in the Matchbox library.
"""
logger = logging.getLogger("matchbox")
logger.addHandler(logging.NullHandler())

"""Console for Matchbox.

Used for any CLI utilities in the Matchbox library.
"""
console = Console()


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
