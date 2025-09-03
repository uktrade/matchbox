"""Textual-based entity resolution evaluation tool."""

import contextlib
import logging
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import polars as pl
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Static,
)
from textual_plotext import PlotextPlot

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.plot import compute_pr_envelope, interpolate_pr_curve
from matchbox.client.cli.eval.utils import EvalData, EvaluationItem, get_samples
from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName

logger = logging.getLogger(__name__)


class EvaluationQueue:
    """Deque-based queue that maintains position illusion."""

    def __init__(self):
        """Initialise the queue."""
        self.items: deque[EvaluationItem] = deque()
        self._position_offset: int = 0  # Tracks "virtual position"

    @property
    def current(self) -> EvaluationItem | None:
        """Get the current item (always at index 0)."""
        return self.items[0] if self.items else None

    @property
    def current_position(self) -> int:
        """Current position for display (1-based)."""
        return self._position_offset + 1 if self.items else 0

    @property
    def total_count(self) -> int:
        """Total number of items in queue."""
        return len(self.items)

    @property
    def painted_items(self) -> list[EvaluationItem]:
        """Get all currently painted items in the queue."""
        return [item for item in self.items if item.is_painted]

    @property
    def painted_count(self) -> int:
        """Count of painted items ready for submission."""
        return len(self.painted_items)

    def move_next(self):
        """Rotate forward, increment position."""
        if len(self.items) > 1:
            self.items.append(self.items.popleft())
            self._position_offset = (self._position_offset + 1) % len(self.items)

    def move_previous(self):
        """Rotate backward, decrement position."""
        if len(self.items) > 1:
            self.items.appendleft(self.items.pop())
            self._position_offset = (self._position_offset - 1) % len(self.items)

    def submit_all_painted(self) -> list[EvaluationItem]:
        """Remove all painted items permanently."""
        painted = self.painted_items
        self.items = deque([item for item in self.items if not item.is_painted])
        # Reset position offset if we removed items
        if painted and self.items:
            self._position_offset = 0
        return painted

    def add_items(self, items: list[EvaluationItem]):
        """Add new items to the end of the queue."""
        self.items.extend(items)

    def clear(self):
        """Clear the entire queue."""
        self.items.clear()
        self._position_offset = 0


class EvaluationState:
    """Single source of truth for all application state."""

    def __init__(self):
        """Initialise evaluation state."""
        # Queue Management - replaces samples dict and entity_judgements
        self.queue: EvaluationQueue = EvaluationQueue()
        self.sample_limit: int = 100

        # UI State
        self.current_group_selection: str = ""  # Currently selected group letter
        self.compact_view_mode: bool = True  # Default to compact view
        self.show_plot: bool = False  # Plot display toggle

        # Plot State
        self.eval_data: EvalData | None = None  # EvalData object loaded at startup
        self.is_loading_eval_data: bool = False  # Loading state flag
        self.eval_data_error: str | None = None  # Error message if loading fails

        # Display State (derived from current queue item)
        self.display_leaf_ids: list[int] = []

        # User/Connection State
        self.user_name: str = ""
        self.user_id: int | None = None
        self.resolution: str = ""
        self.warehouse: str | None = None

        # Status/Feedback State
        self.status_message: str = ""
        self.status_color: str = "bright_white"
        self.is_submitting: bool = False
        self._status_timer_id: str | None = (
            None  # Track current timer to prevent conflicts
        )

        # Observer pattern for view updates
        self.listeners: list[Callable] = []

        # Number key mapping
        self.number_keys = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "0": 10,
        }

    @property
    def painted_count(self) -> int:
        """Get count of painted entities."""
        return self.queue.painted_count

    @property
    def current_cluster_id(self) -> int | None:
        """Get current cluster ID."""
        current = self.queue.current
        return current.cluster_id if current else None

    @property
    def current_df(self) -> pl.DataFrame | None:
        """Get current DataFrame."""
        current = self.queue.current
        return current.dataframe if current else None

    @property
    def current_assignments(self) -> dict[int, str]:
        """Get current entity's group assignments."""
        current = self.queue.current
        return current.assignments if current else {}

    def set_group_selection(self, group: str) -> None:
        """Set the current active group."""
        if group.isalpha() and len(group) == 1:
            self.current_group_selection = group.lower()
            self._notify_listeners()

    def clear_group_selection(self) -> None:
        """Clear the current group selection."""
        self.current_group_selection = ""
        self._notify_listeners()

    def toggle_view_mode(self) -> None:
        """Toggle between compact and detailed view modes."""
        self.compact_view_mode = not self.compact_view_mode
        self._notify_listeners()

    def toggle_plot(self) -> None:
        """Toggle plot display on/off."""
        self.show_plot = not self.show_plot
        self._notify_listeners()

    def clear_eval_data(self) -> None:
        """Clear eval data (e.g., when resolution changes)."""
        self.eval_data = None
        self.eval_data_error = None

    def set_eval_data_loading(self, loading: bool) -> None:
        """Set the loading state for eval data."""
        self.is_loading_eval_data = loading
        if loading:
            self.eval_data_error = None
        self._notify_listeners()

    def set_eval_data_error(self, error: str) -> None:
        """Set an error message for eval data loading."""
        self.eval_data_error = error
        self.is_loading_eval_data = False
        self._notify_listeners()

    def set_eval_data(self, eval_data: Any) -> None:
        """Set the eval data object."""
        self.eval_data = eval_data
        self.is_loading_eval_data = False
        self.eval_data_error = None
        self._notify_listeners()

    def assign_column_to_group(self, column_number: int, group: str) -> None:
        """Assign a display column to a group."""
        display_col_index = column_number - 1
        current = self.queue.current
        if current and 0 <= display_col_index < len(current.display_columns):
            current.assignments[display_col_index] = group
            self._notify_listeners()

    def clear_current_assignments(self) -> None:
        """Clear all group assignments for current entity."""
        current = self.queue.current
        if current:
            current.assignments.clear()
        self._notify_listeners()

    def set_display_data(self, display_leaf_ids: list[int]) -> None:
        """Set the display data."""
        self.display_leaf_ids = display_leaf_ids
        self._notify_listeners()

    def clear_display_data(self) -> None:
        """Clear all display data."""
        self.display_leaf_ids = []
        self._notify_listeners()

    def get_group_counts(self) -> dict[str, int]:
        """Get count of display columns in each group for current entity."""
        current = self.queue.current
        if not current:
            return {}

        assignments = self.current_assignments
        counts = {}

        # Count actual underlying leaf IDs, not just display columns
        for display_col_index, group in assignments.items():
            if display_col_index < len(current.duplicate_groups):
                duplicate_group_size = len(current.duplicate_groups[display_col_index])
                counts[group] = counts.get(group, 0) + duplicate_group_size

        # Add selected group with (0) if not already present
        if self.current_group_selection and self.current_group_selection not in counts:
            counts[self.current_group_selection] = 0

        # Include unassigned count if there are unassigned display columns
        assigned_display_cols = set(assignments.keys())
        unassigned_leaf_count = 0
        for display_col_index in range(len(current.duplicate_groups)):
            if display_col_index not in assigned_display_cols:
                duplicate_group = current.duplicate_groups[display_col_index]
                unassigned_leaf_count += len(duplicate_group)

        if unassigned_leaf_count > 0:
            counts["unassigned"] = unassigned_leaf_count

        return counts

    def get_judgement_groups(self) -> list[list[int]]:
        """Convert current assignments to judgement format."""
        current = self.queue.current
        if current and self.user_id:
            judgement = current.to_judgement(self.user_id)
            return judgement.endorsed
        return []

    def parse_number_key(self, key: str) -> int | None:
        """Convert number key to column number."""
        return self.number_keys.get(key)

    def has_current_assignments(self) -> bool:
        """Check if current entity has any group assignments."""
        return len(self.current_assignments) > 0

    def update_status(
        self,
        message: str,
        color: str = "bright_white",
        auto_clear_after: float | None = None,
    ) -> None:
        """Update status message with optional color and auto-clearing.

        Args:
            message: Status message to display
            color: Color for the message
            auto_clear_after: Seconds after which to auto-clear (None = no auto-clear)
        """
        # Cancel any existing status timer to prevent conflicts
        if self._status_timer_id is not None:
            self._cancel_status_timer()

        self.status_message = message
        self.status_color = color
        self._notify_listeners()

        # Set up auto-clear timer if requested
        if auto_clear_after is not None and auto_clear_after > 0:
            self._schedule_status_clear(auto_clear_after)

    def clear_status(self) -> None:
        """Clear status message and cancel any pending timers."""
        # Cancel any pending clear timer
        if self._status_timer_id is not None:
            self._cancel_status_timer()

        self.status_message = ""
        self.status_color = "bright_white"
        self._notify_listeners()

    def _schedule_status_clear(self, delay: float) -> None:
        """Schedule status clearing after a delay."""
        timer_id = str(uuid.uuid4())
        self._status_timer_id = timer_id

        # Store reference to app for timer scheduling - will be set by the main app
        if hasattr(self, "_app_ref") and self._app_ref is not None:
            self._app_ref.set_timer(
                delay, lambda: self._clear_status_if_current(timer_id)
            )

    def _cancel_status_timer(self) -> None:
        """Cancel current status timer."""
        self._status_timer_id = None

    def _clear_status_if_current(self, expected_timer_id: str) -> None:
        """Clear status only if this timer is still the current one."""
        if self._status_timer_id == expected_timer_id:
            self.clear_status()

    def add_listener(self, callback: Callable) -> None:
        """Add a callback to be notified when state changes."""
        self.listeners.append(callback)

    def _notify_listeners(self) -> None:
        """Notify all listeners of state changes."""
        for callback in self.listeners:
            with contextlib.suppress(Exception):
                # Don't let listener errors crash the UI
                callback()


class ComparisonDisplayTable(Widget):
    """Minimal table for side-by-side record comparison with colour highlighting."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Table:
        """Render the table with current state using Rich Table."""
        current = self.state.queue.current
        if not current or not hasattr(current, "display_dataframe"):
            return self._create_loading_table()

        if self.state.compact_view_mode:
            return self._render_compact_view(current.display_dataframe)
        else:
            return self._render_detailed_view()

    def _create_loading_table(self) -> Table:
        """Create a simple loading table."""
        loading_table = Table(show_header=False, show_lines=False)
        loading_table.add_column("")
        loading_table.add_row("Loading...")
        return loading_table

    def _render_compact_view(self, display_df: pl.DataFrame) -> Table:
        """Render compact view - one row per field, deduplicated columns."""
        current = self.state.queue.current
        if not current or display_df.is_empty():
            return self._create_loading_table()

        # Create Rich Table
        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=False,
            row_styles=[],
            box=None,
            padding=(0, 1),
        )

        # Add field name column
        table.add_column("Field", style="bright_white", min_width=20, max_width=50)

        # Add columns for each display column (deduplicated)
        current_assignments = self.state.current_assignments
        for display_col_index in range(len(current.display_columns)):
            col_num = display_col_index + 1
            duplicate_count = len(current.duplicate_groups[display_col_index])

            # Create header with duplicate count indicator
            if duplicate_count > 1:
                header_text = f"{col_num} (×{duplicate_count})"
            else:
                header_text = str(col_num)

            if display_col_index in current_assignments:
                group = current_assignments[display_col_index]
                colour, symbol = GroupStyler.get_style(group)
                header = f"[{colour}]{symbol} {header_text}[/]"
                table.add_column(header, style=colour, min_width=15, max_width=50)
            else:
                table.add_column(header_text, style="dim", min_width=15, max_width=50)

        # Group by field_name and create compact rows
        field_groups = (
            display_df.group_by("field_name")
            .agg([pl.col("leaf_id"), pl.col("value")])
            .sort("field_name")
        )  # Sort to maintain consistent field order

        # Filter out empty groups and create table rows
        for field_row in field_groups.iter_rows(named=True):
            field_name = field_row["field_name"]
            leaf_ids_in_group = field_row["leaf_id"]
            values_in_group = field_row["value"]

            # Create a mapping from leaf_id to value for this field
            leaf_to_value = {}
            for leaf_id, value in zip(leaf_ids_in_group, values_in_group, strict=True):
                leaf_to_value[leaf_id] = value

            # Build row data using representative leaf IDs
            row_data = [field_name]
            for representative_leaf_id in current.display_columns:
                cell_value = leaf_to_value.get(representative_leaf_id, "")
                row_data.append(cell_value)

            # Only add row if it has some data
            if any(cell for cell in row_data[1:]):  # Skip field name in check
                table.add_row(*row_data)

        return table

    def _render_detailed_view(self) -> Table:
        """Render detailed view - source attribution with deduplication."""
        current = self.state.queue.current
        if not current or current.display_dataframe.is_empty():
            return self._create_loading_table()

        # Create Rich Table with minimal styling - focus on colour
        table = Table(
            show_header=True,
            header_style="bold",
            show_lines=False,
            row_styles=[],
            box=None,
            padding=(0, 1),
        )

        # Add field name column
        table.add_column("Field", style="bright_white", min_width=20, max_width=50)

        # Add columns for each display column with group styling
        current_assignments = self.state.current_assignments
        for display_col_index, _ in enumerate(current.display_columns):
            col_num = display_col_index + 1
            duplicate_count = len(current.duplicate_groups[display_col_index])

            # Create header with duplicate count indicator
            if duplicate_count > 1:
                header_text = f"{col_num} (×{duplicate_count})"
            else:
                header_text = str(col_num)

            if display_col_index in current_assignments:
                group = current_assignments[display_col_index]
                colour, symbol = GroupStyler.get_style(group)
                header = f"[{colour}]{symbol} {header_text}[/]"
                table.add_column(header, style=colour, min_width=15, max_width=50)
            else:
                table.add_column(header_text, style="dim", min_width=15, max_width=50)

        # Group by field and source for detailed view
        field_source_groups = (
            current.display_dataframe.group_by(["field_name", "source_name"])
            .agg([pl.col("leaf_id"), pl.col("value")])
            .sort(["field_name", "source_name"])  # Sort for consistency
        )

        # Organize data for detailed view with source attribution
        field_source_data = {}
        for row in field_source_groups.iter_rows(named=True):
            field_name = row["field_name"]
            source_name = row["source_name"]
            leaf_ids = row["leaf_id"]
            values = row["value"]

            if field_name not in field_source_data:
                field_source_data[field_name] = {}

            # Map leaf_ids to values for this field+source combination
            for leaf_id, value in zip(leaf_ids, values, strict=True):
                field_source_data[field_name][f"{field_name} ({source_name})"] = {
                    leaf_id: value
                }

        # Add data rows with source attribution
        for field_name in sorted(field_source_data.keys()):
            source_data = field_source_data[field_name]
            # Add separator between field groups
            if len([r for r in table.rows]) > 0:
                separator_row = ["─" * 15] + ["─" * 8] * len(current.display_columns)
                table.add_row(*separator_row, style="dim")

            # Add rows for each source's version of this field
            for source_field_name in sorted(source_data.keys()):
                leaf_value_map = source_data[source_field_name]
                row_data = [source_field_name]

                # Build row using representative leaf IDs
                for representative_leaf_id in current.display_columns:
                    cell_value = leaf_value_map.get(representative_leaf_id, "")
                    row_data.append(cell_value)

                # Only add row if it has some data (empty row filtering)
                if any(cell.strip() for cell in row_data[1:]):
                    table.add_row(*row_data)

        return table

    def _on_state_change(self) -> None:
        """Handle state changes by refreshing the display."""
        self.refresh()


class GroupStyler:
    """Generate consistent colours and symbols with cycling to minimise duplicates."""

    # High contrast colours distributed to avoid similar adjacents
    COLOURS = [
        "red",
        "blue",
        "green",
        "yellow",
        "magenta",
        "cyan",
        "bright_red",
        "bright_green",
        "bright_blue",
        "bright_yellow",
        "bright_magenta",
        "bright_cyan",
        "white",
        "bright_white",
    ]

    # Distinct Unicode symbols for visual differentiation
    SYMBOLS = [
        "■",
        "●",
        "▲",
        "◆",
        "★",
        "⬢",
        "♦",
        "▼",
        "○",
        "△",
        "◇",
        "☆",
        "⬡",
        "✦",
        "✧",
        "⟐",
    ]

    # Class-level tracking for consistent assignments
    _group_styles = {}  # group_name -> (colour, symbol)
    _used_colours = set()
    _used_symbols = set()
    _colour_index = 0
    _symbol_index = 0

    @classmethod
    def get_style(cls, group_name: str) -> tuple[str, str]:
        """Get consistent colour and symbol for a group name using cycling."""
        # Return cached style if already assigned
        if group_name in cls._group_styles:
            return cls._group_styles[group_name]

        # Assign next available colour
        colour = cls._get_next_colour()
        symbol = cls._get_next_symbol()

        # Cache the assignment
        cls._group_styles[group_name] = (colour, symbol)
        return colour, symbol

    @classmethod
    def _get_next_colour(cls) -> str:
        """Get the next colour in cycle, avoiding duplicates when possible."""
        # If we haven't used all colours yet, find an unused one
        if len(cls._used_colours) < len(cls.COLOURS):
            while cls.COLOURS[cls._colour_index] in cls._used_colours:
                cls._colour_index = (cls._colour_index + 1) % len(cls.COLOURS)

        colour = cls.COLOURS[cls._colour_index]
        cls._used_colours.add(colour)
        cls._colour_index = (cls._colour_index + 1) % len(cls.COLOURS)

        return colour

    @classmethod
    def _get_next_symbol(cls) -> str:
        """Get the next symbol in cycle, avoiding duplicates when possible."""
        # If we haven't used all symbols yet, find an unused one
        if len(cls._used_symbols) < len(cls.SYMBOLS):
            while cls.SYMBOLS[cls._symbol_index] in cls._used_symbols:
                cls._symbol_index = (cls._symbol_index + 1) % len(cls.SYMBOLS)

        symbol = cls.SYMBOLS[cls._symbol_index]
        cls._used_symbols.add(symbol)
        cls._symbol_index = (cls._symbol_index + 1) % len(cls.SYMBOLS)

        return symbol

    @classmethod
    def get_display_text(cls, group_name: str, count: int) -> tuple[str, str]:
        """Get formatted display text with colour and symbol."""
        colour, symbol = cls.get_style(group_name)
        text = f"{symbol} {group_name.upper()} ({count})"
        return text, colour

    @classmethod
    def reset(cls):
        """Reset all assignments (useful for testing)."""
        cls._group_styles.clear()
        cls._used_colours.clear()
        cls._used_symbols.clear()
        cls._colour_index = 0
        cls._symbol_index = 0


class PRCurveDisplay(PlotextPlot):
    """Widget for displaying precision-recall curves using textual-plotext."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialize the PR curve display widget."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)
        self._has_plotted = False

    def on_mount(self) -> None:
        """Called when widget is mounted."""
        self.update_plot()

    def update_plot(self) -> None:
        """Update the plot based on current state."""
        # Clear any existing data first
        self.plt.clear_data()
        self.plt.clear_figure()

        # Handle early exit conditions
        if self._should_skip_plotting():
            return

        try:
            self._generate_pr_plot()
        except Exception as e:  # noqa: BLE001
            self._handle_plot_error(e)

    def _should_skip_plotting(self) -> bool:
        """Check if plotting should be skipped due to current state."""
        if self.state.is_loading_eval_data:
            self.plt.title("Loading evaluation data...")
            self._has_plotted = False
            return True

        if self.state.eval_data_error:
            self.plt.title("Error loading data - check status")
            self._has_plotted = False
            return True

        if self.state.eval_data is None:
            self.plt.title("No evaluation data loaded")
            self._has_plotted = False
            return True

        return False

    def _generate_pr_plot(self) -> None:
        """Generate the precision-recall plot with confidence intervals."""
        # Get raw data from EvalData
        pr_data = self.state.eval_data.precision_recall()

        # Compute smooth envelope using PCHIP interpolation
        r_grid, p_upper, p_lower = compute_pr_envelope(pr_data)

        # Compute interpolated PR curve with extrapolation tracking
        r_curve, p_curve, is_extrapolated = interpolate_pr_curve(pr_data)

        # Plot confidence bounds in magenta
        self.plt.plot(r_grid, p_upper, color="magenta", marker="braille")
        self.plt.plot(r_grid, p_lower, color="magenta", marker="braille")

        # Plot PR curve - split by extrapolation status
        # Find transition points between interpolated and extrapolated regions
        transitions = np.where(np.diff(is_extrapolated.astype(int)))[0]
        indices = np.concatenate([[0], transitions + 1, [len(r_curve)]])

        # Plot each segment with appropriate marker style
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            segment_r = r_curve[start_idx:end_idx]
            segment_p = p_curve[start_idx:end_idx]

            if is_extrapolated[start_idx]:
                # Extrapolated region: use braille markers
                self.plt.plot(segment_r, segment_p, color="green", marker="braille")
            else:
                # Interpolated region: use fhd markers for better quality
                self.plt.plot(segment_r, segment_p, color="green", marker="fhd")
        self.plt.xlim(0, 1)
        self.plt.ylim(0, 1)
        self.plt.xlabel("Recall")
        self.plt.ylabel("Precision")
        self.plt.title("Precision-Recall Curve (PCHIP Envelope)")
        self._has_plotted = True

    def _handle_plot_error(self, error: Exception) -> None:
        """Handle errors during plot generation with appropriate user messaging."""
        error_str = str(error).lower()

        if "cannot be empty" in error_str and "judgement" in error_str:
            # Expected case when no judgements exist yet
            self.plt.title("📊 Submit some judgements first")
        else:
            # Unexpected error - log details and show generic message
            self._log_plot_error(error)
            self.plt.title("Plot generation failed")

        self._has_plotted = False

    def _log_plot_error(self, error: Exception) -> None:
        """Log comprehensive error information for debugging plot issues."""
        logger = logging.getLogger(__name__)

        # Log basic error information
        logger.error(f"Plot generation failed - {type(error).__name__}: {error}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Log eval_data state for debugging
        if self.state.eval_data:
            logger.error("EvalData object is available")
            try:
                judgements_info = (
                    f"length={len(self.state.eval_data.judgements)}"
                    if self.state.eval_data.judgements is not None
                    else "None"
                )
                logger.error(f"EvalData judgements: {judgements_info}")
            except Exception as eval_error:  # noqa: BLE001
                logger.error(f"Error accessing EvalData judgements: {eval_error}")
        else:
            logger.error("EvalData object is None")

    def _on_state_change(self) -> None:
        """Handle state changes by updating the plot."""
        self.update_plot()


class StatusBarLeft(Widget):
    """Left side of status bar with entity progress and groups."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the left status widget."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Text:
        """Render entity progress and group assignments."""
        text = Text()

        # Show entity progress info
        total = self.state.queue.total_count
        current_pos = self.state.queue.current_position
        painted_count = self.state.painted_count

        # Entity progress
        current_painted = "■ " if self.state.has_current_assignments() else ""
        text.append(
            f"{current_painted}Entity {current_pos}/{total}", style="bright_white"
        )

        # Painted count if any
        if painted_count > 0:
            text.append(f" | ■ {painted_count} painted", style="green")

        text.append(" | ", style="dim")

        # Show group assignments
        groups = self.state.get_group_counts()

        if not groups:
            text.append("No groups assigned", style="dim")
        else:
            text.append("Groups: ", style="bright_white")
            current_group = self.state.current_group_selection
            for i, (group, count) in enumerate(groups.items()):
                if i > 0:
                    text.append("  ")

                # Use GroupStyler for consistent colour and symbol
                display_text, colour = GroupStyler.get_display_text(group, count)

                # Add underline styling if this is the currently selected group
                if group == current_group:
                    text.append(display_text, style=f"bold {colour} underline")
                else:
                    text.append(display_text, style=f"bold {colour}")

        # Add helpful instruction
        text.append(" | ", style="dim")
        text.append("Press letter to select group", style="yellow")

        return text

    def _on_state_change(self) -> None:
        """Handle state changes by refreshing the display."""
        self.refresh()


class StatusBarRight(Widget):
    """Right side of status bar with status indicator.

    IMPORTANT: This widget has limited display space (~12 characters max).
    Status messages must be concise and include appropriate symbols/emoji.
    Messages longer than MAX_STATUS_LENGTH will be rejected with an error.

    Examples of good status messages:
    - "⏳ Loading"
    - "✓ Loaded"
    - "⚡ Working"
    - "✓ Done"
    - "⚠ Error"
    - "◯ Empty"
    - "📊 Got 5"

    Write status messages to fit within the length limit from the start.
    """

    MAX_STATUS_LENGTH = 12

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the right status widget."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Text:
        """Render status indicator with validation."""
        text = Text()

        if self.state.status_message:
            # Validate message length
            if len(self.state.status_message) > self.MAX_STATUS_LENGTH:
                # This should never happen in production - it's a development error

                logger = logging.getLogger(__name__)
                msg_len = len(self.state.status_message)
                logger.error(
                    f"Status message too long ({msg_len} chars): "
                    f"'{self.state.status_message}'"
                )
                logger.error(
                    "Status messages must be <= 12 characters. Fix the calling code."
                )

                # Show error indicator to make the problem visible
                text.append("⚠ TOO LONG", style="red")
                return text

            # Simple pass-through - the message should already be properly formatted
            text.append(self.state.status_message, style=self.state.status_color)
        else:
            # Show placeholder when no status message
            text.append("○ Ready", style="dim")

        return text

    def _on_state_change(self) -> None:
        """Handle state changes by refreshing the display."""
        self.refresh()


class StatusBar(Widget):
    """Container widget for status bar with left and right sections."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the status bar container."""
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        """Compose the status bar with left and right sections."""
        with Horizontal():
            yield StatusBarLeft(self.state, id="status-left")
            yield StatusBarRight(self.state, id="status-right")


class HelpModal(ModalScreen):
    """Help screen showing commands and shortcuts."""

    def compose(self) -> ComposeResult:
        """Compose the help modal UI."""
        with Container(id="help-dialog"):
            yield Static("Entity resolution tool - Help", id="help-title")
            yield Static(
                dedent("""
                    Simple keyboard shortcuts (comparison view):
                    • Press A - Selects group A
                    • Press 1,3,5 - Assigns columns 1,3,5 to group A
                    • Press B - Switches to group B
                    • Press 2,4 - Assigns columns 2,4 to group B
                    • Press A - Switches back to group A
                    • Press 6 - Assigns column 6 to group A

                    Group selection:
                    • Any letter (a-z) selects that group (26 groups available)
                    • Only one group active at a time
                    • Clear visual feedback shows which group is selected

                    Column assignment:
                    • 1-9: Assign to columns 1-9
                    • 0: Assign to column 10 (if exists)
                    • Numbers only work when a group is selected

                    Navigation shortcuts:
                    • → or Enter - Next entity
                    • ← - Previous entity
                    • Space - Submit current judgement & fetch more samples  
                    • Ctrl+G - Jump to entity number
                    • ? or F1 - Show this help
                    • Esc - Clear current group selection
                    • ` (backtick) - Toggle between compact and detailed view
                    • / (slash) - Toggle precision-recall plot display
                    • Ctrl+C or Ctrl+Q - Quit

                    Visual feedback:
                    • Records are columns, fields are rows
                    • Same field types grouped together for easy comparison
                    • Each group gets unique colour + symbol combination
                    • Column headers show group assignment with coloured symbols
                    • Status bar shows group counts with visual indicators

                    View modes:
                    • Compact view (default): Shows non-empty values, one row per field
                    • Detailed view: Shows source attribution (e.g. "field (source)")
                    • Empty rows are automatically filtered out in both modes

                    Tips for speed:
                    • Press letter with left hand, numbers with right
                    • Same group always gets same colour/symbol for consistency
                    • Work in patterns: group obvious matches first
                    • Use Esc to clear if you get confused
                """).strip(),
                id="help-content",
            )
            yield Button("Close (Esc)", id="close-help")

    @on(Button.Pressed, "#close-help")
    def close_help(self) -> None:
        """Close the help modal."""
        self.dismiss()

    def on_key(self, event) -> None:
        """Handle key events for closing the help modal."""
        if event.key == "escape" or event.key == "question_mark":
            self.dismiss()


class PlotModal(ModalScreen):
    """Modal screen for displaying precision-recall plots."""

    def __init__(self, state: EvaluationState):
        """Initialize the plot modal with evaluation state."""
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        """Compose the plot modal UI."""
        with Container(id="plot-dialog"):
            yield Static("Precision-Recall Curve", id="plot-title")
            yield PRCurveDisplay(self.state, id="plot-widget")
            yield Static("Press Escape to close", id="plot-help")

    def on_key(self, event) -> None:
        """Handle key events for closing the plot modal."""
        if event.key == "escape":
            self.dismiss()


class EntityResolutionApp(App):
    """Main Textual application for entity resolution evaluation."""

    CSS_PATH = Path(__file__).parent / "styles.css"

    BINDINGS = [
        ("right,enter", "next_entity", "Next"),
        ("left", "previous_entity", "Previous"),
        ("space", "submit_and_fetch", "Submit & fetch more"),
        ("ctrl+g", "jump_to_entity", "Jump"),
        ("question_mark,f1", "show_help", "Help"),
        ("escape", "clear_assignments", "Clear"),
        ("grave_accent", "toggle_view_mode", "Toggle view"),
        ("ctrl+q,ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        resolution: ModelResolutionName = DEFAULT_RESOLUTION,
        num_samples: int = 100,
        user: str | None = None,
        warehouse: str | None = None,
    ):
        """Initialise the entity resolution app."""
        super().__init__()

        # Create single centralised state
        self.state = EvaluationState()

        # Set app reference for timer management
        self.state._app_ref = self

        # Initialise state with provided parameters
        self.state.resolution = resolution
        self.state.sample_limit = num_samples
        self.state.user_name = user or ""
        self.state.warehouse = warehouse

    async def on_mount(self) -> None:
        """Initialise the application."""
        await self.authenticate()
        await self.load_samples()
        await self.load_eval_data()
        if self.state.queue.current:
            await self.refresh_display()

    async def authenticate(self) -> None:
        """Authenticate with the server."""
        # Use injected user or fall back to settings
        user_name = self.state.user_name or settings.user
        if not user_name:
            raise MatchboxClientSettingsException("User name is unset.")

        self.state.user_name = user_name
        self.state.user_id = _handler.login(user_name=user_name)

    async def load_samples(self) -> None:
        """Load evaluation samples from the server."""
        samples_dict = await self._fetch_additional_samples(self.state.sample_limit)
        if samples_dict:
            # samples_dict now contains EvaluationItems, not DataFrames
            self.state.queue.add_items(list(samples_dict.values()))

    async def load_eval_data(self) -> None:
        """Load EvalData for precision/recall calculations."""
        if not self.state.resolution:
            return

        self.state.set_eval_data_loading(True)
        self.state.update_status("⏳ Loading", "yellow")

        try:
            await self._perform_eval_data_loading()
        except Exception as e:  # noqa: BLE001
            self._handle_eval_data_error(e)

    async def _perform_eval_data_loading(self) -> None:
        """Perform the actual EvalData loading operation."""
        # Enable debug logging for this operation
        logger = logging.getLogger("matchbox.client.cli.eval.utils")
        logger.setLevel(logging.INFO)

        eval_data = EvalData.from_resolution(self.state.resolution)
        self.state.set_eval_data(eval_data)
        self.state.update_status("✓ Loaded", "green", auto_clear_after=2.0)

        # Log successful loading
        logger.info(
            f"Successfully loaded EvalData for resolution '{self.state.resolution}'"
        )

    def _handle_eval_data_error(self, error: Exception) -> None:
        """Handle errors during EvalData loading with appropriate user messaging."""
        # Log the error details
        logger = logging.getLogger(__name__)
        logger.error(f"EvalData loading failed: {error}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Generate user-friendly error message
        error_msg = self._create_eval_data_error_message(error)

        self.state.set_eval_data_error(error_msg)
        self.state.update_status(error_msg, "red", auto_clear_after=8.0)

    def _create_eval_data_error_message(self, error: Exception) -> str:
        """Create a user-friendly error message for EvalData loading failures."""
        error_details = str(error).lower()

        if "not found" in error_details:
            return f"Model '{self.state.resolution}' not found"
        elif "empty" in error_details:
            return f"No data available for model '{self.state.resolution}'"
        else:
            return f"EvalData error ({type(error).__name__}): {error}"

    async def refresh_display(self) -> None:
        """Refresh display with current queue item."""
        current = self.state.queue.current
        if current:
            # Use the display columns from EvaluationItem
            self.state.set_display_data(current.display_columns)
        else:
            # No current item, clear display
            self.state.clear_display_data()

    def compose(self) -> ComposeResult:
        """Compose the main application UI."""
        yield Header()
        yield Vertical(
            StatusBar(self.state, classes="status-bar", id="status-bar"),
            ComparisonDisplayTable(self.state, id="record-table"),
            id="main-container",
        )
        yield Footer()

    async def on_key(self, event) -> None:
        """Handle keyboard events for group assignment shortcuts."""
        key = event.key

        # Handle navigation keys first - let the normal key binding system handle these
        if key in ["left", "right", "enter", "space"]:
            return

        # Handle special keys
        if key == "escape":
            # Clear current group selection
            self.state.clear_group_selection()
            event.prevent_default()
            return

        # Handle letter key presses (set current group)
        if key.isalpha() and len(key) == 1:
            self.state.set_group_selection(key)
            event.prevent_default()
            return

        # Handle slash key (plot toggle)
        if key == "slash":
            await self._handle_plot_toggle()
            event.prevent_default()
            return

        # Handle number key presses (assign columns to current group)
        column_number = self.state.parse_number_key(key)
        if column_number is not None:
            current_group = self.state.current_group_selection
            if current_group:  # Only assign if we have a group selected
                current = self.state.queue.current
                if current and 1 <= column_number <= len(current.display_columns):
                    self.state.assign_column_to_group(column_number, current_group)
            event.prevent_default()
            return

    async def action_next_entity(self) -> None:
        """Move to the next entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_next()
            await self.refresh_display()
        else:
            # All done - auto-submit everything and quit
            await self.action_submit_and_fetch()
            await self.action_quit()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_previous()
            await self.refresh_display()

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments and held letters for current entity."""
        self.state.clear_current_assignments()
        self.state.clear_group_selection()

    async def action_toggle_view_mode(self) -> None:
        """Toggle between compact and detailed view modes."""
        self.state.toggle_view_mode()

    async def _handle_plot_toggle(self) -> None:
        """Simple plot toggle with proper state checking."""
        # Basic state checks
        if self.state.is_loading_eval_data:
            self.state.update_status("⏳ Loading", "yellow", auto_clear_after=2.0)
            return

        if self.state.eval_data_error:
            self.state.update_status("⚠ Error", "red", auto_clear_after=2.0)
            return

        if self.state.eval_data is None:
            self.state.update_status("⚠ No data", "red", auto_clear_after=2.0)
            return

        # Check data sufficiency without try/except
        pr_data = self.state.eval_data.precision_recall()
        if pr_data is None or len(pr_data) < 2:
            self.state.update_status("∅ Sparse", "yellow", auto_clear_after=2.0)
            return

        # Refresh judgements before showing plot
        success = await self._refresh_judgements_for_plot()
        if success:
            self.push_screen(PlotModal(self.state))

    async def _refresh_judgements_for_plot(self) -> bool:
        """Refresh judgements data and update status appropriately.

        Returns:
            bool: True if refresh was successful, False otherwise
        """
        self.state.update_status("⏳ Loading", "yellow")

        try:
            self.state.eval_data.refresh_judgements()
            self._update_judgements_status()
            return True
        except Exception as e:  # noqa: BLE001
            self._handle_judgements_refresh_error(e)
            return False

    def _update_judgements_status(self) -> None:
        """Update status based on current judgements count."""
        judgements_count = (
            len(self.state.eval_data.judgements)
            if self.state.eval_data.judgements is not None
            else 0
        )

        if judgements_count > 0:
            self.state.update_status(
                f"📊 Got {judgements_count}", "green", auto_clear_after=2.0
            )
        else:
            self.state.update_status("◯ Empty", "dim", auto_clear_after=2.0)

    def _handle_judgements_refresh_error(self, error: Exception) -> None:
        """Handle errors during judgements refresh with appropriate status."""
        error_str = str(error).lower()

        if "cannot be empty" in error_str and "judgement" in error_str:
            self.state.update_status("◯ Empty", "dim", auto_clear_after=4.0)
        else:
            self.state.update_status("⚠ Error", "red", auto_clear_after=4.0)
            logger.error(f"Failed to refresh judgements: {error}")

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    async def action_submit_and_fetch(self) -> None:
        """Submit all painted entities, remove from queue, fetch new samples."""
        painted_items = self.state.queue.painted_items
        painted_count = len(painted_items)

        if painted_count == 0:
            self.state.update_status("◯ Nothing", "dim", auto_clear_after=2.0)
            return

        # Update status to show we're submitting
        self.state.is_submitting = True
        self.state.update_status("⚡ Sending", "yellow")

        # Submit each painted item
        successful_submissions = 0
        for item in painted_items:
            judgement = item.to_judgement(self.state.user_id)
            _handler.send_eval_judgement(judgement=judgement)
            successful_submissions += 1

        # Remove all painted items from queue (they're done forever)
        self.state.queue.submit_all_painted()

        # Update status to show completion
        remaining_count = self.state.queue.total_count
        self.state.update_status("✓ Sent", "green")

        # Log detailed submission info for debugging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Successfully submitted {successful_submissions}/{painted_count} "
            "painted entities"
        )
        logger.info(
            f"Removed submitted items from queue, {remaining_count} entities remaining"
        )

        # Refresh display to show current entity (queue auto-advances)
        await self.refresh_display()

        # Fetch new samples to backfill the queue
        await self._backfill_samples()

        self.state.is_submitting = False

        # Show final status
        final_count = self.state.queue.total_count
        if final_count > remaining_count:
            self.state.update_status("✓ Ready", "green", auto_clear_after=4.0)
            # Log detailed backfill info
            logger.info(f"Queue backfilled: now has {final_count} entities available")
        else:
            self.state.update_status("✓ Done", "green", auto_clear_after=4.0)
            # Log completion info
            logger.info(
                f"Submission complete: {final_count} entities remaining in queue"
            )

    async def _backfill_samples(self) -> None:
        """Fetch new samples to replace submitted ones."""
        try:
            await self._perform_backfill_operation()
        except Exception as e:  # noqa: BLE001
            self._handle_backfill_error(e)

    async def _perform_backfill_operation(self) -> None:
        """Perform the backfill operation with appropriate logging."""
        current_count = self.state.queue.total_count
        desired_count = self.state.sample_limit
        needed = max(0, desired_count - current_count)

        logger = logging.getLogger(__name__)

        # Handle case where queue is already at capacity
        if needed <= 0:
            self.state.update_status("✓ Ready", "green")
            logger.info(f"Queue already at capacity: {current_count}/{desired_count}")
            return

        # Fetch new samples
        self.state.update_status("⚡ Fetching", "yellow")
        logger.info(
            f"Backfilling queue: need {needed} samples "
            f"to reach limit of {desired_count}"
        )

        new_samples_dict = await self._fetch_additional_samples(needed)

        if new_samples_dict and len(new_samples_dict) > 0:
            await self._process_new_samples(new_samples_dict, logger)
        else:
            self._handle_no_samples_available(needed, logger)

    async def _process_new_samples(self, new_samples_dict: dict, logger) -> None:
        """Process newly fetched samples and update the queue."""
        new_items = list(new_samples_dict.values())
        self.state.queue.add_items(new_items)

        self.state.update_status("✓ Ready", "green")
        logger.info(f"Successfully added {len(new_items)} new samples to queue")

        # Refresh display if currently viewing an empty state
        if self.state.current_df is None and len(new_items) > 0:
            await self.refresh_display()

    def _handle_no_samples_available(self, needed: int, logger) -> None:
        """Handle the case where no new samples are available."""
        self.state.update_status("◯ Empty", "dim")
        logger.warning(f"No new samples available - requested {needed} but got none")

    def _handle_backfill_error(self, error: Exception) -> None:
        """Handle errors during backfill operation."""
        self.state.update_status("⚠ Error", "red")

        logger = logging.getLogger(__name__)
        logger.error(f"Failed to fetch replacement samples: {error}")
        logger.error(f"Exception type: {type(error).__name__}")

    async def _fetch_additional_samples(
        self, count: int
    ) -> dict[int, EvaluationItem] | None:
        """Fetch additional samples from the server."""
        try:
            # Temporarily patch warehouse setting if provided
            original_warehouse = None
            if self.state.warehouse:
                original_warehouse = getattr(settings, "default_warehouse", None)
                settings.default_warehouse = self.state.warehouse

            try:
                return get_samples(
                    n=count,
                    resolution=self.state.resolution,
                    user_id=self.state.user_id,
                    clients={},
                    use_default_client=True,
                )
            finally:
                # Restore original warehouse setting
                if self.state.warehouse:
                    if original_warehouse:
                        settings.default_warehouse = original_warehouse
                    elif hasattr(settings, "default_warehouse"):
                        delattr(settings, "default_warehouse")
        except Exception:  # noqa: BLE001
            return None

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


if __name__ == "__main__":
    app = EntityResolutionApp()
    app.run()
