"""Textual-based entity resolution evaluation tool."""

from collections import deque
from pathlib import Path
from textwrap import dedent
from typing import Callable

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

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.utils import EvaluationItem, get_samples
from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName


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

        # Display State (derived from current queue item)
        self.display_leaf_ids: list[int] = []

        # User/Connection State
        self.user_name: str = ""
        self.user_id: int | None = None
        self.resolution: str = ""
        self.warehouse: str | None = None

        # Status/Feedback State
        self.status_message: str = ""
        self.is_submitting: bool = False

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

    def update_status(self, message: str) -> None:
        """Update status message."""
        self.status_message = message
        self._notify_listeners()

    def clear_status(self) -> None:
        """Clear status message."""
        self.status_message = ""
        self._notify_listeners()

    def add_listener(self, callback: Callable) -> None:
        """Add a callback to be notified when state changes."""
        self.listeners.append(callback)

    def _notify_listeners(self) -> None:
        """Notify all listeners of state changes."""
        for callback in self.listeners:
            try:
                callback()
            except Exception:
                # Don't let listener errors crash the UI
                pass


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
        for display_col_index, _representative_leaf_id in enumerate(
            current.display_columns
        ):
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
    """Right side of status bar with status indicator."""

    def __init__(self, state: EvaluationState, **kwargs):
        """Initialise the right status widget."""
        super().__init__(**kwargs)
        self.state = state
        self.state.add_listener(self._on_state_change)

    def render(self) -> Text:
        """Render status indicator."""
        text = Text()

        if self.state.status_message:
            # Convert verbose messages to symbolic indicators with labels
            message = self.state.status_message.lower()
            if "submitting" in message or "fetching" in message:
                status_text = "⚡ Working"
                status_style = "yellow"
            elif "submitted" in message or "complete" in message or "ready" in message:
                status_text = "✓ Done"
                status_style = "green"
            elif "error" in message or "failed" in message:
                status_text = "⚠ Error"
                status_style = "red"
            elif "nothing to submit" in message:
                status_text = "◯ Nothing"
                status_style = "dim"
            else:
                status_text = "● Status"
                status_style = "blue"

            text.append(status_text, style=status_style)
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

        # Initialise state with provided parameters
        self.state.resolution = resolution
        self.state.sample_limit = num_samples
        self.state.user_name = user or ""
        self.state.warehouse = warehouse

    async def on_mount(self) -> None:
        """Initialise the application."""
        await self.authenticate()
        await self.load_samples()
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

    def on_key(self, event) -> None:
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

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    async def action_submit_and_fetch(self) -> None:
        """Submit all painted entities, remove from queue, fetch new samples."""
        painted_items = self.state.queue.painted_items
        painted_count = len(painted_items)

        if painted_count == 0:
            self.state.update_status("Nothing to submit - no fully painted entities")
            self.set_timer(2.0, self.state.clear_status)
            return

        # Update status to show we're submitting
        self.state.is_submitting = True
        self.state.update_status(f"✓ Submitting {painted_count} painted entities...")

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
        self.state.update_status(
            f"✓ Submitted {successful_submissions}/{painted_count} entities, "
            f"removed from queue ({remaining_count} remaining)"
        )

        # Refresh display to show current entity (queue auto-advances)
        await self.refresh_display()

        # Fetch new samples to backfill the queue
        await self._backfill_samples()

        self.state.is_submitting = False

        # Show final status
        final_count = self.state.queue.total_count
        if final_count > remaining_count:
            self.state.update_status(
                f"✓ Queue ready - {final_count} entities available"
            )
        else:
            self.state.update_status(
                f"✓ Submission complete - {final_count} entities in queue"
            )

        # Clear status after a longer delay to show the result
        self.set_timer(4.0, self.state.clear_status)

    async def _backfill_samples(self) -> None:
        """Fetch new samples to replace submitted ones."""
        try:
            current_count = self.state.queue.total_count
            desired_count = self.state.sample_limit
            needed = max(0, desired_count - current_count)

            if needed > 0:
                self.state.update_status(f"Fetching {needed} replacement samples...")

                # Fetch replacement samples
                new_samples_dict = await self._fetch_additional_samples(needed)

                if new_samples_dict and len(new_samples_dict) > 0:
                    # new_samples_dict now contains EvaluationItems
                    new_items = list(new_samples_dict.values())
                    self.state.queue.add_items(new_items)

                    # Update status
                    self.state.update_status(
                        f"Queue updated: added {len(new_items)} new samples"
                    )

                    # If we're currently viewing an empty state, refresh display
                    if not self.state.current_df and len(new_items) > 0:
                        await self.refresh_display()
                else:
                    self.state.update_status(
                        "No new samples available - queue exhausted"
                    )
            else:
                self.state.update_status("Queue is at capacity")

        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            self.state.update_status(f"Error fetching samples: {error_msg}")

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
        except Exception:
            return None

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


if __name__ == "__main__":
    app = EntityResolutionApp()
    app.run()
