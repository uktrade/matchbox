"""State management for entity resolution evaluation tool."""

import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

from matchbox.client.cli.eval.utils import EvaluationItem


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

    def submit_painted(self, painted_items: list[EvaluationItem]):
        """Remove submitted painted items permanently."""
        self.items = deque([item for item in self.items if item not in painted_items])
        if painted_items and self.items:
            self._position_offset = 0

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
        self.eval_data: Any | None = None  # EvalData object loaded at startup
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
        self.status_colour: str = "bright_white"
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
    def current_df(self) -> Any | None:
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
        colour: str = "bright_white",
        auto_clear_after: float | None = None,
    ) -> None:
        """Update status message with optional colour and auto-clearing.

        Args:
            message: Status message to display
            colour: Colour for the message
            auto_clear_after: Seconds after which to auto-clear (None = no auto-clear)
        """
        # Cancel any existing status timer to prevent conflicts
        if self._status_timer_id is not None:
            self._cancel_status_timer()

        self.status_message = message
        self.status_colour = colour
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
        self.status_colour = "bright_white"
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
            callback()
