"""Status bar components for entity resolution evaluation."""

import logging

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget

from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.widgets.styling import GroupStyler

logger = logging.getLogger(__name__)


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
            text.append(self.state.status_message, style=self.state.status_colour)
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
