"""Modal screens for entity resolution evaluation."""

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Static

HELP_TEXT = """
A model thinks this data is one entity. Is it?

[b]Paint the columns to label this as one or more entities.[/b]

• Each column is data from a single source.
• Each row is a field from that source.

If you think the model got it right, paint all columns the same.

If you think column 1 and 2 belong to one entity, and 3-5 another,
paint them that way.

When you're done, submit, and do another!

• Press letter (a-z): Select that group (26 groups available)
• Press number (1-9, 0): Assign column to selected group
• Esc: Clear group selection
""".strip()

NO_SAMPLES_TEXT = """
No samples are available for this resolution.

Possible reasons:
• All clusters have been recently judged by you
• The resolution has no probability data
• No clusters exist for this resolution

Press Ctrl+Q to quit.
""".strip()


class HelpModal(ModalScreen):
    """Help screen showing commands and shortcuts."""

    def compose(self) -> ComposeResult:
        """Compose the help modal UI."""
        with Container(id="help-dialog"):
            yield Static("Paint the columns!", id="help-title")
            yield Static(HELP_TEXT, id="help-content")
            yield Button("Close (Esc)", id="close-help")

    @on(Button.Pressed, "#close-help")
    def close_help(self) -> None:
        """Close the help modal."""
        self.dismiss()

    def on_key(self, event: events.Key) -> None:
        """Handle key events for closing the help modal."""
        if event.key in ("escape", "question_mark", "f1"):
            self.dismiss()


class NoSamplesModal(ModalScreen):
    """Modal screen showing no samples are available."""

    def compose(self) -> ComposeResult:
        """Compose the no samples modal UI."""
        with Container(id="no-samples-dialog"):
            yield Static("No samples available", id="no-samples-title")
            yield Static(NO_SAMPLES_TEXT, id="no-samples-content")
            yield Button("Quit (Ctrl+Q)", id="quit-no-samples")

    @on(Button.Pressed, "#quit-no-samples")
    def quit_app(self) -> None:
        """Quit the application."""
        self.app.exit()

    def on_key(self, event: events.Key) -> None:
        """Handle key events for the no samples modal."""
        if event.key in ("escape", "ctrl+q", "ctrl+c"):
            self.app.exit()
