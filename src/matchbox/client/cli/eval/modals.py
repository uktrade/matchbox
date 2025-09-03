"""Modal screens for entity resolution evaluation."""

from textwrap import dedent

from textual import on
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from matchbox.client.cli.eval.plot.widget import PRCurveDisplay


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

    def __init__(self, state):
        """Initialise the plot modal with evaluation state."""
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
