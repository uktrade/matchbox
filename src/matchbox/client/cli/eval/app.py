"""Main application for entity resolution evaluation."""

import logging
from collections import deque
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Label

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.modals import HelpModal, NoSamplesModal
from matchbox.client.cli.eval.widgets.styling import get_display_text
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable
from matchbox.client.dags import DAG
from matchbox.client.eval import EvaluationItem, create_judgement, get_samples
from matchbox.common.dtos import ModelResolutionName, ModelResolutionPath
from matchbox.common.exceptions import MatchboxClientSettingsException

logger = logging.getLogger(__name__)


class EvaluationQueue:
    """Deque-based queue with current item always at front."""

    def __init__(self) -> None:
        """Initialise the queue."""
        self.items: deque[EvaluationItem] = deque()

    @property
    def current(self) -> EvaluationItem | None:
        """Get the current item (always at index 0)."""
        return self.items[0] if self.items else None

    @property
    def total_count(self) -> int:
        """Total number of items in queue."""
        return len(self.items)

    def skip_current(self) -> None:
        """Move current to back of queue."""
        if len(self.items) > 1:
            self.items.rotate(-1)

    def remove_current(self) -> EvaluationItem | None:
        """Remove and return current item."""
        return self.items.popleft() if self.items else None

    def add_items(self, items: list[EvaluationItem]) -> int:
        """Add new items to queue, preventing duplicates.

        Returns:
            Number of unique items added.
        """
        if not items:
            return 0

        existing_ids = {item.cluster_id for item in self.items}
        unique_items = [item for item in items if item.cluster_id not in existing_ids]

        if unique_items:
            self.items.extend(unique_items)

        return len(unique_items)


class EntityResolutionApp(App):
    """Main Textual application for entity resolution evaluation."""

    CSS_PATH = Path(__file__).parent / "styles.tcss"
    TITLE = "Matchbox evaluate"
    SUB_TITLE = "match labelling tool"

    # Triggered by action_* methods
    BINDINGS = [
        ("right", "skip", "Skip"),
        ("space", "submit", "Submit"),
        ("escape", "clear", "Clear"),
        ("question_mark,f1", "show_help", "Help"),
        ("ctrl+q,ctrl+c", "quit", "Quit"),
    ]

    # Reactive variables that trigger UI updates
    current_group: reactive[str] = reactive("")
    status: reactive[tuple[str, str]] = reactive(("○ Ready", "dim"))

    sample_limit: int
    resolution: ModelResolutionPath
    user_id: int
    user_name: str
    dag: DAG
    show_help: bool

    queue: EvaluationQueue
    timer: Timer | None = None

    def __init__(
        self,
        resolution: ModelResolutionName,
        user: str,
        num_samples: int = 5,
        dag: DAG | None = None,
        show_help: bool = False,
    ) -> None:
        """Initialise the entity resolution app.

        Args:
            resolution: The model resolution to evaluate
            num_samples: Number of clusters to sample for evaluation
            user: Username for authentication (overrides settings)
            dag: Pre-loaded DAG with warehouse location attached
            show_help: Whether to show help on start
        """
        super().__init__()

        self.queue = EvaluationQueue()
        self.sample_limit = num_samples
        self.user_name = user
        self.dag = dag
        self.resolution = dag.get_model(resolution).resolution_path
        self.show_help = show_help

    # Lifecycle methods
    def compose(self) -> ComposeResult:
        """Compose the main application UI."""
        yield Vertical(
            Horizontal(
                Label(id="status-left"),
                Label(id="status-right"),
                id="status-bar",
                classes="status-bar",
            ),
            ComparisonDisplayTable(id="record-table"),
            id="main-container",
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Initialise the application."""
        await self.authenticate()

        if self.dag is None:
            raise RuntimeError(
                "DAG not loaded. EntityResolutionApp requires a pre-loaded DAG."
            )

        await self.load_samples()

        if self.queue.total_count == 0:
            await self._handle_no_samples()
            return

        self._load_current_item()

        if self.show_help:
            self.call_after_refresh(self.action_show_help)

    async def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts for group assignment.

        Textual's basic key event handler. Handles keys beyond BINDINGS.
        """
        key = event.key

        if key.isalpha() and len(key) == 1:
            self.current_group = key.lower()
            event.stop()
            return

        if key.isdigit() and self.current_group:
            col_num = 10 if key == "0" else int(key)
            current = self.queue.current
            if current and col_num <= len(current.display_columns):
                current.assignments[col_num - 1] = self.current_group

                table = self.query_one(ComparisonDisplayTable)
                table.refresh()
                self._update_status_labels()

            event.stop()
            return

    # Reactive watchers
    def watch_status(self, new_value: tuple[str, str]) -> None:
        """React to status changes."""
        self._update_status_labels()

    def watch_current_group(self, new_value: str) -> None:
        """React to current group changes."""
        self._update_status_labels()

    # Private methods
    def _load_current_item(self) -> None:
        """Load current queue item into the display."""
        current = self.queue.current
        if current:
            # Check if cluster has too many columns
            if len(current.display_columns) > 10:
                logger.warning(
                    f"Cluster {current.cluster_id} has {len(current.display_columns)} "
                    "columns. Skipping cluster as only columns 1-10 can be assigned "
                    "via keyboard shortcuts."
                )
                self._update_status("⚠ Cluster skipped", "red", clear_after=3.0)
                self.queue.remove_current()
                self._load_current_item()
                return

            self.current_group = ""

            table = self.query_one(ComparisonDisplayTable)
            table.load_comparison(current)

    def _update_status_labels(self) -> None:
        """Update both status bar labels."""
        self.query_one("#status-left", Label).update(self._build_status_left())
        self.query_one("#status-right", Label).update(self._build_status_right())

    def _compute_group_counts(self) -> dict[str, int]:
        """Compute group counts for current entity."""
        current = self.queue.current
        if not current:
            return {}

        counts = {}

        for display_col_index, group in current.assignments.items():
            if display_col_index < len(current.duplicate_groups):
                duplicate_group_size = len(current.duplicate_groups[display_col_index])
                counts[group] = counts.get(group, 0) + duplicate_group_size

        if self.current_group and self.current_group not in counts:
            counts[self.current_group] = 0

        assigned_display_cols = set(current.assignments.keys())
        unassigned_leaf_count = 0
        for display_col_index in range(len(current.duplicate_groups)):
            if display_col_index not in assigned_display_cols:
                duplicate_group = current.duplicate_groups[display_col_index]
                unassigned_leaf_count += len(duplicate_group)

        if unassigned_leaf_count > 0:
            counts["unassigned"] = unassigned_leaf_count

        return counts

    def _build_status_left(self) -> str:
        """Build left status text with groups."""
        if self.queue.total_count == 0:
            return "[yellow]No samples to evaluate[/]"

        group_counts = self._compute_group_counts()
        if not group_counts:
            return "[dim]No groups assigned[/]"

        group_parts = []
        for group, count in group_counts.items():
            display_text, colour = get_display_text(group, count)

            if group == self.current_group:
                group_parts.append(f"[bold {colour} underline]{display_text}[/]")
            else:
                group_parts.append(f"[bold {colour}]{display_text}[/]")

        return "  ".join(group_parts)

    def _build_status_right(self) -> str:
        """Build right status text with status indicator."""
        message, colour = self.status
        if message:
            return f"[{colour}]{message}[/]"
        return "[dim]○ Ready[/]"

    async def _handle_no_samples(self) -> None:
        """Handle empty queue state."""
        self._update_status("◯ No data", "yellow")
        logger.warning(f"No samples available for resolution '{self.resolution}'.")
        await self.action_show_no_samples()

    def _update_status(
        self,
        message: str,
        colour: str = "dim",
        clear_after: float | None = None,
    ) -> None:
        """Update status message with optional auto-clear.

        Args:
            message: Status message to display
            colour: Colour for the message
            clear_after: Seconds after which to auto-clear
        """
        if self.timer:
            self.timer.stop()

        self.status = (message, colour)

        if clear_after:
            self.timer = self.set_timer(
                clear_after, lambda: self._update_status("○ Ready", "dim")
            )

    # Public methods
    async def authenticate(self) -> None:
        """Authenticate with the server."""
        user_name = self.user_name or settings.user
        if not user_name:
            raise MatchboxClientSettingsException("User name is unset.")

        self.user_name = user_name
        self.user_id = _handler.login(user_name=user_name)

    async def load_samples(self) -> None:
        """Load evaluation samples from the server."""
        needed = max(0, self.sample_limit - self.queue.total_count)

        if needed <= 0:
            return

        self._update_status("⚡ Fetching", "yellow")
        logger.info(
            "Fetching %s samples to reach limit of %s",
            needed,
            self.sample_limit,
        )

        new_samples_dict = None
        try:
            new_samples_dict = get_samples(
                n=needed,
                resolution=self.resolution.name,
                user_id=self.user_id,
                dag=self.dag,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch samples: {type(e).__name__}: {e}")

        if new_samples_dict:
            self.queue.add_items(list(new_samples_dict.values()))

        if self.queue.total_count == 0:
            await self._handle_no_samples()
        else:
            self._update_status("✓ Ready", "green")

    # Action methods (public interface)
    async def action_skip(self) -> None:
        """Skip current entity (moves to back of queue)."""
        if self.queue.total_count > 1:
            self.queue.skip_current()
            self._load_current_item()
        else:
            await self.action_submit()
            if self.queue.total_count == 0:
                self.exit()

    async def action_submit(self) -> None:
        """Submit current entity if fully painted."""
        current = self.queue.current

        if not current:
            return

        # Check if all columns are assigned
        is_painted = len(current.assignments) == len(current.display_columns)
        if not is_painted:
            self._update_status("⚠ Incomplete", "yellow", clear_after=3.0)
            return

        try:
            if self.user_id is None:
                raise RuntimeError("User ID is not set")
            judgement = create_judgement(current, self.user_id)
            _handler.send_eval_judgement(judgement)
        except Exception as exc:
            self._update_status("⚠ Send failed", "red", clear_after=4.0)
            logger.exception(f"Failed to submit: {exc}")
            return

        self.queue.remove_current()
        await self.load_samples()
        self._load_current_item()
        self._update_status("✓ Sent", "green", clear_after=2.0)

    async def action_clear(self) -> None:
        """Clear current entity's group assignments."""
        if current := self.queue.current:
            current.assignments.clear()
            self.current_group = ""

            table = self.query_one(ComparisonDisplayTable)
            table.refresh()
            self._update_status_labels()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    async def action_show_no_samples(self) -> None:
        """Show the no samples modal."""
        self.push_screen(NoSamplesModal())

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
