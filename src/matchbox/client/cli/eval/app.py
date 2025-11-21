"""Main application for entity resolution evaluation."""

import logging
from collections import deque
from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Label, TabbedContent, TabPane
from textual.widgets._tabbed_content import ContentTabs

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.modals import HelpModal, NoSamplesModal
from matchbox.client.cli.eval.widgets.styling import get_display_text, get_group_style
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
    AUTO_FOCUS = None

    BINDINGS = [
        ("space", "submit", "Submit"),
        ("shift+right", "skip", "Skip"),
        ("escape", "clear", "Clear"),
        ("question_mark,f1", "show_help", "Help"),
        ("ctrl+q,ctrl+c", "quit", "Quit"),
        ("up", "page_up", "▲ Page"),
        ("down", "page_down", "▼ Page"),
        ("left", "previous_tab", "◀ Tab"),
        ("right", "next_tab", "▶ Tab"),
    ]

    current_group: reactive[str] = reactive("")
    status: reactive[tuple[str, str]] = reactive(("● Ready", "dim"))

    sample_limit: int
    resolution: ModelResolutionPath
    user_id: int
    user_name: str
    dag: DAG
    show_help: bool

    queue: EvaluationQueue
    timer: Timer | None = None

    current_page_by_tab: dict[int, int]
    rows_per_page: int
    cols_per_page: int

    def __init__(
        self,
        resolution: ModelResolutionName,
        user: str,
        num_samples: int = 5,
        dag: DAG | None = None,
        show_help: bool = False,
        sample_file: str | None = None,
    ) -> None:
        """Initialise the entity resolution app.

        Args:
            resolution: The model resolution to evaluate
            num_samples: Number of clusters to sample for evaluation
            user: Username for authentication (overrides settings)
            dag: Pre-loaded DAG with warehouse location attached
            show_help: Whether to show help on start
            sample_file: Path to pre-compiled sample file.
                If set, ignores resolutions.
        """
        super().__init__()

        self.queue = EvaluationQueue()
        self.sample_limit = num_samples
        self.user_name = user
        self.dag = dag
        self.resolution: ModelResolutionPath = dag.get_model(resolution).resolution_path
        self.show_help = show_help
        self.current_page_by_tab = {}
        self.rows_per_page = 20
        self.cols_per_page = 10
        self.sample_file = sample_file

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
            Vertical(id="content-container"),
            id="main-container",
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Initialise the application."""
        self.authenticate()

        if self.dag is None:
            raise RuntimeError(
                "DAG not loaded. EntityResolutionApp requires a pre-loaded DAG."
            )

        self.load_samples()

        if self.queue.total_count == 0:
            self.action_show_no_samples()
            return

        await self._load_current_item()

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
            current = self.queue.current
            if not current:
                event.stop()
                return

            key_position = 10 if key == "0" else int(key)
            tab_index = self._get_current_tab_index()
            col_start = tab_index * self.cols_per_page
            col_end = min(col_start + self.cols_per_page, len(current.display_columns))

            if 1 <= key_position <= col_end - col_start:
                actual_col_idx = col_start + (key_position - 1)
                current.assignments[actual_col_idx] = self.current_group
                self._refresh_current_tab_content()
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
    def _get_current_tab_index(self) -> int:
        """Get the currently active tab index."""
        tabs_list = list(self.query(TabbedContent))
        if not tabs_list:
            return 0

        active_pane = tabs_list[0].active
        if active_pane and active_pane.startswith("tab-"):
            try:
                return int(active_pane.split("-")[1])
            except (ValueError, IndexError):
                return 0

        return 0

    def _calculate_cols_per_page(self) -> int:
        """Calculate how many columns fit on screen based on terminal width."""
        available_width = max(80, self.size.width - 26)
        num_cols = available_width // 27
        return max(4, min(10, num_cols))

    def _calculate_rows_per_page(self) -> int:
        """Calculate how many rows fit on screen."""
        return max(10, self.size.height - 8)

    def _build_tab_label(self, tab_index: int) -> str:
        """Build tab label with coloured blocks representing column states."""
        current = self.queue.current
        if not current:
            return "█" * self.cols_per_page

        blocks = []
        col_start = tab_index * self.cols_per_page
        col_end = min(col_start + self.cols_per_page, len(current.display_columns))

        for col_idx in range(col_start, col_end):
            if col_idx in current.assignments:
                group = current.assignments[col_idx]
                _, colour = get_group_style(group)
                blocks.append(f"[{colour}]█[/]")
            else:
                blocks.append("[dim]█[/]")

        return "".join(blocks)

    def _update_tab_labels(self, tabs: TabbedContent, num_tabs: int) -> None:
        """Update tab labels to reflect current assignment state."""
        if num_tabs <= 1:
            return

        content_tabs_list = list(tabs.query(ContentTabs))
        if not content_tabs_list:
            return

        content_tabs = content_tabs_list[0]
        tab_widgets = list(content_tabs.query("Tab"))

        for idx, tab_widget in enumerate(tab_widgets):
            if idx >= num_tabs:
                break

            new_label_markup = self._build_tab_label(idx)
            new_label_text = Text.from_markup(new_label_markup)
            tab_widget.label = new_label_text
            tab_widget.refresh()

        content_tabs.refresh()

    def _update_current_table(
        self,
        current: EvaluationItem,
        tab_index: int,
        col_start: int,
        col_end: int,
        row_start: int,
        row_end: int,
    ) -> None:
        """Update the table in the current tab pane."""
        tabs_list = list(self.query(TabbedContent))
        if not tabs_list:
            return

        tabs = tabs_list[0]
        current_pane = tabs.get_pane(f"tab-{tab_index}")
        if not current_pane:
            return

        table = current_pane.query_one(ComparisonDisplayTable)
        table.load_comparison(current, col_start, col_end, row_start, row_end)

    def _refresh_current_tab_content(self) -> None:
        """Refresh the data and labels within the currently visible tab."""
        current = self.queue.current
        if not current:
            return

        self.rows_per_page = self._calculate_rows_per_page()
        self.cols_per_page = self._calculate_cols_per_page()

        num_cols = len(current.display_columns)
        tab_index = self._get_current_tab_index()

        # Calculate which columns are visible in this tab (horizontal pagination)
        col_start = tab_index * self.cols_per_page
        col_end = min(col_start + self.cols_per_page, num_cols)

        # Calculate which rows are visible in this page (in-tab vertical pagination)
        page = self.current_page_by_tab.get(tab_index, 0)
        row_start = page * self.rows_per_page
        row_end = row_start + self.rows_per_page

        tabs_list = list(self.query(TabbedContent))
        if not tabs_list:
            return

        tabs = tabs_list[0]
        num_tabs = (num_cols + self.cols_per_page - 1) // self.cols_per_page
        self._update_tab_labels(tabs, num_tabs)
        self._update_current_table(
            current, tab_index, col_start, col_end, row_start, row_end
        )

    async def _load_current_item(self) -> None:
        """Load current queue item into the display.

        Builds a multi-tab interface where each tab shows a horizontal slice of columns
        (because there may be too many columns to fit on screen at once).
        Each tab contains a table that can be paginated vertically.
        """
        current = self.queue.current
        if not current:
            return

        if len(current.display_columns) > 80:
            logger.warning(
                f"Cluster {current.cluster_id} has {len(current.display_columns)} "
                "columns - skipping (max 80 supported)"
            )
            self._update_status("⚠ Cluster skipped", "red", clear_after=3.0)
            self.queue.remove_current()
            await self._load_current_item()
            return

        self.current_group = ""
        self.rows_per_page = self._calculate_rows_per_page()
        self.cols_per_page = self._calculate_cols_per_page()

        container = self.query_one("#content-container", Vertical)
        for child in list(container.children):
            child.remove()

        num_cols = len(current.display_columns)
        num_tabs = (num_cols + self.cols_per_page - 1) // self.cols_per_page
        self.current_page_by_tab = {i: 0 for i in range(num_tabs)}

        tabs = TabbedContent()
        await container.mount(tabs)

        for tab_idx in range(num_tabs):
            label = self._build_tab_label(tab_idx)
            col_start = tab_idx * self.cols_per_page
            col_end = min(col_start + self.cols_per_page, num_cols)

            table = ComparisonDisplayTable()
            table.load_comparison(current, col_start, col_end, 0, self.rows_per_page)

            pane = TabPane(label, table, id=f"tab-{tab_idx}")
            await tabs.add_pane(pane)

        self._update_status_labels()

    def _update_status_labels(self) -> None:
        """Update both status bar labels."""
        self.query_one("#status-left", Label).update(self._build_status_left())
        self.query_one("#status-right", Label).update(self._build_status_right())

    def _compute_group_counts(self) -> dict[str, int]:
        """Compute group counts for current entity.

        Display columns may represent multiple identical underlying entities (shown as
        "x2", "x3", etc.). When we assign a display column to a group, we're actually
        assigning all the duplicates it represents. This counts the total underlying
        entities per group, not just display columns.
        """
        current = self.queue.current
        if not current:
            return {}

        counts = {}

        # Count underlying entities (not just display columns) for each assigned group
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
        return f"[{colour}]{message}[/]" if message else "[dim]● Ready[/]"

    def _update_status(
        self,
        message: str,
        colour: str = "dim",
        clear_after: float | None = None,
    ) -> None:
        """Update status message with optional auto-clear."""
        if self.timer:
            self.timer.stop()

        self.status = (message, colour)

        if clear_after:
            self.timer = self.set_timer(
                clear_after, lambda: self._update_status("● Ready", "dim")
            )

    def _navigate_page(self, delta: int) -> None:
        """Navigate pages by delta (-1 for up, +1 for down)."""
        tab_idx = self._get_current_tab_index()
        current = self.queue.current
        if not current or not current.display_data:
            return

        new_page = self.current_page_by_tab.get(tab_idx, 0) + delta
        max_page = max(0, (len(current.display_data) - 1) // self.rows_per_page)

        if 0 <= new_page <= max_page:
            self.current_page_by_tab[tab_idx] = new_page
            self._refresh_current_tab_content()

    # Public methods
    def authenticate(self) -> None:
        """Authenticate with the server."""
        user_name = self.user_name or settings.user
        if not user_name:
            raise MatchboxClientSettingsException("User name is unset.")

        self.user_name = user_name
        self.user_id = _handler.login(user_name=user_name)

    def load_samples(self) -> None:
        """Load evaluation samples from the server."""
        needed = max(0, self.sample_limit - self.queue.total_count)

        if needed <= 0:
            return

        self._update_status("⚡ Fetching", "yellow")

        try:
            new_samples_dict = get_samples(
                n=needed,
                resolution=self.resolution.name,
                user_id=self.user_id,
                dag=self.dag,
                sample_file=self.sample_file,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch samples: {type(e).__name__}: {e}")
            new_samples_dict = None

        if new_samples_dict:
            self.queue.add_items(list(new_samples_dict.values()))

        if self.queue.total_count == 0:
            self._update_status("○ No data", "yellow")
            logger.warning(f"No samples available for resolution '{self.resolution}'")
        else:
            self._update_status("✓ Ready", "green")

    # Action methods (public interface)
    def action_page_up(self) -> None:
        """Navigate to previous page in current tab."""
        self._navigate_page(-1)

    def action_page_down(self) -> None:
        """Navigate to next page in current tab."""
        self._navigate_page(1)

    def action_previous_tab(self) -> None:
        """Navigate to previous tab."""
        tabs = self.query_one(TabbedContent).query_one(ContentTabs)
        tabs.action_previous_tab()

    def action_next_tab(self) -> None:
        """Navigate to next tab."""
        tabs = self.query_one(TabbedContent).query_one(ContentTabs)
        tabs.action_next_tab()

    async def action_skip(self) -> None:
        """Skip current entity (moves to back of queue)."""
        if self.queue.total_count > 1:
            self.queue.skip_current()
            await self._load_current_item()
        else:
            self.action_submit()
            if self.queue.total_count == 0:
                self.exit()

    async def action_submit(self) -> None:
        """Submit current entity if fully painted."""
        current = self.queue.current

        if not current:
            return

        is_painted = len(current.assignments) == len(current.display_columns)

        if not is_painted:
            self._update_status("⚠ Incomplete", "yellow", clear_after=3.0)
            return

        if self.user_id is None:
            raise RuntimeError("User ID is not set")

        try:
            judgement = create_judgement(current, self.user_id)
            _handler.send_eval_judgement(judgement)
        except Exception as exc:
            self._update_status("⚠ Send failed", "red", clear_after=4.0)
            logger.exception(f"Failed to submit: {exc}")
            return

        self.queue.remove_current()
        self.load_samples()
        await self._load_current_item()
        self._update_status("✓ Sent", "green", clear_after=2.0)

    def action_clear(self) -> None:
        """Clear current entity's group assignments."""
        if current := self.queue.current:
            current.assignments.clear()
            self.current_group = ""

            self._refresh_current_tab_content()
            self._update_status_labels()

    def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    def action_show_no_samples(self) -> None:
        """Show the no samples modal."""
        self.push_screen(NoSamplesModal())

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
