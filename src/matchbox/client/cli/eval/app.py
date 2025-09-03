"""Main application for entity resolution evaluation."""

import logging
import traceback
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.cli.eval.handlers import EvaluationHandlers
from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.utils import EvalData, get_samples, temp_warehouse
from matchbox.client.cli.eval.widgets.status import StatusBar
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable
from matchbox.common.exceptions import MatchboxClientSettingsException
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName

logger = logging.getLogger(__name__)


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

        # Create handlers for input and actions
        self.handlers = EvaluationHandlers(self)

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
        eval_logger = logging.getLogger("matchbox.client.cli.eval.utils")
        eval_logger.setLevel(logging.INFO)

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
        if "empty" in error_details:
            return f"No data available for model '{self.state.resolution}'"
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
        """Delegate key handling to handlers."""
        await self.handlers.handle_key_input(event)

    # Action methods - delegate to handlers
    async def action_next_entity(self) -> None:
        """Move to the next entity via queue rotation."""
        await self.handlers.action_next_entity()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity via queue rotation."""
        await self.handlers.action_previous_entity()

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments and held letters for current entity."""
        await self.handlers.action_clear_assignments()

    async def action_toggle_view_mode(self) -> None:
        """Toggle between compact and detailed view modes."""
        await self.handlers.action_toggle_view_mode()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        await self.handlers.action_show_help()

    async def action_submit_and_fetch(self) -> None:
        """Submit all painted entities, remove from queue, fetch new samples."""
        await self.handlers.action_submit_and_fetch()

    async def _fetch_additional_samples(self, count: int) -> dict[int, Any] | None:
        """Fetch additional samples from the server."""
        default_client = None
        with temp_warehouse(self.state.warehouse):
            try:
                if self.state.warehouse:
                    default_client = create_engine(self.state.warehouse)

                return get_samples(
                    n=count,
                    resolution=self.state.resolution,
                    user_id=self.state.user_id,
                    clients={},
                    use_default_client=True,
                    default_client=default_client,
                )
            except Exception:  # noqa: BLE001
                return None
            finally:
                if default_client:
                    default_client.dispose()

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


if __name__ == "__main__":
    app = EntityResolutionApp()
    app.run()
