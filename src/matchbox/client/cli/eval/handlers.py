"""Input handlers and action methods for entity resolution evaluation."""

import logging

from matchbox.client import _handler
from matchbox.client.cli.eval.modals import HelpModal, PlotModal
from matchbox.client.cli.eval.plot.data import (
    can_show_plot,
    refresh_judgements_for_plot,
)

logger = logging.getLogger(__name__)


class EvaluationHandlers:
    """Handles all input events and actions for the evaluation app."""

    def __init__(self, app):
        """Initialise handlers with app reference."""
        self.app = app
        self.state = app.state

    async def handle_key_input(self, event) -> None:
        """Handle keyboard events for group assignment shortcuts."""
        key = event.key

        if key in ["left", "right", "enter", "space"]:
            return

        if key == "escape":
            self.state.clear_group_selection()
            event.prevent_default()
            return

        if key.isalpha() and len(key) == 1:
            self.state.set_group_selection(key)
            event.prevent_default()
            return

        if key == "slash":
            await self.handle_plot_toggle()
            event.prevent_default()
            return

        column_number = self.state.parse_number_key(key)
        if column_number is not None:
            current_group = self.state.current_group_selection
            if current_group:
                current = self.state.queue.current
                if current and 1 <= column_number <= len(current.display_columns):
                    self.state.assign_column_to_group(column_number, current_group)
            event.prevent_default()

    async def handle_plot_toggle(self) -> None:
        """Handle plot toggle with centralised data validation."""
        can_show, status_msg = can_show_plot(self.state)
        if not can_show:
            self.state.update_status(status_msg, "yellow", auto_clear_after=2.0)
            return

        self.state.update_status("⏳ Loading", "yellow")
        success, refresh_status = refresh_judgements_for_plot(self.state)
        if not success:
            self.state.update_status(refresh_status, "red", auto_clear_after=4.0)
            return

        self.state.update_status(refresh_status, "green", auto_clear_after=2.0)
        self.app.push_screen(PlotModal(self.state))

    async def action_next_entity(self) -> None:
        """Move to the next entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_next()
            await self.app.refresh_display()
        else:
            await self.action_submit_and_fetch()
            await self.app.action_quit()

    async def action_previous_entity(self) -> None:
        """Move to the previous entity via queue rotation."""
        if self.state.queue.total_count > 1:
            self.state.queue.move_previous()
            await self.app.refresh_display()

    async def action_clear_assignments(self) -> None:
        """Clear all group assignments for current entity."""
        self.state.clear_current_assignments()
        self.state.clear_group_selection()

    async def action_toggle_view_mode(self) -> None:
        """Toggle between compact and detailed view modes."""
        self.state.toggle_view_mode()

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.app.push_screen(HelpModal())

    async def _submit_painted_items(self) -> tuple[int, list]:
        """Submit all painted items and return the number of successful submissions."""
        painted_items = self.state.queue.painted_items
        if not painted_items:
            self.state.update_status("◯ Nothing", "dim", auto_clear_after=2.0)
            return 0, []

        self.state.is_submitting = True
        self.state.update_status("⚡ Sending", "yellow")

        successful_submissions = 0
        for item in painted_items:
            judgement = item.to_judgement(self.state.user_id)
            _handler.send_eval_judgement(judgement=judgement)
            successful_submissions += 1

        logger.info(
            f"Successfully submitted {successful_submissions}/{len(painted_items)} "
            "painted entities"
        )
        return successful_submissions, painted_items

    async def _post_submission_update(
        self, successful_submissions: int, painted_items: list
    ) -> None:
        """Update queue and UI after submission."""
        if successful_submissions > 0:
            self.state.queue.submit_painted(painted_items)
            remaining_count = self.state.queue.total_count
            self.state.update_status("✓ Sent", "green")
            logger.info(
                "Removed submitted items from queue, %s entities remaining",
                remaining_count,
            )
            await self.app.refresh_display()

    async def action_submit_and_fetch(self) -> None:
        """Submit painted entities, remove them, and fetch new samples."""
        remaining_before_backfill = self.state.queue.total_count

        successful_submissions, painted_items = await self._submit_painted_items()
        await self._post_submission_update(successful_submissions, painted_items)

        if successful_submissions > 0:
            await self._backfill_samples()

        self.state.is_submitting = False

        final_count = self.state.queue.total_count
        if final_count > remaining_before_backfill:
            self.state.update_status("✓ Ready", "green", auto_clear_after=4.0)
            logger.info(f"Queue backfilled: now has {final_count} entities available")
        else:
            self.state.update_status("✓ Done", "green", auto_clear_after=4.0)
            logger.info(
                f"Submission complete: {final_count} entities remaining in queue"
            )

    async def _backfill_samples(self) -> None:
        """Fetch new samples to replace submitted ones."""
        current_count = self.state.queue.total_count
        desired_count = self.state.sample_limit
        needed = max(0, desired_count - current_count)

        if needed <= 0:
            self.state.update_status("✓ Ready", "green")
            logger.info(f"Queue already at capacity: {current_count}/{desired_count}")
            return

        self.state.update_status("⚡ Fetching", "yellow")
        logger.info(
            "Backfilling queue: need %s samples to reach limit of %s",
            needed,
            desired_count,
        )

        new_samples_dict = await self.app._fetch_additional_samples(needed)
        if new_samples_dict:
            await self._process_new_samples(new_samples_dict)
        else:
            self._handle_no_samples_available(needed)

    async def _process_new_samples(self, new_samples_dict: dict) -> None:
        """Process newly fetched samples and update the queue."""
        new_items = list(new_samples_dict.values())
        self.state.queue.add_items(new_items)
        self.state.update_status("✓ Ready", "green")
        logger.info(f"Successfully added {len(new_items)} new samples to queue")

        if self.state.current_df is None and new_items:
            await self.app.refresh_display()

    def _handle_no_samples_available(self, needed: int) -> None:
        """Handle the case where no new samples are available."""
        self.state.update_status("◯ Empty", "dim")
        logger.warning(f"No new samples available - requested {needed} but got none")
