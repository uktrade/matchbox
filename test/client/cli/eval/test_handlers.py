"""Unit tests for input handlers and actions."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from matchbox.client.cli.eval.handlers import EvaluationHandlers


class TestEvaluationHandlers:
    """Test the EvaluationHandlers class."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock app for testing."""
        app = Mock()
        app.state = Mock()
        app.push_screen = Mock()
        app.refresh_display = AsyncMock()
        app.action_quit = AsyncMock()
        app._fetch_additional_samples = AsyncMock(return_value={})

        # Mock queue.current to return None by default (no current item)
        mock_current = Mock()
        mock_current.display_columns = []
        app.state.queue.current = None

        return app

    @pytest.fixture
    def handlers(self, mock_app):
        """Create handlers instance with mock app."""
        return EvaluationHandlers(mock_app)

    @pytest.fixture
    def mock_event(self):
        """Create a mock key event."""
        event = Mock()
        event.prevent_default = Mock()
        return event

    class TestKeyHandling:
        """Test keyboard event handling."""

        @pytest.mark.asyncio
        async def test_navigation_keys_not_handled(self, handlers, mock_event):
            """Test that navigation keys are passed through to bindings."""
            navigation_keys = ["left", "right", "enter", "space"]

            for key in navigation_keys:
                mock_event.key = key

                await handlers.handle_key_input(mock_event)

                # Navigation keys should not prevent default
                mock_event.prevent_default.assert_not_called()
                mock_event.reset_mock()

        @pytest.mark.asyncio
        async def test_escape_clears_group_selection(self, handlers, mock_event):
            """Test that escape key clears group selection."""
            mock_event.key = "escape"

            await handlers.handle_key_input(mock_event)

            handlers.state.clear_group_selection.assert_called_once()
            mock_event.prevent_default.assert_called_once()

        @pytest.mark.asyncio
        async def test_letter_keys_set_group_selection(self, handlers, mock_event):
            """Test that letter keys set group selection."""
            letters = ["a", "b", "z", "A", "B", "Z"]

            for letter in letters:
                mock_event.key = letter
                handlers.state.set_group_selection.reset_mock()
                mock_event.reset_mock()

                await handlers.handle_key_input(mock_event)

                handlers.state.set_group_selection.assert_called_once_with(letter)
                mock_event.prevent_default.assert_called_once()

        @pytest.mark.asyncio
        async def test_non_alpha_keys_ignored(self, handlers, mock_event):
            """Test that non-alphabetic keys don't set group selection."""
            non_alpha = ["1", "!", "@", "space"]

            for key in non_alpha:
                if key in ["left", "right", "enter", "space"]:
                    continue  # Skip navigation keys

                mock_event.key = key
                handlers.state.set_group_selection.reset_mock()
                mock_event.reset_mock()

                await handlers.handle_key_input(mock_event)

                # Non-alpha keys should not set group selection
                handlers.state.set_group_selection.assert_not_called()

        @pytest.mark.asyncio
        async def test_slash_key_triggers_plot_toggle(self, handlers, mock_event):
            """Test that slash key triggers plot toggle."""
            mock_event.key = "slash"
            handlers.handle_plot_toggle = AsyncMock()

            await handlers.handle_key_input(mock_event)

            handlers.handle_plot_toggle.assert_called_once()
            mock_event.prevent_default.assert_called_once()

        @pytest.mark.asyncio
        async def test_number_keys_assign_columns(self, handlers, mock_event):
            """Test that number keys assign columns to current group."""
            # Set up state
            handlers.state.current_group_selection = "a"
            handlers.state.parse_number_key.return_value = 3

            mock_current = Mock()
            mock_current.display_columns = [1, 2, 3, 4, 5]
            handlers.state.queue.current = mock_current

            mock_event.key = "3"

            await handlers.handle_key_input(mock_event)

            handlers.state.parse_number_key.assert_called_once_with("3")
            handlers.state.assign_column_to_group.assert_called_once_with(3, "a")
            mock_event.prevent_default.assert_called_once()

        @pytest.mark.asyncio
        async def test_number_keys_no_group_selected(self, handlers, mock_event):
            """Test that number keys do nothing when no group is selected."""
            handlers.state.current_group_selection = ""  # No group selected
            handlers.state.parse_number_key.return_value = 3

            mock_event.key = "3"

            await handlers.handle_key_input(mock_event)

            # Should not assign column
            handlers.state.assign_column_to_group.assert_not_called()
            mock_event.prevent_default.assert_called_once()  # Still prevent default

    class TestPlotToggleHandling:
        """Test plot toggle functionality."""

        @pytest.mark.asyncio
        @patch("matchbox.client.cli.eval.handlers.can_show_plot")
        async def test_plot_toggle_cannot_show(self, mock_can_show, handlers):
            """Test plot toggle when plot cannot be shown."""
            mock_can_show.return_value = (False, "⏳ Loading")

            await handlers.handle_plot_toggle()

            # Should update status and not show modal
            handlers.state.update_status.assert_called_once_with(
                "⏳ Loading", "yellow", auto_clear_after=2.0
            )
            handlers.app.push_screen.assert_not_called()

        @pytest.mark.asyncio
        @patch("matchbox.client.cli.eval.handlers.can_show_plot")
        @patch("matchbox.client.cli.eval.handlers.refresh_judgements_for_plot")
        async def test_plot_toggle_can_show_success(
            self, mock_refresh, mock_can_show, handlers
        ):
            """Test plot toggle when plot can be shown and refresh succeeds."""
            mock_can_show.return_value = (True, "")
            mock_refresh.return_value = (True, "📊 Got 5")

            await handlers.handle_plot_toggle()

            # Should show loading, then success status, then modal
            status_calls = handlers.state.update_status.call_args_list
            assert len(status_calls) == 2
            assert status_calls[0][0] == ("⏳ Loading", "yellow")
            assert status_calls[1][0] == ("📊 Got 5", "green")

            handlers.app.push_screen.assert_called_once()

        @pytest.mark.asyncio
        @patch("matchbox.client.cli.eval.handlers.can_show_plot")
        @patch("matchbox.client.cli.eval.handlers.refresh_judgements_for_plot")
        async def test_plot_toggle_can_show_refresh_fails(
            self, mock_refresh, mock_can_show, handlers
        ):
            """Test plot toggle when plot can be shown but refresh fails."""
            mock_can_show.return_value = (True, "")
            mock_refresh.return_value = (False, "⚠ Error")

            await handlers.handle_plot_toggle()

            # Should show loading, then error status, but no modal
            status_calls = handlers.state.update_status.call_args_list
            assert len(status_calls) == 2
            assert status_calls[0][0] == ("⏳ Loading", "yellow")
            assert status_calls[1][0] == ("⚠ Error", "red")

            handlers.app.push_screen.assert_not_called()

    class TestEntityNavigation:
        """Test entity navigation actions."""

        @pytest.mark.asyncio
        async def test_next_entity_multiple_items(self, handlers):
            """Test moving to next entity when multiple items exist."""
            handlers.state.queue.total_count = 3

            await handlers.action_next_entity()

            handlers.state.queue.move_next.assert_called_once()
            handlers.app.refresh_display.assert_called_once()

        @pytest.mark.asyncio
        async def test_next_entity_single_item_submits_and_quits(self, handlers):
            """Test moving to next when only one item submits and quits."""
            handlers.state.queue.total_count = 1
            handlers.action_submit_and_fetch = AsyncMock()

            await handlers.action_next_entity()

            handlers.state.queue.move_next.assert_not_called()
            handlers.action_submit_and_fetch.assert_called_once()
            handlers.app.action_quit.assert_called_once()

        @pytest.mark.asyncio
        async def test_previous_entity_multiple_items(self, handlers):
            """Test moving to previous entity when multiple items exist."""
            handlers.state.queue.total_count = 3

            await handlers.action_previous_entity()

            handlers.state.queue.move_previous.assert_called_once()
            handlers.app.refresh_display.assert_called_once()

        @pytest.mark.asyncio
        async def test_previous_entity_single_item_does_nothing(self, handlers):
            """Test moving to previous when only one item does nothing."""
            handlers.state.queue.total_count = 1

            await handlers.action_previous_entity()

            handlers.state.queue.move_previous.assert_not_called()
            handlers.app.refresh_display.assert_not_called()

    class TestAssignmentActions:
        """Test assignment-related actions."""

        @pytest.mark.asyncio
        async def test_clear_assignments(self, handlers):
            """Test clearing assignments and group selection."""
            await handlers.action_clear_assignments()

            handlers.state.clear_current_assignments.assert_called_once()
            handlers.state.clear_group_selection.assert_called_once()

        @pytest.mark.asyncio
        async def test_toggle_view_mode(self, handlers):
            """Test toggling view mode."""
            await handlers.action_toggle_view_mode()

            handlers.state.toggle_view_mode.assert_called_once()

    class TestModalActions:
        """Test modal-related actions."""

        @pytest.mark.asyncio
        async def test_show_help(self, handlers):
            """Test showing help modal."""
            await handlers.action_show_help()

            handlers.app.push_screen.assert_called_once()
            # Verify it's a HelpModal (by checking the call)
            call_args = handlers.app.push_screen.call_args[0][0]
            assert call_args.__class__.__name__ == "HelpModal"

    class TestSubmitAndFetch:
        """Test submit and fetch functionality."""

        @pytest.mark.asyncio
        async def test_submit_no_painted_items(self, handlers):
            """Test submit when no items are painted."""
            handlers.state.queue.painted_items = []

            (
                successful_submissions,
                painted_items,
            ) = await handlers._submit_painted_items()

            assert successful_submissions == 0
            assert painted_items == []
            handlers.state.update_status.assert_called_once_with(
                "◯ Nothing", "dim", auto_clear_after=2.0
            )

        @pytest.mark.asyncio
        @patch("matchbox.client.cli.eval.handlers._handler.send_eval_judgement")
        async def test_submit_painted_items(self, mock_send, handlers):
            """Test submitting painted items."""
            painted_items = [Mock(), Mock()]
            handlers.state.queue.painted_items = painted_items
            handlers.state.user_id = 123

            for item in painted_items:
                item.to_judgement.return_value = Mock()

            (
                successful_submissions,
                submitted_items,
            ) = await handlers._submit_painted_items()

            assert successful_submissions == 2
            assert submitted_items == painted_items
            assert mock_send.call_count == 2
            handlers.state.update_status.assert_called_once_with("⚡ Sending", "yellow")

        @pytest.mark.asyncio
        async def test_post_submission_update(self, handlers):
            """Test post-submission UI and queue updates."""
            painted_items = [Mock(), Mock()]
            handlers.state.queue.total_count = 0  # No items left after submit

            await handlers._post_submission_update(2, painted_items)

            handlers.state.queue.submit_painted.assert_called_once_with(painted_items)
            handlers.state.update_status.assert_called_once_with("✓ Sent", "green")
            handlers.app.refresh_display.assert_called_once()

        @pytest.mark.asyncio
        async def test_backfill_samples_success(self, handlers):
            """Test successful sample backfilling."""
            handlers.state.queue.total_count = 80
            handlers.state.sample_limit = 100
            handlers.app._fetch_additional_samples = AsyncMock(
                return_value={"1": Mock(), "2": Mock()}
            )
            handlers.state.current_df = Mock()  # Not empty state

            await handlers._backfill_samples()

            handlers.app._fetch_additional_samples.assert_called_once_with(20)
            handlers.state.queue.add_items.assert_called_once()

        @pytest.mark.asyncio
        async def test_backfill_samples_already_at_capacity(self, handlers):
            """Test backfill when queue is already at capacity."""
            handlers.state.queue.total_count = 100
            handlers.state.sample_limit = 100

            await handlers._backfill_samples()

            handlers.app._fetch_additional_samples.assert_not_called()
            handlers.state.update_status.assert_called_with("✓ Ready", "green")

        @pytest.mark.asyncio
        async def test_backfill_samples_no_samples_available(self, handlers):
            """Test backfill when no new samples are available."""
            handlers.state.queue.total_count = 80
            handlers.state.sample_limit = 100
            handlers.app._fetch_additional_samples = AsyncMock(return_value=None)

            await handlers._backfill_samples()

            handlers.state.update_status.assert_called_with("◯ Empty", "dim")
