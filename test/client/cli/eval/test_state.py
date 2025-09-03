"""Unit tests for state management components."""

from unittest.mock import Mock

import pytest

from matchbox.client.cli.eval.state import EvaluationQueue, EvaluationState


class TestEvaluationQueue:
    """Test the EvaluationQueue class."""

    @pytest.fixture
    def queue(self):
        """Create a fresh queue for each test."""
        return EvaluationQueue()

    @pytest.fixture
    def sample_items(self):
        """Create sample evaluation items for testing."""
        items = []
        for i in range(3):
            item = Mock()
            item.cluster_id = f"cluster_{i}"
            item.is_painted = False
            item.assignments = {}
            item.display_columns = [1, 2, 3]
            item.duplicate_groups = [[1], [2], [3]]
            items.append(item)
        return items

    def test_empty_queue_properties(self, queue):
        """Test properties of an empty queue."""
        assert queue.current is None
        assert queue.current_position == 0
        assert queue.total_count == 0
        assert queue.painted_count == 0
        assert len(queue.painted_items) == 0

    def test_add_items(self, queue, sample_items):
        """Test adding items to the queue."""
        queue.add_items(sample_items)

        assert queue.total_count == 3
        assert queue.current is sample_items[0]
        assert queue.current_position == 1

    def test_move_next(self, queue, sample_items):
        """Test moving to the next item."""
        queue.add_items(sample_items)

        # Initially at position 1 (first item)
        assert queue.current_position == 1
        assert queue.current is sample_items[0]

        # Move to next
        queue.move_next()
        assert queue.current_position == 2
        assert queue.current is sample_items[1]

        # Move to next again
        queue.move_next()
        assert queue.current_position == 3
        assert queue.current is sample_items[2]

        # Move to next wraps around
        queue.move_next()
        assert queue.current_position == 1
        assert queue.current is sample_items[0]

    def test_move_previous(self, queue, sample_items):
        """Test moving to the previous item."""
        queue.add_items(sample_items)

        # Initially at position 1 (first item)
        assert queue.current_position == 1
        assert queue.current is sample_items[0]

        # Move to previous wraps around
        queue.move_previous()
        assert queue.current_position == 3
        assert queue.current is sample_items[2]

        # Move to previous
        queue.move_previous()
        assert queue.current_position == 2
        assert queue.current is sample_items[1]

    def test_painted_items_tracking(self, queue, sample_items):
        """Test tracking of painted items."""
        queue.add_items(sample_items)

        # No painted items initially
        assert queue.painted_count == 0
        assert len(queue.painted_items) == 0

        # Paint first and third items
        sample_items[0].is_painted = True
        sample_items[2].is_painted = True

        assert queue.painted_count == 2
        painted = queue.painted_items
        assert len(painted) == 2
        assert sample_items[0] in painted
        assert sample_items[2] in painted
        assert sample_items[1] not in painted

    def test_submit_painted(self, queue, sample_items):
        """Test submitting painted items removes them from queue."""
        queue.add_items(sample_items)

        # Paint first and third items
        sample_items[0].is_painted = True
        sample_items[2].is_painted = True

        # Submit painted items
        painted_items = queue.painted_items
        queue.submit_painted(painted_items)

        # Check remaining queue
        assert queue.total_count == 1
        assert queue.current is sample_items[1]
        assert queue.painted_count == 0

    def test_clear(self, queue, sample_items):
        """Test clearing the queue."""
        queue.add_items(sample_items)
        assert queue.total_count == 3

        queue.clear()
        assert queue.total_count == 0
        assert queue.current is None
        assert queue.current_position == 0


class TestEvaluationState:
    """Test the EvaluationState class."""

    @pytest.fixture
    def state(self):
        """Create a fresh state for each test."""
        return EvaluationState()

    @pytest.fixture
    def mock_current_item(self):
        """Create a mock current queue item."""
        item = Mock()
        item.cluster_id = "test_cluster"
        item.dataframe = Mock()
        item.assignments = {0: "a", 2: "b"}
        item.display_columns = [1, 2, 3, 4]
        item.duplicate_groups = [[1], [2], [3], [4]]
        return item

    def test_initial_state(self, state):
        """Test initial state values."""
        assert state.sample_limit == 100
        assert not state.current_group_selection
        assert state.compact_view_mode is True
        assert state.show_plot is False
        assert state.eval_data is None
        assert state.is_loading_eval_data is False
        assert state.eval_data_error is None
        assert not state.status_message
        assert state.status_colour == "bright_white"
        assert len(state.listeners) == 0

    def test_queue_delegation(self, state):
        """Test that state properly delegates to queue."""
        assert state.painted_count == state.queue.painted_count
        assert state.current_cluster_id is None  # No current item
        assert state.current_df is None  # No current item

    def test_current_assignments(self, state, mock_current_item):
        """Test current assignments property."""
        # No current item
        assert state.current_assignments == {}

        # With current item
        state.queue.items.append(mock_current_item)
        assignments = state.current_assignments
        assert assignments == {0: "a", 2: "b"}

    def test_group_selection(self, state):
        """Test group selection functionality."""
        listener = Mock()
        state.add_listener(listener)

        # Set valid group
        state.set_group_selection("A")
        assert state.current_group_selection == "a"  # Lowercased
        listener.assert_called_once()

        # Clear selection
        listener.reset_mock()
        state.clear_group_selection()
        assert not state.current_group_selection
        listener.assert_called_once()

        # Invalid group selection
        listener.reset_mock()
        state.set_group_selection("123")  # Not alpha
        assert not state.current_group_selection  # Unchanged
        listener.assert_not_called()

    def test_view_mode_toggle(self, state):
        """Test view mode toggling."""
        listener = Mock()
        state.add_listener(listener)

        assert state.compact_view_mode is True

        state.toggle_view_mode()
        assert state.compact_view_mode is False
        listener.assert_called_once()

        state.toggle_view_mode()
        assert state.compact_view_mode is True

    def test_eval_data_management(self, state):
        """Test eval data state management."""
        listener = Mock()
        state.add_listener(listener)

        # Set loading
        state.set_eval_data_loading(True)
        assert state.is_loading_eval_data is True
        assert state.eval_data_error is None
        listener.assert_called()

        # Set error
        listener.reset_mock()
        state.set_eval_data_error("Test error")
        assert state.eval_data_error == "Test error"
        assert state.is_loading_eval_data is False
        listener.assert_called()

        # Set data
        listener.reset_mock()
        test_data = Mock()
        state.set_eval_data(test_data)
        assert state.eval_data is test_data
        assert state.is_loading_eval_data is False
        assert state.eval_data_error is None
        listener.assert_called()

        # Clear data
        listener.reset_mock()
        state.clear_eval_data()
        assert state.eval_data is None
        assert state.eval_data_error is None
        # clear_eval_data doesn't notify listeners

    def test_column_assignment(self, state, mock_current_item):
        """Test column assignment functionality."""
        state.queue.items.append(mock_current_item)
        listener = Mock()
        state.add_listener(listener)

        # Assign column 2 to group "c"
        state.assign_column_to_group(2, "c")
        assert mock_current_item.assignments[1] == "c"  # Column 2 = index 1
        listener.assert_called()

        # Invalid column number
        listener.reset_mock()
        initial_assignments = dict(mock_current_item.assignments)
        state.assign_column_to_group(10, "d")  # Column doesn't exist
        assert mock_current_item.assignments == initial_assignments
        listener.assert_not_called()

    def test_clear_assignments(self, state, mock_current_item):
        """Test clearing assignments."""
        state.queue.items.append(mock_current_item)
        listener = Mock()
        state.add_listener(listener)

        # Clear assignments
        state.clear_current_assignments()
        assert len(mock_current_item.assignments) == 0
        listener.assert_called()

    def test_number_key_parsing(self, state):
        """Test number key parsing."""
        assert state.parse_number_key("1") == 1
        assert state.parse_number_key("5") == 5
        assert state.parse_number_key("0") == 10
        assert state.parse_number_key("a") is None
        assert state.parse_number_key("11") is None

    def test_has_current_assignments(self, state, mock_current_item):
        """Test checking for current assignments."""
        # No current item
        assert state.has_current_assignments() is False

        # Current item with no assignments
        mock_current_item.assignments = {}
        state.queue.items.append(mock_current_item)
        assert state.has_current_assignments() is False

        # Current item with assignments
        mock_current_item.assignments = {0: "a"}
        assert state.has_current_assignments() is True

    def test_get_group_counts(self, state, mock_current_item):
        """Test group count calculation."""
        # No current item
        assert state.get_group_counts() == {}

        # Current item with assignments
        state.queue.items.append(mock_current_item)
        state.current_group_selection = "c"

        counts = state.get_group_counts()

        # Group "a" has column 0 (1 item in duplicate group)
        # Group "b" has column 2 (1 item in duplicate group)
        # Group "c" is selected but has no assignments (0 items)
        # Columns 1,3 are unassigned (2 items total)
        expected = {"a": 1, "b": 1, "c": 0, "unassigned": 2}
        assert counts == expected

    def test_status_management(self, state):
        """Test status message management."""
        listener = Mock()
        state.add_listener(listener)

        # Update status
        state.update_status("Test message", "red")
        assert state.status_message == "Test message"
        assert state.status_colour == "red"
        listener.assert_called()

        # Clear status
        listener.reset_mock()
        state.clear_status()
        assert not state.status_message
        assert state.status_colour == "bright_white"
        listener.assert_called()

    def test_listener_error_handling(self, state):
        """Test that listener errors are now propagated (no longer swallowed)."""

        # Add a listener that raises an exception
        def failing_listener():
            raise ValueError("Test error")

        state.add_listener(failing_listener)

        # This should now raise an exception (error handling was removed)
        with pytest.raises(ValueError, match="Test error"):
            state.set_group_selection("a")
