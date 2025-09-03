"""Unit tests for UI widgets."""

from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest
from rich.table import Table
from rich.text import Text

from matchbox.client.cli.eval.widgets.status import (
    StatusBar,
    StatusBarLeft,
    StatusBarRight,
)
from matchbox.client.cli.eval.widgets.styling import GroupStyler
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable


class TestComparisonDisplayTable:
    """Test the comparison display table widget."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        state = Mock()
        state.add_listener = Mock()
        state.compact_view_mode = True
        state.current_assignments = {}
        return state

    @pytest.fixture
    def mock_current_item(self):
        """Create a mock current queue item."""
        item = Mock()
        item.display_columns = [1, 2, 3]
        item.duplicate_groups = [[1], [2], [3]]

        # Create a simple test dataframe
        item.display_dataframe = pl.DataFrame(
            {
                "field_name": ["name", "name", "address", "address"],
                "leaf_id": [1, 2, 1, 3],
                "value": ["Company A", "Company A", "123 Main St", "123 Main Street"],
                "source_name": ["crn", "duns", "crn", "cdms"],
            }
        )

        return item

    def test_table_initialization(self, mock_state):
        """Test table widget initialization."""
        table = ComparisonDisplayTable(mock_state)

        assert table.state is mock_state
        mock_state.add_listener.assert_called_once()

    def test_render_no_current_item(self, mock_state):
        """Test rendering when no current item exists."""
        mock_state.queue.current = None
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)
        # Should show loading table

    def test_render_current_item_no_display_dataframe(self, mock_state):
        """Test rendering when current item has no display_dataframe."""
        mock_current = Mock()
        mock_current.spec = []  # No display_dataframe attribute
        mock_state.queue.current = mock_current
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)

    def test_render_compact_view(self, mock_state, mock_current_item):
        """Test rendering in compact view mode."""
        mock_state.queue.current = mock_current_item
        mock_state.compact_view_mode = True
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)
        # Should have field name column plus display columns
        assert len(result.columns) == 4  # Field + 3 display columns

    def test_render_detailed_view(self, mock_state, mock_current_item):
        """Test rendering in detailed view mode."""
        mock_state.queue.current = mock_current_item
        mock_state.compact_view_mode = False
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)
        assert len(result.columns) == 4  # Field + 3 display columns

    def test_render_with_assignments(self, mock_state, mock_current_item):
        """Test rendering with column assignments."""
        mock_state.queue.current = mock_current_item
        mock_state.current_assignments = {0: "a", 2: "b"}
        table = ComparisonDisplayTable(mock_state)

        result = table.render()

        assert isinstance(result, Table)
        # Assignments should affect column headers with colors/symbols

    def test_state_change_triggers_refresh(self, mock_state):
        """Test that state changes trigger table refresh."""
        table = ComparisonDisplayTable(mock_state)
        table.refresh = Mock()

        # Get the registered listener
        listener = mock_state.add_listener.call_args[0][0]

        # Call the listener
        listener()

        table.refresh.assert_called_once()


class TestStatusBarLeft:
    """Test the left status bar widget."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        state = Mock()
        state.add_listener = Mock()
        state.queue.total_count = 5
        state.queue.current_position = 2
        state.painted_count = 1
        state.has_current_assignments.return_value = True
        state.get_group_counts.return_value = {"a": 3, "b": 2}
        state.current_group_selection = "a"
        return state

    def test_status_bar_left_initialization(self, mock_state):
        """Test left status bar initialization."""
        status_left = StatusBarLeft(mock_state)

        assert status_left.state is mock_state
        mock_state.add_listener.assert_called_once()

    def test_render_with_data(self, mock_state):
        """Test rendering with data."""
        status_left = StatusBarLeft(mock_state)

        result = status_left.render()

        assert isinstance(result, Text)
        # Should contain entity position, painted count, groups info

    def test_render_no_groups(self, mock_state):
        """Test rendering when no groups are assigned."""
        mock_state.get_group_counts.return_value = {}
        status_left = StatusBarLeft(mock_state)

        result = status_left.render()

        assert isinstance(result, Text)

    def test_render_no_painted_items(self, mock_state):
        """Test rendering when no items are painted."""
        mock_state.painted_count = 0
        mock_state.has_current_assignments.return_value = False
        status_left = StatusBarLeft(mock_state)

        result = status_left.render()

        assert isinstance(result, Text)

    def test_state_change_triggers_refresh(self, mock_state):
        """Test that state changes trigger refresh."""
        status_left = StatusBarLeft(mock_state)
        status_left.refresh = Mock()

        # Get the registered listener
        listener = mock_state.add_listener.call_args[0][0]

        # Call the listener
        listener()

        status_left.refresh.assert_called_once()


class TestStatusBarRight:
    """Test the right status bar widget."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        state = Mock()
        state.add_listener = Mock()
        state.status_message = ""
        state.status_color = "bright_white"
        return state

    def test_status_bar_right_initialization(self, mock_state):
        """Test right status bar initialization."""
        status_right = StatusBarRight(mock_state)

        assert status_right.state is mock_state
        mock_state.add_listener.assert_called_once()

    def test_render_no_message(self, mock_state):
        """Test rendering when no status message exists."""
        status_right = StatusBarRight(mock_state)

        result = status_right.render()

        assert isinstance(result, Text)
        # Should show default "○ Ready"

    def test_render_with_message(self, mock_state):
        """Test rendering with status message."""
        mock_state.status_message = "✓ Done"
        mock_state.status_color = "green"
        status_right = StatusBarRight(mock_state)

        result = status_right.render()

        assert isinstance(result, Text)

    def test_render_message_too_long(self, mock_state):
        """Test rendering with message that's too long."""
        mock_state.status_message = "This message is way too long for the status bar"
        status_right = StatusBarRight(mock_state)

        result = status_right.render()

        assert isinstance(result, Text)
        # Should show error indicator

    def test_max_status_length_constant(self):
        """Test that MAX_STATUS_LENGTH is set correctly."""
        assert StatusBarRight.MAX_STATUS_LENGTH == 12

    def test_state_change_triggers_refresh(self, mock_state):
        """Test that state changes trigger refresh."""
        status_right = StatusBarRight(mock_state)
        status_right.refresh = Mock()

        # Get the registered listener
        listener = mock_state.add_listener.call_args[0][0]

        # Call the listener
        listener()

        status_right.refresh.assert_called_once()


class TestStatusBar:
    """Test the status bar container widget."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        state = Mock()
        return state

    def test_status_bar_initialisation(self, mock_state):
        """Test status bar container initialization."""
        status_bar = StatusBar(mock_state)

        assert status_bar.state is mock_state

    def test_status_bar_compose(self, mock_state):
        """Test status bar composition."""

        status_bar = StatusBar(mock_state)

        # Mock the Horizontal container to avoid app context issues
        with patch(
            "matchbox.client.cli.eval.widgets.status.Horizontal"
        ) as mock_horizontal:
            mock_container = MagicMock()
            mock_horizontal.return_value.__enter__.return_value = mock_container
            mock_horizontal.return_value.__exit__.return_value = None

            # Get the composed widgets
            composed = list(status_bar.compose())

            # Should yield left and right status bars
            assert len(composed) == 2  # StatusBarLeft and StatusBarRight


class TestGroupStyler:
    """Test the group styling utility."""

    def teardown_method(self):
        """Reset GroupStyler state after each test."""
        GroupStyler.reset()

    def test_get_style_consistent(self):
        """Test that get_style returns consistent results for same group."""
        style1 = GroupStyler.get_style("group_a")
        style2 = GroupStyler.get_style("group_a")

        assert style1 == style2
        assert len(style1) == 2  # (color, symbol)

    def test_get_style_different_groups(self):
        """Test that different groups get different styles."""
        style_a = GroupStyler.get_style("group_a")
        style_b = GroupStyler.get_style("group_b")

        assert style_a != style_b
        # At least one of color or symbol should be different
        assert style_a[0] != style_b[0] or style_a[1] != style_b[1]

    def test_get_display_text(self):
        """Test display text formatting."""
        text, _ = GroupStyler.get_display_text("test", 5)

        assert "TEST" in text  # Should be uppercase
        assert "(5)" in text  # Should include count
        assert text.startswith(
            GroupStyler.get_style("test")[1]
        )  # Should start with symbol

    def test_color_cycling(self):
        """Test that colors cycle when exhausted."""
        # Get more styles than available colors
        num_colors = len(GroupStyler.COLOURS)
        styles = []

        for i in range(num_colors + 2):
            style = GroupStyler.get_style(f"group_{i}")
            styles.append(style)

        # Should have used all colors and started cycling
        colors_used = [style[0] for style in styles]
        unique_colors = set(colors_used)
        assert len(unique_colors) <= num_colors

    def test_symbol_cycling(self):
        """Test that symbols cycle when exhausted."""
        # Get more styles than available symbols
        num_symbols = len(GroupStyler.SYMBOLS)
        styles = []

        for i in range(num_symbols + 2):
            style = GroupStyler.get_style(f"group_{i}")
            styles.append(style)

        # Should have used all symbols and started cycling
        symbols_used = [style[1] for style in styles]
        unique_symbols = set(symbols_used)
        assert len(unique_symbols) <= num_symbols

    def test_reset(self):
        """Test that reset clears all state."""
        # Assign some styles
        GroupStyler.get_style("test1")
        GroupStyler.get_style("test2")

        # Reset
        GroupStyler.reset()

        # State should be cleared
        assert len(GroupStyler._group_styles) == 0
        assert len(GroupStyler._used_colours) == 0
        assert len(GroupStyler._used_symbols) == 0
        assert GroupStyler._colour_index == 0
        assert GroupStyler._symbol_index == 0

    def test_avoid_duplicates_when_possible(self):
        """Test that duplicates are avoided when colors/symbols are available."""
        # Get several styles
        styles = []
        for i in range(5):
            style = GroupStyler.get_style(f"group_{i}")
            styles.append(style)

        # All colors should be different (when enough are available)
        colors = [style[0] for style in styles]
        symbols = [style[1] for style in styles]

        assert len(set(colors)) == len(colors)  # All unique
        assert len(set(symbols)) == len(symbols)  # All unique
