"""Minimal widget tests - verify custom configuration and display logic."""

from unittest.mock import Mock

from matchbox.client.cli.eval.widgets.assignment import AssignmentBar
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable


class TestComparisonDisplayTable:
    """Test table widget configuration and display logic."""

    def test_build_header_formatting(self) -> None:
        """Test that headers correctly format numbers, counts, and symbols."""
        table = ComparisonDisplayTable()

        # 1. Basic Numbering
        # Position 1, single record -> "1"
        header = table._build_header(1, [1], None)
        assert header.plain == "1"

        # 2. Duplicate Counts
        # Position 2, 3 duplicate records -> "2 (×3)"
        header = table._build_header(2, [1, 2, 3], None)
        assert header.plain == "2 (×3)"

        # 3. Assignments & Symbols
        # Group 'a' (◇ diamond), Position 1 -> "◇ 1"
        header = table._build_header(1, [1], "a")
        assert "◇" in header.plain
        assert "1" in header.plain

        # Group 'b' (⬤ circle), Position 2, duplicates -> "⬤ 2 (×2)"
        header = table._build_header(2, [1, 2], "b")
        assert "⬤" in header.plain
        assert "2 (×2)" in header.plain

    def test_build_header_dimming(self) -> None:
        """Test that unassigned headers are dimmed and assigned ones are colored."""
        table = ComparisonDisplayTable()

        # Unassigned: Should be dim
        header_unassigned = table._build_header(1, [1], None)
        # Rich text spans structure: Span(start, end, style)
        # We expect 'dim' style on the number
        assert "dim" in str(header_unassigned.spans)

        # Assigned: Should NOT be dim (will have group color)
        header_assigned = table._build_header(1, [1], "a")
        # Should not have dim style (it gets stylized with group color)
        assert "dim" not in str(header_assigned.spans)


class TestAssignmentBar:
    """Test assignment bar widget status display."""

    def test_initializes_empty(self) -> None:
        """Test bar starts empty."""
        bar = AssignmentBar()
        assert bar.positions == []

    def test_reset(self) -> None:
        """Test reset creates correct number of empty slots."""
        bar = AssignmentBar()
        bar.update = Mock()  # Mock update to prevent rendering errors/side effects

        bar.reset(5)
        assert len(bar.positions) == 5
        assert all(p is None for p in bar.positions)

    def test_set_position_updates_state(self) -> None:
        """Test setting a position updates internal state."""
        bar = AssignmentBar()
        bar.update = Mock()
        bar.reset(3)

        bar.set_position(1, "a", "red")

        assert bar.positions[0] is None
        assert bar.positions[1] is not None
        assert bar.positions[1].letter == "a"
        assert bar.positions[1].colour == "red"

    def test_render_bar_logic(self) -> None:
        """Test the visual rendering logic (letters vs dots)."""
        bar = AssignmentBar()
        bar.update = Mock()

        # 1. Initial empty state
        bar.reset(3)
        bar.update.assert_called_with("[dim]•[/dim][dim]•[/dim][dim]•[/dim]")

        # 2. Set middle to 'a' -> shows letter 'a'
        bar.set_position(1, "a", "red")
        bar.update.assert_called_with("[dim]•[/dim][red]a[/red][dim]•[/dim]")

        # 3. Set last to 'a' -> adjacent same group shows dot '•'
        bar.set_position(2, "a", "red")
        bar.update.assert_called_with("[dim]•[/dim][red]a[/red][red]•[/red]")

        # 4. Set first to 'b' -> new group shows letter 'b'
        bar.set_position(0, "b", "blue")
        bar.update.assert_called_with("[blue]b[/blue][red]a[/red][red]•[/red]")
