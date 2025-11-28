"""AssignmentBar widget for displaying column assignments in the evaluation UI."""

from typing import Any, NamedTuple

from textual.widgets import Static


class GroupSelect(NamedTuple):
    """A letter and colour combination representing a group assignment."""

    letter: str
    colour: str


class AssignmentBar(Static):
    """A status bar widget showing column assignments as a visual bar.

    Each position represents a column in the comparison table. Positions can be:

        - None (unassigned): displayed as dim •
        - GroupSelect: displayed as letter (first occurrence) or • (subsequent blocks)

    Consecutive positions with the same group assignment are rendered in that
    group's colour, showing the letter only at the first position.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialise an empty AssignmentBar."""
        super().__init__(**kwargs)
        self.positions: list[GroupSelect | None] = []
        self._render_bar()

    def reset(self, num_positions: int) -> None:
        """Reset the bar with a new number of positions.

        Args:
            num_positions: The number of columns to represent
        """
        self.positions = [None] * num_positions
        self._render_bar()

    def set_position(self, index: int, letter: str, colour: str) -> None:
        """Set a position to a letter and colour.

        Args:
            index: The position index to set
            letter: The group letter (a-z)
            colour: The colour name for the group
        """
        if 0 <= index < len(self.positions):
            self.positions[index] = GroupSelect(letter, colour)
            self._render_bar()

    def _render_bar(self) -> None:
        """Generate the Rich markup for the assignment bar."""
        if not self.positions:
            self.update("")
            return

        parts = []
        for i, pos in enumerate(self.positions):
            colour = "dim" if pos is None else pos.colour

            # Add the character (letter if new block, else ■)
            if pos is None or (i > 0 and self.positions[i - 1] == pos):
                char = "•"
            else:
                char = pos.letter

            parts.append(f"[{colour}]{char}[/{colour}]")

        self.update("".join(parts))
