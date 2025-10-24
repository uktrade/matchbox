"""Comparison display table for entity resolution evaluation."""

from typing import Any

from rich import box
from rich.table import Table
from textual.widget import Widget

from matchbox.client.cli.eval.widgets.styling import get_group_style
from matchbox.client.eval import EvaluationItem


class ComparisonDisplayTable(Widget):
    """Table for side-by-side record comparison with colour highlighting."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.current_item: EvaluationItem | None = None

    def load_comparison(self, item: EvaluationItem) -> None:
        """Load new comparison data.

        Args:
            item: The evaluation item to display
        """
        self.current_item = item
        self.refresh()

    def render(self) -> Table:
        """Render the table with current item."""
        if not self.current_item:
            return self._create_loading_table()

        return self._render_compact_view(self.current_item)

    def _create_loading_table(self) -> Table:
        """Create a simple loading table."""
        loading_table = Table(show_header=False, show_lines=False)
        loading_table.add_column("")
        loading_table.add_row("Loading...")
        return loading_table

    def _add_table_columns(self, table: Table, current: EvaluationItem) -> None:
        """Add styled columns to the table for each display column."""
        # First column for field names with distinct styling
        table.add_column(
            "",  # No header
            style="bright_white",
            min_width=20,
            max_width=50,
        )

        # Data columns
        for display_col_index, _ in enumerate(current.display_columns):
            col_num = display_col_index + 1
            duplicate_count = len(current.duplicate_groups[display_col_index])

            header_text = (
                f"{col_num} (×{duplicate_count})"
                if duplicate_count > 1
                else str(col_num)
            )

            if display_col_index in current.assignments:
                group = current.assignments[display_col_index]
                symbol, colour = get_group_style(group)
                header = f"[{colour} bold]{symbol} {header_text}[/]"
                table.add_column(
                    header,
                    style=colour,
                    min_width=15,
                    max_width=50,
                )
            else:
                table.add_column(
                    f"[dim]{header_text}[/]",
                    style="dim",
                    min_width=15,
                    max_width=50,
                )

    def _render_compact_view(self, current: EvaluationItem) -> Table:
        """Render compact view with improved styling."""
        if not current.display_data:
            return self._create_loading_table()

        table = Table(
            show_header=True,
            header_style="bold bright_white",
            show_lines=False,
            row_styles=["", "on #0a1628"],
            box=box.SIMPLE_HEAD,
            padding=(0, 1),
            expand=True,
            border_style="#E4A242",
        )

        self._add_table_columns(table, current)

        for field_name, values in current.display_data.items():
            table.add_row(field_name, *values)

        return table
