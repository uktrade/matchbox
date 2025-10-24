"""Minimal widget tests - verify rendering only."""

import polars as pl
import pytest
from rich.table import Table

from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable
from matchbox.client.eval import EvaluationItem


class TestComparisonDisplayTable:
    """Test table widget renders correctly."""

    @pytest.fixture
    def mock_item(self) -> EvaluationItem:
        """Create a mock evaluation item."""
        df = pl.DataFrame({"leaf": [1, 2, 3]})

        item = EvaluationItem(
            cluster_id=1,
            dataframe=df,
            display_data={
                "name": ["Company A", "Company B", ""],
                "address": ["", "", "123 Main St"],
            },
            duplicate_groups=[[1], [2], [3]],
            display_columns=[1, 2, 3],
            assignments={},
        )
        return item

    def test_table_renders(self, mock_item: EvaluationItem) -> None:
        """Test that table can render."""
        table = ComparisonDisplayTable()
        table.load_comparison(mock_item)

        result = table.render()

        assert isinstance(result, Table)

    def test_table_renders_with_assignments(self, mock_item: EvaluationItem) -> None:
        """Test that table renders with column assignments."""
        mock_item.assignments = {0: "a", 2: "b"}

        table = ComparisonDisplayTable()
        table.load_comparison(mock_item)

        result = table.render()

        assert isinstance(result, Table)
        assert len(result.columns) == 4  # Field + 3 display columns

    def test_table_handles_no_item(self) -> None:
        """Test that table handles no current item."""
        table = ComparisonDisplayTable()

        result = table.render()

        assert isinstance(result, Table)
