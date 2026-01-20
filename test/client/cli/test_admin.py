"""Tests for admin CLI commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from matchbox.client.cli.main import app
from matchbox.common.dtos import CRUDOperation, ResourceOperationStatus


class TestAdminCLI:
    """Test the admin CLI commands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("matchbox.client._handler.delete_orphans")
    def test_prune_orphans(self, mock_prune: MagicMock) -> None:
        """Test pruning orphan clusters."""
        mock_prune.return_value = ResourceOperationStatus(
            success=True,
            target="Database",
            operation=CRUDOperation.DELETE,
            details="Deleted 5 orphans",
        )

        result = self.runner.invoke(app, ["admin", "prune"])

        assert result.exit_code == 0
        assert "Deleted 5 orphans" in result.output
        mock_prune.assert_called_once()
