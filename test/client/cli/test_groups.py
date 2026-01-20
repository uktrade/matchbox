"""Tests for group CLI commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from matchbox.client.cli.main import app
from matchbox.common.dtos import (
    CRUDOperation,
    Group,
    ResourceOperationStatus,
    User,
)


class TestGroupsCLI:
    """Test the groups CLI commands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("matchbox.client._handler.list_groups")
    def test_list_groups(self, mock_list: MagicMock) -> None:
        """Test listing groups."""
        mock_list.return_value = [
            Group(name="admins", description="System admins", is_system=True),
            Group(name="devs", description="Developers", is_system=False),
        ]

        result = self.runner.invoke(app, ["groups"])

        assert result.exit_code == 0
        assert "admins" in result.output
        assert "System admins" in result.output
        assert "✓" in result.output  # System checkmark
        assert "devs" in result.output

    @patch("matchbox.client._handler.create_group")
    def test_create_group(self, mock_create: MagicMock) -> None:
        """Test creating a group."""
        mock_create.return_value = ResourceOperationStatus(
            success=True,
            target="Group new-group",
            operation=CRUDOperation.CREATE,
        )

        result = self.runner.invoke(
            app, ["groups", "create", "-g", "new-group", "-d", "A new group"]
        )

        assert result.exit_code == 0
        assert "✓ Group new-group" in result.output
        mock_create.assert_called_with(name="new-group", description="A new group")

    @patch("matchbox.client._handler.delete_group")
    def test_delete_group(self, mock_delete: MagicMock) -> None:
        """Test deleting a group."""
        mock_delete.return_value = ResourceOperationStatus(
            success=True,
            target="Group old-group",
            operation=CRUDOperation.DELETE,
        )

        result = self.runner.invoke(
            app, ["groups", "delete", "-g", "old-group", "--certain"]
        )

        assert result.exit_code == 0
        assert "✓ Group old-group" in result.output
        mock_delete.assert_called_with(name="old-group", certain=True)

    @patch("matchbox.client._handler.get_group")
    def test_show_group(self, mock_get: MagicMock) -> None:
        """Test showing group details."""
        mock_get.return_value = Group(
            name="team-a",
            description="Team A",
            members=[User(sub="alice", email=None)],
        )

        result = self.runner.invoke(app, ["groups", "show", "-g", "team-a"])

        assert result.exit_code == 0
        assert "team-a" in result.output
        assert "Team A" in result.output
        assert "alice" in result.output

    @patch("matchbox.client._handler.add_user_to_group")
    def test_add_member(self, mock_add: MagicMock) -> None:
        """Test adding a member."""
        result = self.runner.invoke(app, ["groups", "add", "-g", "team-a", "-u", "bob"])

        assert result.exit_code == 0
        assert "✓ Added bob to team-a" in result.output
        mock_add.assert_called_with(group_name="team-a", user_name="bob")

    @patch("matchbox.client._handler.remove_user_from_group")
    def test_remove_member(self, mock_remove: MagicMock) -> None:
        """Test removing a member."""
        result = self.runner.invoke(
            app, ["groups", "remove", "-g", "team-a", "-u", "bob"]
        )

        assert result.exit_code == 0
        assert "✓ Removed bob from team-a" in result.output
        mock_remove.assert_called_with(group_name="team-a", user_name="bob")
