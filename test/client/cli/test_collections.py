"""Tests for Collection CLI commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from matchbox.client.cli.main import app
from matchbox.common.dtos import (
    CollectionName,
    CRUDOperation,
    PermissionGrant,
    PermissionType,
    ResourceOperationStatus,
)


class TestCollectionCLI:
    """Test the collection CLI commands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("matchbox.client._handler.list_collections")
    def test_list_collections(self, mock_list: MagicMock) -> None:
        """Test listing collections."""
        mock_list.return_value = [CollectionName("coll-1"), CollectionName("coll-2")]

        result = self.runner.invoke(app, ["collections"])

        assert result.exit_code == 0
        assert "coll-1" in result.output
        assert "coll-2" in result.output

    @patch("matchbox.client._handler.create_collection")
    def test_create_collection(self, mock_create: MagicMock) -> None:
        """Test creating a collection."""
        mock_create.return_value = ResourceOperationStatus(
            success=True,
            target="Collection new-coll",
            operation=CRUDOperation.CREATE,
        )

        # Default (public)
        result = self.runner.invoke(app, ["collections", "create", "-c", "new-coll"])
        assert result.exit_code == 0
        assert "✓ Created collection 'new-coll'" in result.output
        mock_create.assert_called_with("new-coll", admin_group="public")

        # With admin group
        result = self.runner.invoke(
            app, ["collections", "create", "-c", "secure-coll", "--group", "admins"]
        )
        assert result.exit_code == 0
        assert "✓ Created collection 'secure-coll'" in result.output
        assert "Admin permission granted to group 'admins'" in result.output
        mock_create.assert_called_with("secure-coll", admin_group="admins")

    @patch("matchbox.client._handler.get_collection_permissions")
    def test_list_permissions(self, mock_list: MagicMock) -> None:
        """Test listing collection permissions."""
        mock_list.return_value = [
            PermissionGrant(group_name="team-a", permission=PermissionType.READ)
        ]

        result = self.runner.invoke(
            app, ["collections", "permissions", "-c", "my-coll"]
        )

        assert result.exit_code == 0
        assert "team-a" in result.output
        assert "read" in result.output

    @patch("matchbox.client._handler.grant_collection_permission")
    def test_grant_permission(self, mock_grant: MagicMock) -> None:
        """Test granting permission."""
        mock_grant.return_value = ResourceOperationStatus(
            success=True,
            target="read on my-coll for team-a",
            operation=CRUDOperation.CREATE,
        )

        result = self.runner.invoke(
            app,
            [
                "collections",
                "grant",
                "-c",
                "my-coll",
                "-g",
                "team-a",
                "-p",
                "read",
            ],
        )

        assert result.exit_code == 0
        assert "✓ Granted read on 'my-coll' to 'team-a'" in result.output
        mock_grant.assert_called_with(
            collection="my-coll",
            group_name="team-a",
            permission=PermissionType.READ,
        )

    @patch("matchbox.client._handler.revoke_collection_permission")
    def test_revoke_permission(self, mock_revoke: MagicMock) -> None:
        """Test revoking permission."""
        mock_revoke.return_value = ResourceOperationStatus(
            success=True,
            target="read on my-coll for team-a",
            operation=CRUDOperation.DELETE,
        )

        result = self.runner.invoke(
            app,
            [
                "collections",
                "revoke",
                "-cmy-coll",
                "-g",
                "team-a",
                "-p",
                "read",
            ],
        )

        assert result.exit_code == 0
        assert "✓ Revoked read on 'my-coll' from 'team-a'" in result.output
        mock_revoke.assert_called_with(
            collection="my-coll",
            group_name="team-a",
            permission=PermissionType.READ,
        )
