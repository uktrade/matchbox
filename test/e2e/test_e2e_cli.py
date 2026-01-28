"""Integration tests for the CLI."""

from collections.abc import Generator

import pytest
from httpx import Client
from typer.testing import CliRunner

from matchbox.client.cli.main import app
from matchbox.common.dtos import AuthStatusResponse, LoginResponse, PermissionType

runner = CliRunner()


@pytest.mark.docker
class TestE2ECLI:
    """End-to-end tests for the Matchbox CLI commands."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_client(self, matchbox_client: Client) -> Generator[None, None, None]:
        """Patch the CLI client with the authenticated test client."""
        # Clear database before starting to ensure clean state
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200
        yield

    def test_basic_commands(self) -> None:
        """Test version, health, and auth status commands."""
        # Version
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Matchbox version:" in result.stdout

        # Health
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        assert "OK" in result.stdout

        # Login
        result = runner.invoke(app, ["auth", "login"])
        assert result.exit_code == 0
        login_response = LoginResponse.model_validate_json(result.stdout)
        assert login_response.setup_mode_admin

        # Auth status
        result = runner.invoke(app, ["auth", "status"])
        assert result.exit_code == 0
        auth_response = AuthStatusResponse.model_validate_json(result.stdout)
        assert auth_response.authenticated

    def test_group_management_flow(self) -> None:
        """Test the lifecycle of group management."""
        # Login
        result = runner.invoke(app, ["auth", "login"])
        assert result.exit_code == 0

        login_response = LoginResponse.model_validate_json(result.stdout)
        assert login_response.setup_mode_admin

        # Test constants
        group_name: str = "analysts"
        user_name: str = login_response.user.user_name

        # 1. Create group
        result = runner.invoke(app, ["groups", "create", "-g", group_name])
        assert result.exit_code == 0
        # Output format: ✓ Group <name>
        assert f"✓ Created group {group_name}" in result.stdout

        # 2. List groups
        result = runner.invoke(app, ["groups"])
        assert result.exit_code == 0
        assert group_name in result.stdout

        # 3. Add member
        result = runner.invoke(
            app, ["groups", "add", "-g", group_name, "-u", user_name]
        )
        assert result.exit_code == 0
        assert f"Added {user_name} to {group_name}" in result.stdout

        # 4. Show group
        result = runner.invoke(app, ["groups", "show", "-g", group_name])
        assert result.exit_code == 0
        assert group_name in result.stdout
        assert user_name in result.stdout

        # 5. Remove rember
        result = runner.invoke(
            app, ["groups", "remove", "-g", group_name, "-u", user_name]
        )
        assert result.exit_code == 0
        assert f"Removed {user_name} from {group_name}" in result.stdout

        # 6. Delete group
        result = runner.invoke(app, ["groups", "delete", "-g", group_name, "--certain"])
        assert result.exit_code == 0
        assert f"✓ Deleted group {group_name}" in result.stdout

    def test_collection_management_flow(self) -> None:
        """Test the lifecycle of collection and permission management."""
        collection_name = "companies"
        group_name = "auditors"

        # Setup: Create a group to grant permissions to
        runner.invoke(app, ["groups", "create", "-g", group_name])

        # 1. Create collection
        result = runner.invoke(app, ["collections", "create", "-c", collection_name])
        assert result.exit_code == 0
        assert f"Created collection {collection_name}" in result.stdout

        # 2. List collections
        result = runner.invoke(app, ["collections"])
        assert result.exit_code == 0
        assert collection_name in result.stdout

        # 3. Grant permission
        result = runner.invoke(
            app,
            [
                "collections",
                "grant",
                "-c",
                collection_name,
                "-g",
                group_name,
                "-p",
                PermissionType.READ.value,
            ],
        )
        assert result.exit_code == 0
        assert f"Granted {PermissionType.READ} on {collection_name}" in result.stdout

        # 4. List permissions
        result = runner.invoke(
            app, ["collections", "permissions", "-c", collection_name]
        )
        assert result.exit_code == 0
        assert group_name in result.stdout
        assert PermissionType.READ in result.stdout

        # 5. Revoke permission
        result = runner.invoke(
            app,
            [
                "collections",
                "revoke",
                "-c",
                collection_name,
                "-g",
                group_name,
                "-p",
                PermissionType.READ.value,
            ],
        )
        assert result.exit_code == 0
        assert f"Revoked {PermissionType.READ} on {collection_name}" in result.stdout

    def test_admin_commands(self) -> None:
        """Test system administration commands."""
        # Prune
        result = runner.invoke(app, ["admin", "prune"])
        assert result.exit_code == 0
        assert "success" in result.stdout
