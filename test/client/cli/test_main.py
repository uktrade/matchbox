"""Smoke test for CLI entry point."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from matchbox.client.cli.main import app
from matchbox.common.dtos import AuthStatusResponse, OKMessage, User


class TestMainCLI:
    """Test core and miscellanious CLI commands."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_entry_point_loads(self) -> None:
        """Ensure the app can at least start and show version."""
        result = self.runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "matchbox" in result.stdout.lower()

    @patch("matchbox.client._handler.healthcheck")
    def test_healthcheck(self, mock_healthcheck: MagicMock) -> None:
        """Test the healthcheck command works."""
        test_value = OKMessage(status="OK", version="0.0.0.dev0")
        mock_healthcheck.return_value = test_value
        result = self.runner.invoke(app, ["health"])
        assert result.exit_code == 0

        assert OKMessage.model_validate_json(result.stdout) == test_value

    @patch("matchbox.client._handler.auth_status")
    def test_auth_status(self, mock_auth: MagicMock) -> None:
        """Test the auth status command works."""
        test_value = AuthStatusResponse(
            authenticated=True,
            user=User(
                sub="alice",
                email="test@example.com",
            ),
        )
        mock_auth.return_value = test_value

        result = self.runner.invoke(app, ["auth", "status"])
        assert result.exit_code == 0

        assert AuthStatusResponse.model_validate_json(result.stdout) == test_value
