"""Smoke test for CLI entry point."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from matchbox.client.cli.main import app, run
from matchbox.common.dtos import AuthStatusResponse, OKMessage, User
from matchbox.common.exceptions import (
    MatchboxSystemGroupError,
    MatchboxUnhandledServerResponse,
)


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
                user_name="alice",
                email="test@example.com",
            ),
        )
        mock_auth.return_value = test_value

        result = self.runner.invoke(app, ["auth", "status"])
        assert result.exit_code == 0

        assert AuthStatusResponse.model_validate_json(result.stdout) == test_value

    @patch("matchbox.client._handler.delete_group")
    @patch("sys.stderr", new_callable=StringIO)
    def test_graceful_error_handling(
        self, mock_stderr: StringIO, mock_delete: MagicMock
    ) -> None:
        """Ensure MatchboxHttpExceptions are displayed without stack traces.

        This test uses the run() wrapper rather than invoking the typer app directly,
        because run() is the actual entry point used by the installed console script.

        The sys.argv patch simulates command-line arguments. When a user runs:

            mbx groups delete -g public --certain

        Python sets sys.argv to ['mbx', 'groups', 'delete', '-g', 'public',
        '--certain'].

        By patching sys.argv, we simulate this command without actually running it
        from the command line, allowing the Typer app to parse these arguments normally.
        """
        mock_delete.side_effect = MatchboxSystemGroupError(
            message="Cannot delete system group 'public'"
        )

        # Mock sys.argv to simulate CLI arguments
        with (
            patch("sys.argv", ["mbx", "groups", "delete", "-g", "public", "--certain"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            run()

        assert exc_info.value.code == 1

        stderr_output = mock_stderr.getvalue()

        # 1. Ensure clean message
        assert "Error:" in stderr_output
        assert "Cannot delete system group 'public'" in stderr_output

        # 2. Ensure no Python traceback
        assert "Traceback (most recent call last)" not in stderr_output

    @patch("matchbox.client._handler.delete_group")
    @patch("sys.stderr", new_callable=StringIO)
    def test_non_http_exceptions_show_traceback(
        self, mock_stderr: StringIO, mock_delete: MagicMock
    ) -> None:
        """Ensure non-HTTP exceptions still show full tracebacks.

        Only MatchboxHttpException and its subclasses should be caught and
        displayed cleanly. Other exceptions should propagate normally with
        full tracebacks for debugging purposes.
        """
        # Raise a non-HTTP exception
        mock_delete.side_effect = ValueError("Something went wrong internally")

        with (
            patch("sys.argv", ["mbx", "groups", "delete", "-g", "public", "--certain"]),
            pytest.raises(ValueError, match="Something went wrong internally"),
        ):
            # This should raise ValueError, not SystemExit
            run()

    @patch("matchbox.client._handler.delete_group")
    def test_server_error_shows_traceback(self, mock_delete: MagicMock) -> None:
        """Ensure unhandled server errors (500s) show a full traceback.

        If the server returns a 500 or an exception not in the HTTP registry (like
        MatchboxConnectionError), the client raises MatchboxUnhandledServerResponse.

        This is NOT a MatchboxHttpException, so it should bypass the clean error
        handling and show a traceback.
        """
        # Simulate a 500 Internal Server Error from the API
        # The client handler converts unknown server errors into this exception
        mock_delete.side_effect = MatchboxUnhandledServerResponse(
            http_status=500, details="MatchboxConnectionError: Database unavailable"
        )

        with (
            patch("sys.argv", ["mbx", "groups", "delete", "-g", "public", "--certain"]),
            pytest.raises(MatchboxUnhandledServerResponse) as exc_info,
        ):
            run()

        assert "MatchboxConnectionError" in str(exc_info.value)
