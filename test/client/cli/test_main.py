"""Tests for CLI main entry point."""

import re

from typer.testing import CliRunner

from matchbox.client.cli.main import app


class TestMainCLI:
    """Test the main CLI entry point."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_app_exists(self) -> None:
        """Test that the main CLI app can be imported."""
        assert app is not None

    def test_help_command(self) -> None:
        """Test that help command works."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Matchbox: Entity resolution" in result.output

    def test_eval_help_command(self) -> None:
        """Test that eval help command works."""
        result = self.runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "Entity evaluation and manual labelling tools" in result.output

    def test_eval_start_help(self) -> None:
        """Test that eval start help works."""
        result = self.runner.invoke(app, ["eval", "start", "--help"])
        assert result.exit_code == 0

        # Strip ANSI codes for reliable text matching
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

        assert "interactive entity resolution" in clean_output.lower()
        assert "--resolution" in clean_output
        assert "--samples" in clean_output
