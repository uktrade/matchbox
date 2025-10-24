"""Tests for CLI main entry point."""

import re

from typer.testing import CliRunner

from matchbox.client.cli.main import app


class TestMainCLI:
    """Test the main CLI entry point."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_help_command(self) -> None:
        """Test that help command works."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # With a single command, Typer shows that command's help directly
        assert "interactive entity resolution" in result.output.lower()

    def test_health_start_help(self) -> None:
        """Test that eval help works."""
        result = self.runner.invoke(app, ["health", "--help"])
        assert result.exit_code == 0

        # Strip ANSI codes for reliable text matching
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

        assert "checks the health" in clean_output.lower()

    def test_eval_start_help(self) -> None:
        """Test that eval help works."""
        result = self.runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0

        # Strip ANSI codes for reliable text matching
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

        assert "interactive entity resolution" in clean_output.lower()
        assert "--collection" in clean_output
        assert "--resolution" in clean_output
