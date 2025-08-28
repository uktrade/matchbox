"""Tests for CLI main entry point."""

from typer.testing import CliRunner

from matchbox.client.cli.main import app


class TestMainCLI:
    """Test the main CLI entry point."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_app_exists(self):
        """Test that the main CLI app can be imported."""
        assert app is not None

    def test_help_command(self):
        """Test that help command works."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Matchbox: Entity resolution" in result.output

    def test_eval_help_command(self):
        """Test that eval help command works."""
        result = self.runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "Entity evaluation and manual labeling tools" in result.output

    def test_eval_start_help(self):
        """Test that eval start help works."""
        result = self.runner.invoke(app, ["eval", "start", "--help"])
        assert result.exit_code == 0

        # Strip ANSI codes for reliable text matching
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

        assert "interactive entity resolution" in clean_output.lower()
        assert "--resolution" in clean_output
        assert "--samples" in clean_output
