"""Smoke test for CLI entry point."""

from typer.testing import CliRunner

from matchbox.client.cli.main import app


def test_entry_point_loads() -> None:
    """Ensure the app can at least start and show version."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "matchbox" in result.stdout.lower()
