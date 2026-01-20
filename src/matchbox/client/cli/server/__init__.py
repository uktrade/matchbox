"""Matchbox server CLI commands."""

from matchbox.client.cli.server import admin, main

app = main.app
app.add_typer(admin.app, name="admin")
