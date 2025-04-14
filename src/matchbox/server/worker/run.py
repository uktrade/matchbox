"""Worker for the Matchbox server.

Used to run tasks in the background.
"""

from procrastinate import App, PsycopgConnector

from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)
from matchbox.server.postgresql import MatchboxPostgresSettings


def get_backend() -> MatchboxDBAdapter:
    """Get the backend adapter with injected settings."""
    base_settings = MatchboxServerSettings()
    SettingsClass = get_backend_settings(base_settings.backend_type)
    return settings_to_backend(SettingsClass())


postgres_settings: MatchboxPostgresSettings = MatchboxPostgresSettings()
backend: MatchboxDBAdapter = get_backend()

app = App(
    connector=PsycopgConnector(
        kwargs={
            "host": postgres_settings.postgres.host,
            "port": postgres_settings.postgres.port,
            "user": postgres_settings.postgres.user,
            "password": postgres_settings.postgres.password,
            "database": postgres_settings.postgres.database,
        }
    )
)


@app.task(name="index")
def index():
    """Index a dataset."""
    pass


@app.task(name="index")
def results():
    """Process results from a model and insert."""
    pass


if __name__ == "__main__":
    pass
