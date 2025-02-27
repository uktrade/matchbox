"""Module to load client settings from env file"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from matchbox.common.exceptions import MatchboxClientSettingsException


class Settings(BaseSettings):
    api_root: str
    timeout: float | None = None
    retry_delay: int = 5
    default_warehouse: str | None = None

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="MB__CLIENT__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )


try:
    settings = Settings()
except ValueError as e:
    raise MatchboxClientSettingsException from e
