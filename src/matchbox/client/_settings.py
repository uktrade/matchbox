"""Module to load client settings from env file."""

import base64
import json
from datetime import UTC, datetime

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from matchbox.common.exceptions import MatchboxClientSettingsException


class ClientSettings(BaseSettings):
    api_root: str
    timeout: float | None = None
    retry_delay: float = 15.0
    default_warehouse: str | None = None
    jwt: str | None = None
    user: str | None = None
    batch_size: int = Field(default=10_000)

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="MB__CLIENT__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @field_validator("jwt")
    @classmethod
    def validate_jwt_structure(cls, v: str | None) -> str:
        """Perform basic JWT validation."""
        if v is None:
            return v

        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("JWT must have exactly 3 parts")

        try:
            # Decode header and payload
            header = json.loads(base64.urlsafe_b64decode(parts[0] + "=="))
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
        except Exception as e:  # noqa: BLE001
            raise ValueError("Invalid JWT.") from e

        # Basic header checks
        if "alg" not in header or header["alg"] == "none":
            raise ValueError("Invalid or missing algorithm")

        # Time-based validation
        now = datetime.now(UTC).timestamp()

        if "exp" in payload and payload["exp"] < now:
            raise ValueError("Token has expired")

        if "nbf" in payload and payload["nbf"] > now:
            raise ValueError("Token not yet valid")

        return v


try:
    settings = ClientSettings()
except ValueError as e:
    raise MatchboxClientSettingsException from e
