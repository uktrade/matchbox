"""Base classes and utilities for Matchbox database adapters."""

import json
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import boto3
from botocore.exceptions import ClientError
from pyarrow import Table
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from matchbox.common.dtos import ModelAncestor, ModelMetadata
from matchbox.common.graph import ResolutionGraph
from matchbox.common.logging import LogLevelType
from matchbox.common.sources import Match, Source, SourceAddress

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


class MatchboxBackends(StrEnum):
    """The available backends for Matchbox."""

    POSTGRES = "postgres"


class MatchboxSnapshot(BaseModel):
    """A snapshot of the Matchbox database."""

    backend_type: MatchboxBackends
    data: Any

    @field_validator("data")
    @classmethod
    def check_serialisable(cls, value: Any) -> Any:
        """Validate that the value can be serialised to JSON."""
        try:
            json.dumps(value)
            return value
        except (TypeError, OverflowError) as e:
            raise ValueError(f"Value is not JSON serialisable: {e}") from e


class MatchboxDatastoreSettings(BaseSettings):
    """Settings specific to the datastore configuration."""

    host: str | None = None
    port: int | None = None
    access_key_id: SecretStr | None = None
    secret_access_key: SecretStr | None = None
    default_region: str | None = None
    cache_bucket_name: str

    def get_client(self) -> S3Client:
        """Returns an S3 client for the datastore.

        Creates S3 buckets if they don't exist.
        """
        kwargs = {
            "endpoint_url": f"http://{self.host}:{self.port}"
            if self.host and self.port
            else None,
            "aws_access_key_id": self.access_key_id.get_secret_value()
            if self.access_key_id
            else None,
            "aws_secret_access_key": self.secret_access_key.get_secret_value()
            if self.secret_access_key
            else None,
            "region_name": self.default_region,
        }

        client: S3Client = boto3.client("s3", **kwargs)

        try:
            client.head_bucket(Bucket=self.cache_bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                client.create_bucket(
                    Bucket=self.cache_bucket_name,
                    CreateBucketConfiguration={
                        "LocationConstraint": self.default_region
                    },
                )
            else:
                raise e

        return client


class MatchboxServerSettings(BaseSettings):
    """Settings for the Matchbox application."""

    model_config = SettingsConfigDict(
        env_prefix="MB__SERVER__",
        env_nested_delimiter="__",
        use_enum_values=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    batch_size: int = Field(default=250_000)
    backend_type: MatchboxBackends
    datastore: MatchboxDatastoreSettings
    api_key: SecretStr | None = Field(default=None)
    log_level: LogLevelType = "INFO"


class BackendManager:
    """Manages the Matchbox backend instance and settings."""

    _instance = None
    _settings = None

    @classmethod
    def initialise(cls, settings: "MatchboxServerSettings"):
        """Initialise the backend with the given settings."""
        cls._settings = settings

    @classmethod
    def get_backend(cls) -> "MatchboxDBAdapter":
        """Get the backend instance."""
        if cls._settings is None:
            raise ValueError("BackendManager must be initialized with settings first")

        if cls._instance is None:
            BackendClass = get_backend_class(cls._settings.backend_type)
            cls._instance = BackendClass(cls._settings)
        return cls._instance

    @classmethod
    def get_settings(cls) -> "MatchboxServerSettings":
        """Get the backend settings."""
        if cls._settings is None:
            raise ValueError("BackendManager must be initialized with settings first")
        return cls._settings


def get_backend_settings(
    backend_type: MatchboxBackends,
) -> type[MatchboxServerSettings]:
    """Get the appropriate settings class based on the backend type."""
    if backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql import MatchboxPostgresSettings

        return MatchboxPostgresSettings
    # Add more backend types here as needed
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def get_backend_class(backend_type: MatchboxBackends) -> type["MatchboxDBAdapter"]:
    """Get the appropriate backend class based on the backend type."""
    if backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql import MatchboxPostgres

        return MatchboxPostgres
    # Add more backend types here as needed
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def settings_to_backend(settings: MatchboxServerSettings) -> "MatchboxDBAdapter":
    """Create backend adapter with injected settings."""
    BackendClass = get_backend_class(settings.backend_type)
    return BackendClass(settings)


def initialise_matchbox() -> None:
    """Initialise the Matchbox backend based on environment variables."""
    base_settings = MatchboxServerSettings()

    SettingsClass = get_backend_settings(base_settings.backend_type)
    settings = SettingsClass()

    BackendManager.initialise(settings)


class Countable(Protocol):
    """A protocol for objects that can be counted."""

    def count(self) -> int:
        """Counts the number of items in the object."""
        ...


class Listable(Protocol):
    """A protocol for objects that can be listed."""

    def list_all(self) -> list[str]:
        """Lists the items in the object."""
        ...


class ListableAndCountable(Countable, Listable):
    """A protocol for objects that can be counted and listed."""

    pass


class MatchboxDBAdapter(ABC):
    """An abstract base class for Matchbox database adapters."""

    settings: "MatchboxServerSettings"

    datasets: ListableAndCountable
    models: Countable
    data: Countable
    clusters: Countable
    creates: Countable
    merges: Countable
    proposes: Countable
    source_resolutions: Countable

    # Retrieval

    @abstractmethod
    def query(
        self,
        source_address: SourceAddress,
        resolution_name: str | None = None,
        threshold: int | None = None,
        limit: int = None,
    ) -> Table:
        """Queries the database from an optional point of truth.

        Args:
            source_address: the `SourceAddress` object identifying the source to query
            resolution_name (optional): the resolution to use for filtering results
                If not specified, will use the dataset resolution for the queried source
            threshold (optional): the threshold to use for creating clusters
                If None, uses the models' default threshold
                If an integer, uses that threshold for the specified model, and the
                model's cached thresholds for its ancestors
            limit (optional): the number to use in a limit clause. Useful for testing

        Returns:
            The resulting matchbox IDs in Arrow format
        """
        ...

    @abstractmethod
    def match(
        self,
        source_pk: str,
        source: SourceAddress,
        targets: list[SourceAddress],
        resolution_name: str,
        threshold: int | None = None,
    ) -> list[Match]:
        """Matches an ID in a source dataset and returns the keys in the targets.

        Args:
            source_pk: The primary key to match from the source.
            source: The address of the source dataset.
            targets: The addresses of the target datasets.
            resolution_name: The name of the resolution to use for matching.
            threshold (optional): the threshold to use for creating clusters
                If None, uses the resolutions' default threshold
                If an integer, uses that threshold for the specified resolution, and the
                resolution's cached thresholds for its ancestors
                Will use these threshold values instead of the cached thresholds
        """
        ...

    # Data management

    @abstractmethod
    def index(self, source: Source, data_hashes: Table) -> None:
        """Indexes to Matchbox a source dataset in your warehouse.

        Args:
            source: The source dataset to index.
            data_hashes: The Arrow table with the hash of each data row
        """
        ...

    @abstractmethod
    def get_source(self, address: SourceAddress) -> Source:
        """Get a source from its address.

        Args:
            address: The name address for the source

        Returns:
            A Source object
        """
        ...

    @abstractmethod
    def get_resolution_sources(
        self,
        resolution_name: str,
    ) -> list[Source]:
        """Get a list of sources queriable from a resolution.

        Args:
            resolution_name: Name of the resolution to query.

        Returns:
            List of relevant Source objects.
        """
        ...

    @abstractmethod
    def validate_ids(self, ids: list[int]) -> bool:
        """Validates a list of IDs exist in the database.

        Args:
            ids: A list of IDs to validate.

        Raises:
            MatchboxDataNotFound: If some items don't exist in the target table.
        """
        ...

    @abstractmethod
    def validate_hashes(self, hashes: list[bytes]) -> bool:
        """Validates a list of hashes exist in the database.

        Args:
            hashes: A list of hashes to validate.

        Raises:
            MatchboxDataNotFound: If some items don't exist in the target table.
        """
        ...

    @abstractmethod
    def cluster_id_to_hash(self, ids: list[int]) -> dict[int, bytes | None]:
        """Get a lookup of Cluster hashes from a list of IDs.

        Args:
            ids: A list of IDs to get hashes for.

        Returns:
            A dictionary mapping IDs to hashes.
        """
        ...

    @abstractmethod
    def get_resolution_graph(self) -> ResolutionGraph:
        """Get the full resolution graph."""
        ...

    @abstractmethod
    def dump(self) -> MatchboxSnapshot:
        """Dumps the entire database to a snapshot.

        Returns:
            A MatchboxSnapshot object of type "postgres" with the database's
                current state.
        """
        ...

    @abstractmethod
    def drop(self, certain: bool) -> None:
        """Hard clear the database by dropping all tables and re-creating.

        Args:
            certain: Whether to drop the database without confirmation.
        """
        ...

    @abstractmethod
    def clear(self, certain: bool) -> None:
        """Soft clear the database by deleting all rows but retaining tables.

        Args:
            certain: Whether to delete the database without confirmation.
        """
        ...

    @abstractmethod
    def restore(self, snapshot: MatchboxSnapshot, clear: bool) -> None:
        """Restores the database from a snapshot.

        Args:
            snapshot: A MatchboxSnapshot object of type "postgres" with the
                database's state
            clear: Whether to clear the database before restoration

        Raises:
            TypeError: If the snapshot is not compatible with PostgreSQL
        """
        ...

    @abstractmethod
    def verify(self) -> None:
        """Checks the database schema against expected and logs outcome."""
        ...

    # Model management

    @abstractmethod
    def insert_model(self, model: ModelMetadata) -> None:
        """Writes a model to Matchbox.

        Args:
            model: ModelMetadata object with the model's metadata

        Raises:
            MatchboxDataNotFound: If, for a linker, the source models weren't found in
                the database
        """
        ...

    @abstractmethod
    def get_model(self, model: str) -> ModelMetadata:
        """Get a model from the database."""
        ...

    @abstractmethod
    def set_model_results(self, model: str, results: Table) -> None:
        """Set the results for a model."""
        ...

    @abstractmethod
    def get_model_results(self, model: str) -> Table:
        """Get the results for a model."""
        ...

    @abstractmethod
    def set_model_truth(self, model: str, truth: float) -> None:
        """Sets the truth threshold for this model, changing the default clusters."""
        ...

    @abstractmethod
    def get_model_truth(self, model: str) -> float:
        """Gets the current truth threshold for this model."""
        ...

    @abstractmethod
    def get_model_ancestors(self, model: str) -> list[ModelAncestor]:
        """Gets the current truth values of all ancestors.

        Returns a list of ModelAncestor objects mapping model names to their current
        truth thresholds.

        Unlike ancestors_cache which returns cached values, this property returns
        the current truth values of all ancestor models.
        """
        ...

    @abstractmethod
    def set_model_ancestors_cache(
        self, model: str, ancestors_cache: list[ModelAncestor]
    ) -> None:
        """Updates the cached ancestor thresholds.

        Args:
            model: The name of the model to update
            ancestors_cache: List of ModelAncestor objects mapping model names to
                their truth thresholds
        """
        ...

    @abstractmethod
    def get_model_ancestors_cache(self, model: str) -> list[ModelAncestor]:
        """Gets the cached ancestor thresholds, converting hashes to model names.

        Returns a list of ModelAncestor objects mapping model names to their cached
        truth thresholds.

        This is required because each point of truth needs to be stable, so we choose
        when to update it, caching the ancestor's values in the model itself.
        """
        ...

    @abstractmethod
    def delete_model(self, model: str, certain: bool) -> None:
        """Delete a model from the database.

        Args:
            model: The name of the model to delete.
            certain: Whether to delete the model without confirmation.
        """
        ...
