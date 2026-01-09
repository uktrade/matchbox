"""Base classes and utilities for Matchbox database adapters."""

import json
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

import boto3
from botocore.exceptions import ClientError
from pyarrow import Table
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from matchbox.common.dtos import (
    BackendResourceType,
    Collection,
    CollectionName,
    Group,
    GroupName,
    Match,
    ModelResolutionPath,
    PermissionGrant,
    PermissionType,
    Resolution,
    ResolutionPath,
    Run,
    RunID,
    SourceResolutionPath,
    UploadStage,
    User,
)
from matchbox.common.eval import Judgement
from matchbox.common.logging import LogLevelType

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


PERMISSION_GRANTS: dict[PermissionType, list[PermissionType]] = {
    PermissionType.READ: [
        PermissionType.READ,
        PermissionType.WRITE,
        PermissionType.ADMIN,
    ],
    PermissionType.WRITE: [PermissionType.WRITE, PermissionType.ADMIN],
    PermissionType.ADMIN: [PermissionType.ADMIN],
}
"""A global variable that defines the permission hierarchy.

Keys are the permission, values are a list of permissions that would
grant the permission.

For example, only `PermissionType.ADMIN` can grant `PermissionType.ADMIN`,
but any permission implies `PermissionType.READ`.
"""


class MatchboxBackends(StrEnum):
    """The available backends for Matchbox."""

    POSTGRES = "postgres"


class MatchboxSnapshot(BaseModel):
    """A snapshot of the Matchbox database."""

    backend_type: MatchboxBackends
    data: dict[str, Any]

    @field_validator("data")
    @classmethod
    def check_serialisable(cls, value: dict[str, Any]) -> dict[str, Any]:
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
    task_runner: Literal["api", "celery"]
    redis_uri: str | None
    uploads_expiry_minutes: int | None
    authorisation: bool = True
    public_key: SecretStr | None = Field(default=None)
    log_level: LogLevelType = "INFO"

    @model_validator(mode="after")
    def check_settings(self) -> Self:
        """Check that legal combinations of settings are provided."""
        if self.task_runner == "celery" and self.redis_uri is None:
            raise ValueError("A Redis URI must be set if using Celery as task runner.")
        if self.task_runner == "celery" and self.uploads_expiry_minutes is None:
            raise ValueError(
                "Upload expiration must be set if using Celery as task runner."
            )

        return self


class BackendManager:
    """Manages the Matchbox backend instance and settings."""

    _instance = None
    _settings = None

    @classmethod
    def initialise(cls, settings: "MatchboxServerSettings") -> None:
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
        from matchbox.server.postgresql import MatchboxPostgresSettings  # noqa: PLC0415

        return MatchboxPostgresSettings
    # Add more backend types here as needed
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def get_backend_class(backend_type: MatchboxBackends) -> type["MatchboxDBAdapter"]:
    """Get the appropriate backend class based on the backend type."""
    if backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql import MatchboxPostgres  # noqa: PLC0415

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

    sources: ListableAndCountable
    models: Countable
    source_clusters: Countable
    model_clusters: Countable
    all_clusters: Countable
    creates: Countable
    merges: Countable
    proposes: Countable
    source_resolutions: Countable

    # Retrieval

    @abstractmethod
    def query(
        self,
        source: SourceResolutionPath,
        point_of_truth: ResolutionPath | None = None,
        threshold: int | None = None,
        return_leaf_id: bool = False,
        limit: int | None = None,
    ) -> Table:
        """Queries the database from an optional point of truth.

        Args:
            source: the resolution pathidentifying the source to query
            point_of_truth (optional): the resolution path to use for filtering results
                If not specified, will use the source resolution for the queried source
            threshold (optional): the threshold to use for creating clusters
                If None, uses the models' default threshold
                If an integer, uses that threshold for the specified model, and the
                model's cached thresholds for its ancestors
            return_leaf_id (optional): whether to return cluster ID of leaves
            limit (optional): the number to use in a limit clause. Useful for testing

        Returns:
            The resulting matchbox IDs in Arrow format
        """
        ...

    @abstractmethod
    def match(
        self,
        key: str,
        source: SourceResolutionPath,
        targets: list[SourceResolutionPath],
        point_of_truth: ResolutionPath,
        threshold: int | None = None,
    ) -> list[Match]:
        """Matches an ID in a source resolution and returns the keys in the targets.

        Args:
            key: The key to match from the source.
            source: The path of the source resolution.
            targets: The paths of the target source resolutions.
            point_of_truth: The path of the resolution to use for matching.
            threshold (optional): the threshold to use for creating clusters
                If None, uses the resolutions' default threshold
                If an integer, uses that threshold for the specified resolution, and the
                resolution's cached thresholds for its ancestors
                Will use these threshold values instead of the cached thresholds
        """
        ...

    # Collection management

    @abstractmethod
    def create_collection(self, name: CollectionName) -> Collection:
        """Create a new collection.

        Args:
            name: The name of the collection to create.

        Returns:
            A Collection object containing its metadata, versions, and resolutions.
        """

    @abstractmethod
    def get_collection(self, name: CollectionName) -> Collection:
        """Get collection metadata.

        Args:
            name: The name of the collection to get.

        Returns:
            A Collection object containing its metadata, versions, and resolutions.
        """
        ...

    @abstractmethod
    def list_collections(self) -> list[CollectionName]:
        """List all collection names.

        Returns:
            A list of collection names.
        """
        ...

    @abstractmethod
    def delete_collection(self, name: CollectionName, certain: bool) -> None:
        """Delete a collection and all its versions.

        Args:
            name: The name of the collection to delete.
            certain: Whether to delete the collection without confirmation.
        """
        ...

    # Version management

    @abstractmethod
    def create_run(self, collection: CollectionName) -> Run:
        """Create a new run.

        Args:
            collection: The name of the collection to create the run in.

        Returns:
            A Run object containing its metadata and resolutions.
        """
        ...

    @abstractmethod
    def set_run_mutable(
        self, collection: CollectionName, run_id: RunID, mutable: bool
    ) -> Run:
        """Set the mutability of a run.

        Args:
            collection: The name of the collection containing the run.
            run_id: The ID of the run to update.
            mutable: Whether the run should be mutable.

        Returns:
            The updated Run object.
        """
        ...

    @abstractmethod
    def set_run_default(
        self, collection: CollectionName, run_id: RunID, default: bool
    ) -> Run:
        """Set the default status of a run.

        Args:
            collection: The name of the collection containing the run.
            run_id: The ID of the run to update.
            default: Whether the run should be the default run.

        Returns:
            The updated Run object.
        """
        ...

    @abstractmethod
    def get_run(self, collection: CollectionName, run_id: RunID) -> Run:
        """Get run metadata and resolutions.

        Args:
            collection: The name of the collection containing the run.
            run_id: The ID of the run to get.

        Returns:
            A Run object containing its metadata and resolutions.
        """
        ...

    @abstractmethod
    def delete_run(
        self, collection: CollectionName, run_id: RunID, certain: bool
    ) -> None:
        """Delete a run and all its resolutions.

        Args:
            collection: The name of the collection containing the run.
            run_id: The ID of the run to delete.
            certain: Whether to delete the run without confirmation.
        """
        ...

    # Resolution management

    @abstractmethod
    def create_resolution(self, resolution: Resolution, path: ResolutionPath) -> None:
        """Writes a resolution to Matchbox.

        Args:
            resolution: Resolution object with a source or model config
            path: The resolution path for the source
        """
        ...

    @abstractmethod
    def get_resolution(self, path: ResolutionPath) -> Resolution:
        """Get a resolution from its path.

        Args:
            path: The resolution path for the source

        Returns:
            A Resolution object
        """
        ...

    @abstractmethod
    def update_resolution(self, resolution: Resolution, path: ResolutionPath) -> None:
        """Updates resolution metadata.

        It cannot be used to update a resolution's fingerprint.

        Args:
            resolution: Resolution object with a source or model config
            path: The resolution path for the source
        """
        ...

    @abstractmethod
    def delete_resolution(self, path: ResolutionPath, certain: bool) -> None:
        """Delete a resolution from the database.

        Args:
            path: The path of the resolution to delete.
            certain: Whether to delete the model without confirmation.
        """
        ...

    # Data insertion

    @abstractmethod
    def lock_resolution_data(self, path: ResolutionPath) -> None:
        """Change resolution upload stage to PROCESSING.

        This will lock uploading data.

        Args:
            path: The path of the resolution to target.
        """
        ...

    @abstractmethod
    def unlock_resolution_data(self, path: ResolutionPath, complete: bool) -> None:
        """Change resolution upload stage to READY.

        This will unlock uploading data.

        Args:
            path: The path of the resolution to target.
            complete: Whether to label the resolution stage as COMPLETE.
        """
        ...

    @abstractmethod
    def get_resolution_stage(self, path: ResolutionPath) -> UploadStage:
        """Retrieves upload stage of resolution data.

        Args:
            path: The path of the resolution to target.
        """
        ...

    @abstractmethod
    def insert_source_data(
        self, path: SourceResolutionPath, data_hashes: Table
    ) -> None:
        """Inserts hash data for a source resolution.

        Only possible if data fingerprint matches fingerprint declared when the
        resolution was created. Data can only be set once on a resolution.

        Args:
            path: The path of the source resolution to index.
            data_hashes: The Arrow table with the hash of each data row
        """
        ...

    @abstractmethod
    def insert_model_data(self, path: ModelResolutionPath, results: Table) -> None:
        """Inserts results data for a model resolution.

        Only possible if data fingerprint matches fingerprint declared when the
        resolution was created. Data can only be set once on a resolution.
        """
        ...

    @abstractmethod
    def get_model_data(self, path: ModelResolutionPath) -> Table:
        """Get the results for a model resolution."""
        ...

    # Data management

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
    def restore(self, snapshot: MatchboxSnapshot) -> None:
        """Restores the database from a snapshot.

        Args:
            snapshot: A MatchboxSnapshot object of type "postgres" with the
                database's state

        Raises:
            TypeError: If the snapshot is not compatible with PostgreSQL
        """
        ...

    @abstractmethod
    def delete_orphans(self) -> int:
        """Deletes orphan clusters.

        Orphan clusters are clusters recorded in the Clusters table but that are
        not referenced in other tables.
        """
        ...

    # User, group and permissions management

    @abstractmethod
    def login(self, user: User) -> User:
        """Upserts the user to the database.

        Args:
            user: A User with a username and optionally an email
        """
        ...

    @abstractmethod
    def get_user_groups(self, user_name: str) -> list[GroupName]:
        """Get names of all groups a user belongs to.

        Args:
            user_name: The username to get the groups of
        """
        ...

    @abstractmethod
    def list_groups(self) -> list[Group]:
        """List all groups."""
        ...

    @abstractmethod
    def get_group(self, name: GroupName) -> Group:
        """Get group details including members.

        Args:
            name: The name of the group to fetch.

        Raises:
            MatchboxGroupNotFoundError: If group doesn't exist.
        """
        ...

    @abstractmethod
    def create_group(self, group: Group) -> None:
        """Create a new group.

        Arg:
            group: The group to create

        Raises:
            MatchboxGroupAlreadyExistsError: If group name taken.
        """
        ...

    @abstractmethod
    def delete_group(self, name: GroupName, certain: bool = False) -> None:
        """Delete a group.

        Args:
            name: The name of the group to delete
            certain: Whether to delete the group without confirmation.

        Raises:
            MatchboxGroupNotFoundError: If group doesn't exist.
            MatchboxSystemGroupError: If attempting to delete a system group.
            MatchboxDeletionNotConfirmed: If certain=False.
        """
        ...

    @abstractmethod
    def add_user_to_group(self, user_name: str, group_name: GroupName) -> None:
        """Add a user to a group. Creates user if they don't exist.

        Raises:
            MatchboxGroupNotFoundError: If group doesn't exist.
        """
        ...

    @abstractmethod
    def remove_user_from_group(self, user_name: str, group_name: GroupName) -> None:
        """Remove a user from a group.

        Raises:
            MatchboxGroupNotFoundError: If group doesn't exist.
            MatchboxUserNotFoundError: If user doesn't exist.
        """
        ...

    @abstractmethod
    def check_permission(
        self,
        user_name: str,
        permission: PermissionType,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> bool:
        """Check user permission against a resource.

        Args:
            user_name: The username to check.
            permission: The permission type to check
            resource: The resource to check. One of "system" or a collection name
        """
        ...

    @abstractmethod
    def get_permissions(
        self,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> list[PermissionGrant]:
        """Get all granted permissions against a resource.

        Args:
            resource: The resource to get permissions for. One of "system" or a
                collection name
        """
        ...

    @abstractmethod
    def grant_permission(
        self,
        group_name: GroupName,
        permission: PermissionType,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> None:
        """Grants a permission on a resource.

        Args:
            group_name: The name of the group to grant permission to
            permission: The permission to grant
            resource: The resource to grant permission on. One of "system" or a
                collection name
        """
        ...

    @abstractmethod
    def revoke_permission(
        self,
        group_name: GroupName,
        permission: PermissionType,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> None:
        """Revoke permission on a resource.

        Args:
            group_name: The name of the group to revoke permission of
            permission: The permission to revoke
            resource: The resource to revoke permission from. One of "system" or a
                collection name
        """
        ...

    # Evaluation management

    @abstractmethod
    def insert_judgement(self, judgement: Judgement) -> None:
        """Adds an evaluation judgement to the database.

        Args:
            judgement: representation of the proposed clusters.
        """
        ...

    @abstractmethod
    def get_judgements(self, tag: str | None = None) -> tuple[Table, Table]:
        """Retrieves all evaluation judgements.

        Args:
            tag: optional string by which to filter judgements

        Returns:
            Two PyArrow tables with the judgments and their expansion.
            See `matchbox.common.arrow` for information on the schema.
        """
        ...

    @abstractmethod
    def sample_for_eval(
        self, n: int, path: ModelResolutionPath, user_name: str
    ) -> Table:
        """Sample a cluster to validate.

        Args:
            n: Number of clusters to sample
            path: Path of resolution from which to sample
            user_name: Name of user requesting the sample

        Returns:
            An Arrow table with the same schema as returned by `query()`
        """
        ...
