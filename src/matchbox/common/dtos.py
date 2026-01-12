"""Data transfer objects for Matchbox API."""

import json
import re
import textwrap
from collections.abc import Iterable
from enum import StrEnum
from importlib.metadata import version
from typing import Annotated, Any, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    PlainSerializer,
    PlainValidator,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from matchbox.common.datatypes import DataTypes
from matchbox.common.exceptions import MatchboxExceptionType, MatchboxNameError
from matchbox.common.hash import base64_to_hash, hash_to_base64


class MatchboxName(str):
    """Sub-class of string which validates names for the Matchbox DB."""

    PATTERN = r"^[a-zA-Z0-9_.-]+$"

    def __new__(cls, value: str) -> Self:
        """Creates new instance of validated name."""
        if not isinstance(value, str):
            raise MatchboxNameError("Name must be a string")
        if not re.match(cls.PATTERN, value):
            raise MatchboxNameError(
                f"Name '{value}' is invalid. It can only include "
                "alphanumeric characters, underscores, dots or hyphens."
            )

        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,  # noqa: ANN401
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Generate core schema for Pydantic compatibility."""
        return core_schema.no_info_plain_validator_function(
            cls,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: x, return_schema=core_schema.str_schema()
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate JSON schema for OpenAPI documentation."""
        json_schema = handler(core_schema.str_schema())
        json_schema.update(
            {
                "type": "string",
                "pattern": cls.PATTERN,
            }
        )
        return json_schema


class OKMessage(BaseModel):
    """Generic HTTP OK response."""

    status: str = Field(default="OK")
    version: str = Field(default_factory=lambda: version("matchbox-db"))


class BackendCountableType(StrEnum):
    """Enumeration of supported backend countable types."""

    SOURCES = "sources"
    MODELS = "models"
    SOURCE_CLUSTERS = "source_clusters"
    MODEL_CLUSTERS = "model_clusters"
    CLUSTERS = "all_clusters"
    CREATES = "creates"
    MERGES = "merges"
    PROPOSES = "proposes"


class ModelResultsType(StrEnum):
    """Enumeration of supported model results types."""

    PROBABILITIES = "probabilities"
    CLUSTERS = "clusters"


class BackendResourceType(StrEnum):
    """Enumeration of resources types referenced by client or API."""

    COLLECTION = "collection"
    RUN = "run"
    RESOLUTION = "resolution"
    CLUSTER = "cluster"
    USER = "user"
    JUDGEMENT = "judgement"
    SYSTEM = "system"


class BackendParameterType(StrEnum):
    """Enumeration of parameter types passable to the API."""

    SAMPLE_SIZE = "sample_size"
    NAME = "name"


class CRUDOperation(StrEnum):
    """Enumeration of CRUD operations."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class LocationType(StrEnum):
    """Enumeration of location types."""

    RDBMS = "rdbms"


class PermissionType(StrEnum):
    """Permission levels for resource access."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class User(BaseModel):
    """User identity."""

    model_config = ConfigDict(populate_by_name=True)

    user_name: str = Field(alias="sub")
    email: EmailStr | None = None


GroupName: TypeAlias = MatchboxName
"""Type alias for group names."""


class Group(BaseModel):
    """Group definition."""

    name: GroupName
    description: str | None = None
    is_system: bool = False
    members: list[User] = []


class PermissionGrant(BaseModel):
    """A permission on a resource.

    Resource context should always be supplied.
    """

    group_name: GroupName
    permission: PermissionType


class AuthStatusResponse(BaseModel):
    """Response model for authentication status."""

    authenticated: bool
    username: str | None = None
    token: str | None = None


CollectionName: TypeAlias = MatchboxName
"""Type alias for collection names."""

RunID: TypeAlias = int
"""Type alias for run IDs."""

SourceResolutionName: TypeAlias = MatchboxName
"""Type alias for source resolution names."""

ModelResolutionName: TypeAlias = MatchboxName
"""Type alias for model resolution names."""

ResolutionName: TypeAlias = SourceResolutionName | ModelResolutionName
"""Type alias for any resolution names."""


class ResolutionPath(BaseModel):
    """Base resolution identifier with collection, run, and name."""

    model_config = ConfigDict(frozen=True)

    collection: CollectionName
    run: RunID
    name: ResolutionName

    def __str__(self) -> str:
        """String representation of the resolution path."""
        return f"{self.collection}/{self.run}/{self.name}"


SourceResolutionPath: TypeAlias = ResolutionPath
"""Type alias for source resolution paths."""

ModelResolutionPath: TypeAlias = ResolutionPath
"""Type alias for model resolution paths."""


class ResolutionType(StrEnum):
    """Types of nodes in a resolution."""

    SOURCE = "source"
    MODEL = "model"


class LocationConfig(BaseModel):
    """Metadata for a location."""

    model_config = ConfigDict(frozen=True)

    type: LocationType
    name: str


class SourceField(BaseModel):
    """A field in a source that can be indexed in the Matchbox database."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description=(
            "The name of the field in the source after the "
            "extract/transform logic has been applied."
        )
    )
    type: DataTypes = Field(
        description="The cached field type. Used to ensure a stable hash.",
    )


class SourceConfig(BaseModel):
    """Configuration of a source that can, or has been, indexed in the backend.

    They are foundational processes on top of which linking and deduplication models can
    build new resolutions.
    """

    model_config = ConfigDict(frozen=True)

    location_config: LocationConfig = Field(
        description=(
            "The location of the source. Used to run the extract/tansform logic."
        ),
    )
    extract_transform: str = Field(
        description=(
            "Logic to extract and transform data from the source. "
            "Language is location dependent."
        )
    )
    # Fields can to be set at creation, or initialised with `.default_columns()`
    key_field: SourceField = Field(
        description=textwrap.dedent("""
            The key field. This is the source's key for unique
            entities, such as a primary key in a relational database.

            Keys must ALWAYS be a string.

            For example, if the source describes companies, it may have used
            a Companies House number as its key.

            This key is ALWAYS correct. It should be something generated and
            owned by the source being indexed.
            
            For example, your organisation's CRM ID is a key field within the CRM.
            
            A CRM ID entered by hand in another dataset shouldn't be used 
            as a key field.
        """),
    )
    index_fields: tuple[SourceField, ...] = Field(
        default=None,
        description=textwrap.dedent(
            """
            The fields to index in this source, after the extract/transform logic 
            has been applied. 

            This is usually set manually, and should map onto the columns that the
            extract/transform logic returns.
            """
        ),
    )

    @model_validator(mode="after")
    def validate_key_field(self) -> Self:
        """Ensure that the key field is a string and not in the index fields."""
        if self.key_field in self.index_fields:
            raise ValueError("Key field must not be in the index fields. ")

        if self.key_field.type != DataTypes.STRING:
            raise ValueError("Key field must have string type.")

        return self

    @property
    def dependencies(self) -> list[ResolutionName]:
        """Return all resolution names that this source needs.

        Provided for symmetry with ModelConfig.
        """
        return []

    @property
    def parents(self) -> list[ResolutionName]:
        """Returns all resolution names directly input to this config.

        Provided for symmetry with ModelConfig.
        """
        return []

    def prefix(self, name: str) -> str:
        """Get the prefix for the source.

        Args:
            name: The name of the source.

        Returns:
            The prefix string (name + "_").
        """
        return name + "_"

    def qualified_key(self, name: str) -> str:
        """Get the qualified key for the source.

        Args:
            name: The name of the source.

        Returns:
            The qualified key field name.
        """
        return self.qualify_field(name, self.key_field.name)

    def qualified_index_fields(self, name: str) -> list[str]:
        """Get the qualified index fields for the source.

        Args:
            name: The name of the source.

        Returns:
            List of qualified index field names.
        """
        return [self.qualify_field(name, field.name) for field in self.index_fields]

    def qualify_field(self, name: str, field: str) -> str:
        """Qualify field names with the source name.

        Args:
            name: The name of the source.
            field: The field name to qualify.

        Returns:
            A single qualified field.
        """
        return self.prefix(name) + field

    def f(self, name: str, fields: str | Iterable[str]) -> str | list[str]:
        """Qualify one or more field names with the source name.

        Args:
            name: The name of the source.
            fields: The field name to qualify, or a list of field names.

        Returns:
            A single qualified field, or a list of qualified field names.
        """
        if isinstance(fields, str):
            return self.qualify_field(name, fields)
        return [self.qualify_field(name, field_name) for field_name in fields]


class QueryCombineType(StrEnum):
    """Enumeration of ways to combine multiple rows having the same matchbox ID."""

    CONCAT = "concat"
    EXPLODE = "explode"
    SET_AGG = "set_agg"


class QueryConfig(BaseModel):
    """Configuration of query generating model inputs."""

    model_config = ConfigDict(frozen=True)

    source_resolutions: tuple[SourceResolutionName, ...]
    model_resolution: ModelResolutionName | None = None
    combine_type: QueryCombineType = QueryCombineType.CONCAT
    threshold: int | None = None
    cleaning: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_resolutions(self) -> Self:
        """Ensure that resolution settings are compatible."""
        if not self.source_resolutions:
            raise ValueError("At least one source resolution required.")
        if len(self.source_resolutions) > 1 and not self.model_resolution:
            raise ValueError(
                "A model resolution must be set if querying from multiple sources"
            )
        return self

    @property
    def dependencies(self) -> list[ResolutionName]:
        """Return all resolutions that this query needs."""
        deps = list(self.source_resolutions)
        if self.model_resolution:
            deps.append(self.model_resolution)

        return deps

    @property
    def point_of_truth(self) -> ResolutionName:
        """Return path of resolution that will be used as point of truth."""
        if self.model_resolution:
            return self.model_resolution
        return self.source_resolutions[0]


class ModelType(StrEnum):
    """Enumeration of supported model types."""

    LINKER = "linker"
    DEDUPER = "deduper"


class ModelConfig(BaseModel):
    """Configuration for model that has or could be added to the server."""

    type: ModelType
    model_class: str
    model_settings: str
    left_query: QueryConfig
    right_query: QueryConfig | None = None  # Only used for linker models

    def __eq__(self, other: object) -> bool:
        """Check equality of model configurations.

        Model configurations don't care about the order of left and right resolutions.
        """
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return self.type == other.type and {
            self.left_query,
            self.right_query,
        } == {other.left_query, other.right_query}

    @model_validator(mode="after")
    def validate_right_query(self) -> Self:
        """Ensure that a right query is set if and only if model is linker."""
        if self.type == ModelType.DEDUPER and self.right_query is not None:
            raise ValueError("Right query can't be set for dedupers")
        if self.type == ModelType.LINKER and self.right_query is None:
            raise ValueError("Right query must be set for linkers")

        return self

    @field_validator("model_settings", mode="after")
    @classmethod
    def validate_settings_json(cls, value: str) -> str:
        """Ensure that the model settings is valid JSON."""
        try:
            isinstance(json.loads(value), dict)
        except json.JSONDecodeError as e:
            raise ValueError("Model settings are not valid JSON") from e
        return value

    @property
    def dependencies(self) -> list[ResolutionName]:
        """Return all resolutions that this model needs."""
        deps = list(self.left_query.dependencies)
        if self.right_query:
            deps.extend(self.right_query.dependencies)

        return deps

    @property
    def parents(self) -> list[ResolutionName]:
        """Returns all resolution names directly input to this config."""
        if self.right_query:
            return [
                self.left_query.point_of_truth,
                self.right_query.point_of_truth,
            ]
        return [self.left_query.point_of_truth]


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: SourceResolutionPath
    source_id: set[str] = Field(default_factory=set)
    target: SourceResolutionPath
    target_id: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def found_or_none(self) -> "Match":
        """Ensure that a match has sources and a cluster if target was found."""
        if self.target_id and not (self.source_id and self.cluster):
            raise ValueError(
                "A match must have sources and a cluster if target was found."
            )
        if self.cluster and not self.source_id:
            raise ValueError("A match must have source if cluster is set.")
        return self

    @field_serializer("source_id", "target_id")
    def serialise_ids(self, id_set: set[str]) -> list[str]:
        """Turn set to sorted list when serialising."""
        return sorted(id_set)


class Resolution(BaseModel):
    """Unified resolution type with common fields and discriminated config."""

    description: str | None = Field(default=None, description="Description")
    truth: int | None = Field(default=None, ge=0, le=100, strict=True)
    fingerprint: Annotated[
        bytes,
        PlainSerializer(hash_to_base64, return_type=str),
        PlainValidator(base64_to_hash),
    ]

    # Discriminator field
    resolution_type: ResolutionType

    # Type-specific config as discriminated union
    config: SourceConfig | ModelConfig

    @field_validator("description", mode="after")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        """Ensure the description is not empty if provided."""
        if value is not None and not value.strip():
            raise ValueError("Description cannot be empty if provided.")
        return value

    @model_validator(mode="after")
    def validate_resolution_type_matches_config(self) -> Self:
        """Ensure resolution_type matches the config type."""
        if self.resolution_type == ResolutionType.SOURCE:
            assert isinstance(self.config, SourceConfig), (
                "Config must be SourceConfig when resolution_type is 'source'"
            )
        else:
            assert isinstance(self.config, ModelConfig), (
                "Config must be ModelConfig when resolution_type is 'model'"
            )
        return self

    @model_validator(mode="after")
    def validate_truth_matches_type(self) -> Self:
        """Ensure truth field matches resolution type requirements."""
        if self.resolution_type == ResolutionType.SOURCE and self.truth is not None:
            raise ValueError("Truth must be None for source resolutions")
        elif self.resolution_type == ResolutionType.MODEL:
            if self.truth is None:
                raise ValueError("Truth is required for model resolutions")
            if not (0 <= self.truth <= 100):
                raise ValueError(
                    "Truth must be between 0 and 100 for model resolutions"
                )
        return self


class Run(BaseModel):
    """A run within a collection."""

    run_id: RunID | None = Field(description="Unique ID of the run")
    is_default: bool = Field(
        default=False,
        description="Whether this run is the default in its collection",
    )
    is_mutable: bool = Field(
        default=False, description="Whether this run can be modified"
    )
    resolutions: dict[ResolutionName, Resolution] = Field(
        default_factory=dict,
        description="Dict of resolution objects by name within this run",
    )


class Collection(BaseModel):
    """A collection of runs."""

    default_run: RunID | None = Field(
        default=None, description="ID of default run for this collection"
    )
    runs: list[RunID] = Field(
        default_factory=list, description="List of run IDs in this collection"
    )

    @model_validator(mode="after")
    def validate_default_run(self) -> Self:
        """Check default run is within all runs."""
        if self.default_run and self.default_run not in self.runs:
            raise ValueError(
                "The default run needs to be included in the list of all runs."
            )

        return self


class ResourceOperationStatus(BaseModel):
    """Status response for any resource operation."""

    success: bool
    target: str
    operation: CRUDOperation
    details: str | None = None

    @classmethod
    def error_examples(cls) -> dict:
        """Examples for error codes."""
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "confirm_delete": {
                            "summary": "Delete operation requires confirmation. ",
                            "value": cls(
                                success=False,
                                target=str(
                                    ModelResolutionPath(
                                        collection="default",
                                        run=1,
                                        name="example_model",
                                    )
                                ),
                                operation=CRUDOperation.DELETE,
                                details=(
                                    "This operation will delete the resolutions "
                                    "deduper_1, deduper_2, linker_1, "
                                    "as well as all probabilities they have created. "
                                    "\n\n"
                                    "It won't delete validation associated with these "
                                    "probabilities. \n\n"
                                    "If you're sure you want to continue, rerun with "
                                    "certain=True"
                                ),
                            ).model_dump(),
                        },
                        "unhandled": {
                            "summary": (
                                "Unhandled exception encountered while updating the "
                                "model's truth value."
                            ),
                            "value": cls(
                                success=False,
                                target=str(
                                    ModelResolutionPath(
                                        collection="default",
                                        run=1,
                                        name="example_model",
                                    )
                                ),
                                operation=CRUDOperation.UPDATE,
                            ).model_dump(),
                        },
                    },
                }
            }
        }


class CountResult(BaseModel):
    """Response model for count results."""

    entities: dict[BackendCountableType, int]


class UploadStage(StrEnum):
    """Enumeration of stages of a file upload and its processing."""

    READY = "ready"
    PROCESSING = "processing"
    COMPLETE = "complete"


class UploadInfo(BaseModel):
    """Response model for file upload processes."""

    stage: UploadStage | None = None
    error: str | None = None


class NotFoundError(BaseModel):
    """API error for a 404 status code."""

    details: str
    entity: BackendResourceType


class InvalidParameterError(BaseModel):
    """API error for a custom 422 status code."""

    details: str
    parameter: BackendParameterType | None


class ErrorResponse(BaseModel):
    """Unified error response for all HTTP error status codes.

    This DTO enables the client to reconstruct the exact exception
    type that was raised on the server.
    """

    exception_type: MatchboxExceptionType = Field(
        description="The name of the exception class raised on the server"
    )
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Exception-specific data for reconstruction"
    )
