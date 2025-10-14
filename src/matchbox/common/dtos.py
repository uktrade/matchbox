"""Data transfer objects for Matchbox API."""

import json
import re
import textwrap
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from importlib.metadata import version
from json import JSONDecodeError
from typing import Any, Self, TypeAlias

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from sqlglot import errors, expressions, parse_one

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS
from matchbox.common.exceptions import MatchboxNameError


class DataTypes(StrEnum):
    """Enumeration of supported data types.

    Uses polars datatypes as its backend.
    """

    # Boolean
    BOOLEAN = "Boolean"

    # Integers
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"

    # Unsigned integers
    UINT8 = "UInt8"
    UINT16 = "UInt16"
    UINT32 = "UInt32"
    UINT64 = "UInt64"

    # Floating point
    FLOAT32 = "Float32"
    FLOAT64 = "Float64"

    # Decimal
    DECIMAL = "Decimal"

    # String & Binary
    STRING = "String"
    BINARY = "Binary"

    # Date & Time related
    DATE = "Date"
    TIME = "Time"
    DATETIME = "Datetime"
    DURATION = "Duration"

    # Container types
    ARRAY = "Array"
    LIST = "List"

    # Special types
    OBJECT = "Object"
    CATEGORICAL = "Categorical"
    ENUM = "Enum"
    STRUCT = "Struct"
    NULL = "Null"

    def to_dtype(self) -> pl.DataType:
        """Convert enum value to actual polars dtype."""
        # Map from enum values to actual polars datatypes
        # We do this because polars datatypes are not directly serialisable in Pydantic
        dtype_map = {
            self.BOOLEAN: pl.Boolean,
            self.INT8: pl.Int8,
            self.INT16: pl.Int16,
            self.INT32: pl.Int32,
            self.INT64: pl.Int64,
            self.UINT8: pl.UInt8,
            self.UINT16: pl.UInt16,
            self.UINT32: pl.UInt32,
            self.UINT64: pl.UInt64,
            self.FLOAT32: pl.Float32,
            self.FLOAT64: pl.Float64,
            self.DECIMAL: pl.Decimal,
            self.STRING: pl.String,
            self.BINARY: pl.Binary,
            self.DATE: pl.Date,
            self.TIME: pl.Time,
            self.DATETIME: pl.Datetime,
            self.DURATION: pl.Duration,
            self.ARRAY: pl.Array,
            self.LIST: pl.List,
            self.OBJECT: pl.Object,
            self.CATEGORICAL: pl.Categorical,
            self.ENUM: pl.Enum,
            self.STRUCT: pl.Struct,
            self.NULL: pl.Null,
        }
        return dtype_map[self]

    def to_pytype(self) -> type:
        """Convert enum value to actual Python type."""
        return self.to_dtype().to_python()

    @classmethod
    def from_dtype(cls, dtype: pl.DataType) -> "DataTypes":
        """Get enum value from a polars dtype."""
        # Find the name of the dtype class
        dtype_name = dtype.__class__.__name__
        # Find the matching enum value
        for enum_val in cls:
            if enum_val.value == dtype_name:
                return enum_val
        raise ValueError(f"No matching polars DataTypes for dtype: {dtype_name}")

    @classmethod
    def from_pytype(cls, pytype: type) -> "DataTypes":
        """Get enum value from a Python type."""
        return DataTypes.from_dtype(pl.DataType.from_python(pytype))


class OKMessage(BaseModel):
    """Generic HTTP OK response."""

    status: str = Field(default="OK")
    version: str = Field(default_factory=lambda: version("matchbox-db"))


class LoginAttempt(BaseModel):
    """Request for log in process."""

    user_name: str


class LoginResult(BaseModel):
    """Response from log in process."""

    user_id: int


class BackendCountableType(StrEnum):
    """Enumeration of supported backend countable types."""

    SOURCES = "sources"
    MODELS = "models"
    DATA = "data"
    CLUSTERS = "clusters"
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


class BackendParameterType(StrEnum):
    """Enumeration of parameter types passable to the API."""

    SAMPLE_SIZE = "sample_size"
    NAME = "name"


class BackendUploadType(StrEnum):
    """Enumeration of supported backend upload types."""

    INDEX = "index"
    RESULTS = "results"

    @property
    def schema(self):
        """Get the schema for the upload type."""
        return {
            BackendUploadType.INDEX: SCHEMA_INDEX,
            BackendUploadType.RESULTS: SCHEMA_RESULTS,
        }[self]


class CRUDOperation(StrEnum):
    """Enumeration of CRUD operations."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class LocationType(StrEnum):
    """Enumeration of location types."""

    RDBMS = "rdbms"


class MatchboxName(str):
    """Sub-class of string which validates names for the Matchbox DB."""

    PATTERN = r"^[a-zA-Z0-9_.-]+$"

    def __new__(cls, value: str):
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
        cls, source_type: Any, handler: GetCoreSchemaHandler
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
            raise ValueError("Key field must be a string. ")

        return self

    @property
    def dependencies(self) -> list[ResolutionPath]:
        """Return all resolution names that this source needs.

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

    source_resolutions: tuple[SourceResolutionPath, ...]
    model_resolution: ModelResolutionPath | None = None
    combine_type: QueryCombineType = QueryCombineType.CONCAT
    threshold: int | None = None
    cleaning: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_resolutions(self) -> Self:
        """Ensure that resolution settings are compatible."""
        if not self.source_resolutions:
            raise ValueError("At least one source resolution required.")
        first_path = self.source_resolutions[0]
        if self.model_resolution and (
            self.model_resolution.collection != first_path.collection
            or self.model_resolution.run != first_path.run
        ):
            raise ValueError(
                "Incompatible collection and runs across model and source resolutions."
            )
        if len(self.source_resolutions) > 1:
            if not self.model_resolution:
                raise ValueError(
                    "A model resolution must be set if querying from multiple sources"
                )
            if any(
                [
                    res.collection != first_path.collection or res.run != first_path.run
                    for res in self.source_resolutions[1:]
                ]
            ):
                raise ValueError(
                    "Incompatible collection and runs across source resolutions."
                )
        return self

    @field_validator("cleaning")
    @classmethod
    def validate_cleaning_dict(cls, v: dict[str, str] | None) -> str | None:
        """Validate cleaning as valid SQL."""
        if v is None:
            return v

        for alias, sql in v.items():
            if sql is not None:
                try:
                    stmt = parse_one(sql, dialect="duckdb")
                except errors.ParseError as e:
                    raise ValueError(f"Invalid SQL in cleaning_dict: {alias}") from e

                for node in stmt.walk():
                    if isinstance(node, expressions.Column) and node.name == "id":
                        raise ValueError(
                            "Cannot transform 'id' column in cleaning_dict. "
                            "It is always selected by default."
                        )

        return v

    @property
    def dependencies(self) -> list[ResolutionPath]:
        """Return all resolution names that this query needs."""
        deps = list(self.source_resolutions)
        if self.model_resolution:
            deps.append(self.model_resolution)

        return deps

    @property
    def point_of_truth(self):
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

    def __eq__(self, other: Self) -> bool:
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
        if self.type == ModelType.LINKER:
            if (
                self.right_query.source_resolutions[0].collection
                != self.left_query.source_resolutions[0].collection
                or self.right_query.source_resolutions[0].run
                != self.left_query.source_resolutions[0].run
            ):
                raise ValueError("Left and right query must share collection and run.")
            if self.right_query is None:
                raise ValueError("Right query must be set for linkers")

        return self

    @field_validator("model_settings", mode="after")
    @classmethod
    def validate_settings_json(cls, value: str) -> str:
        """Ensure that the model settings is valid JSON."""
        try:
            isinstance(json.loads(value), dict)
        except JSONDecodeError as e:
            raise ValueError("Model settings are not valid JSON") from e
        return value

    @property
    def dependencies(self) -> list[ResolutionPath]:
        """Return all resolution names that this model needs."""
        deps = list(self.left_query.dependencies)
        if self.right_query:
            deps.extend(self.right_query.dependencies)

        return deps


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
    def serialise_ids(self, id_set: set[str]):
        """Turn set to sorted list when serialising."""
        return sorted(id_set)


class Resolution(BaseModel):
    """Unified resolution type with common fields and discriminated config."""

    description: str | None = Field(default=None, description="Description")
    truth: int | None = Field(default=None, ge=0, le=100, strict=True)

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
    def validate_resolution_type_matches_config(self):
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
    def validate_truth_matches_type(self):
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
    name: ResolutionPath | CollectionName | RunID
    operation: CRUDOperation
    details: str | None = None

    @classmethod
    def status_409_examples(cls) -> dict:
        """Examples for 409 status code."""
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "confirm_delete": {
                            "summary": "Delete operation requires confirmation. ",
                            "value": cls(
                                success=False,
                                name=ModelResolutionPath(
                                    collection="default",
                                    run=1,
                                    name="example_model",
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
                    },
                }
            }
        }

    @classmethod
    def status_500_examples(cls) -> dict:
        """Examples for 500 status code."""
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "unhandled": {
                            "summary": (
                                "Unhandled exception encountered while updating the "
                                "model's truth value."
                            ),
                            "value": cls(
                                success=False,
                                name=ModelResolutionPath(
                                    collection="default",
                                    run=1,
                                    name="example_model",
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
    AWAITING_UPLOAD = "awaiting_upload"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"
    UNKNOWN = "unknown"


class UploadStatus(BaseModel):
    """Response model for any file upload processes."""

    collection_name: CollectionName
    run_id: RunID
    resolution_name: ResolutionName
    path_str: str
    stage: UploadStage
    update_timestamp: datetime
    details: str | None = None
    entity: BackendUploadType | None = None

    _status_code_mapping = {
        UploadStage.READY: 200,
        UploadStage.COMPLETE: 200,
        UploadStage.FAILED: 400,
        UploadStage.AWAITING_UPLOAD: 202,
        UploadStage.QUEUED: 200,
        UploadStage.PROCESSING: 200,
    }

    def get_http_code(self) -> int:
        """Get the HTTP status code for the upload stage."""
        if self.stage == UploadStage.FAILED:
            return 400
        return self._status_code_mapping[self.stage]

    @classmethod
    def status_400_examples(cls) -> dict:
        """Examples for 400 status code."""
        return {
            "content": {
                "application/json": {
                    "examples": {
                        "expired_id": {
                            "summary": "Upload ID expired",
                            "value": cls(
                                id="example_id",
                                stage=UploadStage.FAILED,
                                details=(
                                    "Upload ID not found or expired. Entries expire "
                                    "after 30 minutes of inactivity, including "
                                    "failed processes."
                                ),
                                entity=BackendUploadType.INDEX,
                                update_timestamp=datetime.now(),
                            ).model_dump(),
                        },
                        "schema_mismatch": {
                            "summary": "Schema validation error",
                            "value": cls(
                                id="example_id",
                                stage=UploadStage.FAILED,
                                details="Schema mismatch. Expected: ... Got: ...",
                                entity=BackendUploadType.INDEX,
                                update_timestamp=datetime.now(),
                            ).model_dump(),
                        },
                    },
                }
            }
        }


class NotFoundError(BaseModel):
    """API error for a 404 status code."""

    details: str
    entity: BackendResourceType


class InvalidParameterError(BaseModel):
    """API error for a custom 422 status code."""

    details: str
    parameter: BackendParameterType | None
