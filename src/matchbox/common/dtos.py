"""Data transfer objects for Matchbox API."""

import re
import textwrap
from datetime import datetime
from enum import StrEnum
from importlib.metadata import version
from typing import Self

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS
from matchbox.common.graph import (
    ModelResolutionName,
    ResolutionName,
)


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

    SOURCE = "source"
    RESOLUTION = "resolution"
    CLUSTER = "cluster"
    USER = "user"
    JUDGEMENT = "judgement"


class BackendParameterType(StrEnum):
    """Enumeration of parameters passable to the API."""

    SAMPLE_SIZE = "sample_size"


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


class ModelType(StrEnum):
    """Enumeration of supported model types."""

    LINKER = "linker"
    DEDUPER = "deduper"


class CRUDOperation(StrEnum):
    """Enumeration of CRUD operations."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class LocationType(StrEnum):
    """Enumeration of location types."""

    RDBMS = "rdbms"


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
    name: str = Field(
        description=(
            "A unique, human-readable name of the source resolution this "
            "object configures."
        )
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

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Ensure the name is a valid source resolution name.

        Raises:
            ValueError: If the name is not a valid source resolution name.
        """
        if not re.match(r"^[a-zA-Z0-9_]+$", value):
            raise ValueError("Source names must be alphanumeric and underscore only. ")
        return value

    @model_validator(mode="after")
    def validate_key_field(self) -> Self:
        """Ensure that the key field is a string and not in the index fields."""
        if self.key_field in self.index_fields:
            raise ValueError("Key field must not be in the index fields. ")

        if self.key_field.type != DataTypes.STRING:
            raise ValueError("Key field must be a string. ")

        return self

    @property
    def prefix(self) -> str:
        """Get the prefix for the source."""
        return self.name + "_"

    @property
    def qualified_key(self) -> str:
        """Get the qualified key for the source."""
        return self.qualify_field(self.key_field.name)

    @property
    def qualified_index_fields(self) -> list[str]:
        """Get the qualified index fields for the source."""
        return [self.qualify_field(field.name) for field in self.index_fields]

    def qualify_field(self, field: str) -> str:
        """Qualify field names with the source name.

        Args:
            field: The field name to qualify.

        Returns:
            A single qualified field.

        """
        return self.prefix + field


class ModelConfig(BaseModel):
    """Metadata for a model."""

    name: ModelResolutionName
    description: str
    type: ModelType
    left_resolution: ResolutionName
    right_resolution: ResolutionName | None = None  # Only used for linker models

    def __eq__(self, other: "ModelConfig") -> bool:
        """Check equality of model configurations.

        Model configurations don't care about the order of left and right resolutions.
        """
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return (
            self.name == other.name
            and self.description == other.description
            and self.type == other.type
            and {self.left_resolution, self.right_resolution}
            == {other.left_resolution, other.right_resolution}
        )


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: str
    source_id: set[str] = Field(default_factory=set)
    target: str
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


class ResolutionOperationStatus(BaseModel):
    """Status response for any resolution operation."""

    success: bool
    name: ModelResolutionName
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
                                name="example_model",
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
                                name="example_model",
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

    id: str
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
    parameter: BackendParameterType
