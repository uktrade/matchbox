"""Data transfer objects for Matchbox API."""

from enum import StrEnum
from importlib.metadata import version
from typing import Literal, TypeAlias

import polars as pl
from pydantic import BaseModel, Field

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS

SourceResolutionName: TypeAlias = str
"""Type alias for source resolution names."""

ModelResolutionName: TypeAlias = str
"""Type alias for model resolution names."""

ResolutionName = SourceResolutionName | ModelResolutionName
"""Type alias for resolution names."""


class OKMessage(BaseModel):
    """Generic HTTP OK response."""

    status: str = Field(default="OK")
    version: str = Field(default_factory=lambda: version("matchbox-db"))


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


class BackendRetrievableType(StrEnum):
    """Enumeration of supported backend retrievable types."""

    SOURCE = "source"
    RESOLUTION = "resolution"


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


class ModelAncestor(BaseModel):
    """A model's ancestor and its truth value."""

    name: ModelResolutionName = Field(..., description="Name of the ancestor model")
    truth: int | None = Field(
        default=None, description="Truth threshold value", ge=0, le=100, strict=True
    )


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


class UploadStatus(BaseModel):
    """Response model for any file upload processes."""

    id: str | None = None
    status: Literal[
        "ready", "awaiting_upload", "queued", "processing", "complete", "failed"
    ]
    details: str | None = None
    entity: BackendUploadType | None = None

    _status_code_mapping = {
        "ready": 200,
        "complete": 200,
        "failed": 400,
        "awaiting_upload": 202,
        "queued": 200,
        "processing": 200,
    }

    def get_http_code(self, status: bool) -> int:
        """Get the HTTP status code for the upload status."""
        if self.status == "failed":
            return 400
        return self._status_code_mapping[self.status]

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
                                status="failed",
                                details=(
                                    "Upload ID not found or expired. Entries expire "
                                    "after 30 minutes of inactivity, including "
                                    "failed processes."
                                ),
                                entity=BackendUploadType.INDEX,
                            ).model_dump(),
                        },
                        "schema_mismatch": {
                            "summary": "Schema validation error",
                            "value": cls(
                                id="example_id",
                                status="failed",
                                details="Schema mismatch. Expected: ... Got: ...",
                                entity=BackendUploadType.INDEX,
                            ).model_dump(),
                        },
                    },
                }
            }
        }


class NotFoundError(BaseModel):
    """API error for a 404 status code."""

    details: str
    entity: BackendRetrievableType


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
