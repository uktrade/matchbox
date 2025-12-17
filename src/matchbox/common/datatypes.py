"""Internal Matchbox data types."""

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class _TypeNames(StrEnum):
    """Enumeration of supported data types."""

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


class DataTypes(BaseModel):
    """Recursive definition of a data type.

    Represents Polars data types in a serialisable format.
    Can be simple (e.g., "String") or nested (e.g., List(String)).
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    base_type: _TypeNames = Field(alias="type")
    inner: "DataTypes | None" = None

    def __call__(self, inner: "DataTypes") -> "DataTypes":
        """Allow constructing nested types: DataTypes.LIST(DataTypes.STRING)."""
        return self.model_copy(update={"inner": inner})

    @property
    def value(self) -> str:
        """Return the string representation for storage.

        - Simple types return the enum value: "String"
        - Complex types return a JSON string: '{"type": "List", "inner": "String"}'
        """
        if self.inner is None:
            return self.base_type.value

        # Recursively build the JSON structure
        data = {"type": self.base_type.value, "inner": self.inner.value}
        return json.dumps(data)

    def to_dtype(self) -> pl.DataType | type[pl.DataType]:
        """Convert to Polars DataType."""
        base_cls = getattr(pl, self.base_type.value, pl.String)

        if self.inner:
            return base_cls(self.inner.to_dtype())

        try:
            return base_cls()
        except TypeError:
            # Some types require arguments (e.g., List)
            return base_cls

    def to_pytype(self) -> type:
        """Convert to Python type."""
        dtype = self.to_dtype()
        if isinstance(dtype, type):
            dtype = dtype()
        return dtype.to_python()

    @classmethod
    def from_dtype(cls, dtype: pl.DataType | type[pl.DataType]) -> "DataTypes":
        """Create DataTypes from a Polars DataType."""
        inner: DataTypes | None = None

        # Handle both Polars type classes and instances
        if isinstance(dtype, type):
            base_name = dtype.__name__
        else:
            base_name = dtype.__class__.__name__
            if hasattr(dtype, "inner") and dtype.inner is not None:
                inner = cls.from_dtype(dtype.inner)

        # Polars uses "Utf8" internally for strings
        base_name = "String" if base_name == "Utf8" else base_name

        return cls(base_type=_TypeNames(base_name), inner=inner)

    @classmethod
    def from_pytype(cls, pytype: type) -> "DataTypes":
        """Create from Python type."""
        return cls.from_dtype(pl.DataType.from_python(pytype))

    @model_validator(mode="before")
    @classmethod
    def _parse_input(cls, value: Any) -> Any:  # noqa: ANN401
        """Parse string, enum, or dict inputs."""
        if isinstance(value, dict):
            return value

        if isinstance(value, _TypeNames):
            return {"type": value}

        if isinstance(value, str):
            # Try parsing JSON for nested types: '{"type": "List", "inner": "String"}'
            if value.startswith("{"):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            # Simple type: "String"
            return {"type": value}

        return value

    @model_serializer
    def _serialise(self) -> str:
        """Serialise to string.

        Simple types: "String"
        Nested types: '{"type": "List", "inner": "String"}'
        """
        return self.value

    def __str__(self) -> str:
        """Human-readable string: List(String)."""
        if self.inner:
            return f"{self.base_type.value}({self.inner})"
        return self.base_type.value

    def __repr__(self) -> str:
        """Interpreter representation."""
        if not self.inner:
            return f"DataTypes.{self.base_type.name}"
        return super().__repr__()

    # Type hints for IDE autocompletion
    if TYPE_CHECKING:
        BOOLEAN: ClassVar["DataTypes"]
        INT8: ClassVar["DataTypes"]
        INT16: ClassVar["DataTypes"]
        INT32: ClassVar["DataTypes"]
        INT64: ClassVar["DataTypes"]
        UINT8: ClassVar["DataTypes"]
        UINT16: ClassVar["DataTypes"]
        UINT32: ClassVar["DataTypes"]
        UINT64: ClassVar["DataTypes"]
        FLOAT32: ClassVar["DataTypes"]
        FLOAT64: ClassVar["DataTypes"]
        DECIMAL: ClassVar["DataTypes"]
        STRING: ClassVar["DataTypes"]
        BINARY: ClassVar["DataTypes"]
        DATE: ClassVar["DataTypes"]
        TIME: ClassVar["DataTypes"]
        DATETIME: ClassVar["DataTypes"]
        DURATION: ClassVar["DataTypes"]
        ARRAY: ClassVar["DataTypes"]
        LIST: ClassVar["DataTypes"]
        OBJECT: ClassVar["DataTypes"]
        CATEGORICAL: ClassVar["DataTypes"]
        ENUM: ClassVar["DataTypes"]
        STRUCT: ClassVar["DataTypes"]
        NULL: ClassVar["DataTypes"]


# Set up enum-like class attributes: DataTypes.STRING, DataTypes.INT64, etc.
for member in _TypeNames:
    setattr(DataTypes, member.name, DataTypes(base_type=member))
