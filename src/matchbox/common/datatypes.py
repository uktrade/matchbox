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
    Arrays have a fixed shape (e.g., Array(Int64, 3) or Array(Int64, (3,))).
    Structs have named fields (e.g., Struct({"name": String, "age": Int64})).
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    base_type: _TypeNames = Field(alias="type")
    inner: "DataTypes | None" = None
    shape: tuple[int, ...] | None = None  # Internally stores as tuples
    fields: dict[str, "DataTypes"] | None = None  # For Struct types

    def __call__(
        self,
        inner: "DataTypes | None" = None,
        shape: tuple[int, ...] | int | None = None,
        fields: dict[str, "DataTypes"] | None = None,
    ) -> "DataTypes":
        """Allow constructing nested types.

        Examples:
            DataTypes.LIST(DataTypes.STRING)
            DataTypes.ARRAY(DataTypes.INT64, shape=3)  # Normalised to (3,)
            DataTypes.ARRAY(DataTypes.INT64, shape=(3,))
            DataTypes.STRUCT(fields={"name": DataTypes.STRING, "age": DataTypes.INT64})
        """
        # Normalise int shape to tuple for consistency with Polars
        if isinstance(shape, int):
            shape = (shape,)
        return self.model_copy(
            update={"inner": inner, "shape": shape, "fields": fields}
        )

    def _to_dict(self) -> str | dict[str, Any]:
        """Convert to dict representation for JSON serialisation.

        Returns either a string (for simple types) or a dict (for complex types).
        """
        if self.inner is None and self.fields is None:
            return self.base_type.value

        data: dict[str, Any] = {"type": self.base_type.value}

        if self.inner is not None:
            data["inner"] = self.inner._to_dict()

        if self.shape is not None:
            data["shape"] = list(self.shape)  # Convert tuple to list for JSON

        if self.fields is not None:
            data["fields"] = {
                name: dtype._to_dict() for name, dtype in self.fields.items()
            }

        return data

    @property
    def value(self) -> str:
        """Return the string representation for storage.

        - Simple types return the enum value: "String"
        - Complex types return a JSON string: '{"type": "List", "inner": "String"}'
        - Arrays include shape: '{"type": "Array", "inner": "Int64", "shape": [3]}'
        - Structs include fields: '{"type": "Struct", "fields": {"name": "String"}}'
        """
        result = self._to_dict()
        if isinstance(result, str):
            return result
        return json.dumps(result)

    def to_dtype(self) -> pl.DataType | type[pl.DataType]:
        """Convert to Polars DataType."""
        BaseClass: pl.DataType = getattr(pl, self.base_type.value, pl.String)

        # Handle Struct types with fields
        if self.base_type == _TypeNames.STRUCT and self.fields is not None:
            # Convert fields dict to list of pl.Field objects
            field_list = [
                pl.Field(name, dtype.to_dtype()) for name, dtype in self.fields.items()
            ]
            return BaseClass(field_list)

        # Handle types with inner (List, Array)
        if self.inner:
            inner_dtype = self.inner.to_dtype()
            # Array types need both inner type and shape
            if self.base_type == _TypeNames.ARRAY and self.shape is not None:
                # Polars accepts tuple directly
                return BaseClass(inner_dtype, self.shape)
            return BaseClass(inner_dtype)

        try:
            return BaseClass()
        except TypeError:
            # Some types require arguments (e.g., List)
            return BaseClass

    def to_pytype(self) -> type:
        """Convert to Python type."""
        dtype = self.to_dtype()
        if isinstance(dtype, type):
            dtype = dtype()
        return dtype.to_python()

    @classmethod
    def from_dtype(cls, dtype: pl.DataType | type[pl.DataType]) -> "DataTypes":
        """Create DataTypes from a Polars DataType."""
        base_name: str
        inner: DataTypes | None = None
        shape: tuple[int, ...] | None = None
        fields: dict[str, DataTypes] | None = None

        # Handle both Polars type classes and instances
        if isinstance(dtype, type):
            base_name = dtype.__name__
        else:
            base_name = dtype.__class__.__name__

            # Extract inner for List/Array types
            if hasattr(dtype, "inner") and dtype.inner is not None:
                inner = cls.from_dtype(dtype.inner)

            # Extract shape for Array types (Polars stores as tuple)
            if hasattr(dtype, "shape") and dtype.shape is not None:
                shape = (
                    dtype.shape if isinstance(dtype.shape, tuple) else (dtype.shape,)
                )

            # Extract fields for Struct types
            if hasattr(dtype, "fields") and dtype.fields is not None:
                fields = {
                    field.name: cls.from_dtype(field.dtype) for field in dtype.fields
                }

        # Polars uses "Utf8" internally for strings
        base_name = "String" if base_name == "Utf8" else base_name

        return cls(
            base_type=_TypeNames(base_name), inner=inner, shape=shape, fields=fields
        )

    @classmethod
    def from_pytype(cls, pytype: type) -> "DataTypes":
        """Create from Python type (current behavior)."""
        return cls.from_dtype(pl.DataType.from_python(pytype))

    @model_validator(mode="before")
    @classmethod
    def _parse_input(cls, value: Any) -> Any:  # noqa: ANN401
        """Parse string, enum, or dict inputs."""
        if isinstance(value, dict):
            # Normalise shape from int or list to tuple
            if "shape" in value:
                shape_val = value["shape"]
                if isinstance(shape_val, int):
                    value["shape"] = (shape_val,)
                elif isinstance(shape_val, list):
                    value["shape"] = tuple(shape_val)

            # Recursively parse fields if present
            if "fields" in value and isinstance(value["fields"], dict):
                value["fields"] = {
                    name: cls._parse_input(dtype)
                    if not isinstance(dtype, cls)
                    else dtype
                    for name, dtype in value["fields"].items()
                }
            return value

        if isinstance(value, _TypeNames):
            return {"type": value}

        if isinstance(value, str):
            # Try parsing JSON for nested types
            if value.startswith("{"):
                try:
                    parsed = json.loads(value)
                    return cls._parse_input(parsed)
                except json.JSONDecodeError:
                    pass
            # Simple type: "String"
            return {"type": value}

        return value

    @model_serializer
    def _serialise(self) -> str:
        """Serialise to string.

        * Simple types: "String"
        * Nested types: '{"type": "List", "inner": "String"}'
        * Arrays: '{"type": "Array", "inner": "Int64", "shape": [3]}'
        * Structs: '{"type": "Struct", "fields": {"name": "String", "age": "Int64"}}'
        """
        return self.value

    def __str__(self) -> str:
        """Human-readable string.

        * List(String)
        * Array(Int64, (3,))
        * Struct({"name": String, "age": Int64})
        """
        if self.fields:
            fields_str = (
                "{" + ", ".join(f'"{k}": {v}' for k, v in self.fields.items()) + "}"
            )
            return f"{self.base_type.value}({fields_str})"
        if self.inner:
            if self.shape is not None:
                return f"{self.base_type.value}({self.inner}, {self.shape})"
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
