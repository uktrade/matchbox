"""Minimal test coverage for DataTypes with Array and Struct support."""

import json

import polars as pl
import pytest

from matchbox.common.datatypes import DataTypes


class TestSimpleTypes:
    """Test creation and serialisation of simple types."""

    @pytest.mark.parametrize(
        ("dtype", "expected_value"),
        [
            (DataTypes.STRING, "String"),
            (DataTypes.INT64, "Int64"),
            (DataTypes.FLOAT64, "Float64"),
            (DataTypes.BOOLEAN, "Boolean"),
            (DataTypes.DATETIME, "Datetime"),
        ],
    )
    def test_simple_type_serialisation(
        self, dtype: DataTypes, expected_value: str
    ) -> None:
        """Simple types serialise to their string name."""
        assert dtype.value == expected_value

    @pytest.mark.parametrize(
        "dtype",
        [DataTypes.STRING, DataTypes.INT64, DataTypes.FLOAT64],
    )
    def test_simple_type_roundtrip(self, dtype: DataTypes) -> None:
        """Simple types can round-trip through their string representation."""
        deserialised = DataTypes.model_validate(dtype.value)
        assert deserialised.value == dtype.value


class TestListTypes:
    """Test List container types."""

    @pytest.mark.parametrize(
        ("inner_type", "expected_json"),
        [
            (DataTypes.STRING, '{"type": "List", "inner": "String"}'),
            (DataTypes.INT64, '{"type": "List", "inner": "Int64"}'),
        ],
    )
    def test_list_serialisation(
        self, inner_type: DataTypes, expected_json: str
    ) -> None:
        """Lists serialise with their inner type."""
        list_type = DataTypes.LIST(inner_type)
        assert list_type.value == expected_json

    def test_list_roundtrip(self) -> None:
        """List types can round-trip through JSON."""
        original = DataTypes.LIST(DataTypes.FLOAT64)
        deserialised = DataTypes.model_validate_json(original.value)
        assert deserialised.value == original.value

    def test_nested_list(self) -> None:
        """Nested lists serialise correctly."""
        nested = DataTypes.LIST(DataTypes.LIST(DataTypes.STRING))
        expected = '{"type": "List", "inner": {"type": "List", "inner": "String"}}'
        assert nested.value == expected


class TestArrayTypes:
    """Test Array container types with fixed shape."""

    @pytest.mark.parametrize(
        ("inner_type", "shape", "expected_json"),
        [
            (DataTypes.INT64, 3, '{"type": "Array", "inner": "Int64", "shape": [3]}'),
            (
                DataTypes.FLOAT32,
                5,
                '{"type": "Array", "inner": "Float32", "shape": [5]}',
            ),
            (
                DataTypes.INT64,
                (3,),
                '{"type": "Array", "inner": "Int64", "shape": [3]}',
            ),
        ],
    )
    def test_array_serialisation(
        self, inner_type: DataTypes, shape: int | tuple[int, ...], expected_json: str
    ) -> None:
        """Arrays serialise with inner type and shape."""
        array_type = DataTypes.ARRAY(inner_type, shape=shape)
        assert array_type.value == expected_json

    def test_array_roundtrip(self) -> None:
        """Array types can round-trip through JSON."""
        original = DataTypes.ARRAY(DataTypes.INT32, shape=10)
        deserialised = DataTypes.model_validate_json(original.value)
        assert deserialised.value == original.value
        assert deserialised.shape == (10,)

    def test_array_of_lists(self) -> None:
        """Arrays can contain Lists."""
        array_of_lists = DataTypes.ARRAY(DataTypes.LIST(DataTypes.STRING), shape=2)
        deserialised = DataTypes.model_validate_json(array_of_lists.value)
        assert deserialised.shape == (2,)
        assert deserialised.inner is not None
        assert deserialised.inner.base_type.value == "List"


class TestStructTypes:
    """Test Struct types with named fields."""

    def test_simple_struct_serialisation(self) -> None:
        """Structs serialise with their fields."""
        struct = DataTypes.STRUCT(
            fields={"name": DataTypes.STRING, "age": DataTypes.INT64}
        )
        parsed = json.loads(struct.value)
        assert parsed["type"] == "Struct"
        assert parsed["fields"]["name"] == "String"
        assert parsed["fields"]["age"] == "Int64"

    def test_struct_roundtrip(self) -> None:
        """Struct types can round-trip through JSON."""
        original = DataTypes.STRUCT(
            fields={"id": DataTypes.INT64, "value": DataTypes.FLOAT64}
        )
        deserialised = DataTypes.model_validate_json(original.value)
        assert deserialised.value == original.value
        assert deserialised.fields is not None
        assert len(deserialised.fields) == 2

    def test_nested_struct(self) -> None:
        """Structs can contain other Structs."""
        inner_struct = DataTypes.STRUCT(fields={"x": DataTypes.FLOAT64})
        outer_struct = DataTypes.STRUCT(fields={"point": inner_struct})

        deserialised = DataTypes.model_validate_json(outer_struct.value)
        assert deserialised.fields is not None
        assert "point" in deserialised.fields
        point_field = deserialised.fields["point"]
        assert point_field.base_type.value == "Struct"
        assert point_field.fields is not None
        assert "x" in point_field.fields

    def test_struct_with_array(self) -> None:
        """Structs can contain Arrays."""
        struct = DataTypes.STRUCT(
            fields={
                "name": DataTypes.STRING,
                "values": DataTypes.ARRAY(DataTypes.FLOAT64, shape=5),
            }
        )
        deserialised = DataTypes.model_validate_json(struct.value)
        assert deserialised.fields is not None
        values_field = deserialised.fields["values"]
        assert values_field.base_type.value == "Array"
        assert values_field.shape == (5,)

    def test_struct_with_list(self) -> None:
        """Structs can contain Lists."""
        struct = DataTypes.STRUCT(
            fields={
                "tags": DataTypes.LIST(DataTypes.STRING),
                "count": DataTypes.INT32,
            }
        )
        deserialised = DataTypes.model_validate_json(struct.value)
        assert deserialised.fields is not None
        tags_field = deserialised.fields["tags"]
        assert tags_field.base_type.value == "List"


class TestPolarsConversion:
    """Test conversion to/from Polars DataTypes."""

    @pytest.mark.parametrize(
        ("pl_dtype", "expected_base_type"),
        [
            (pl.String, "String"),
            (pl.Int64, "Int64"),
            (pl.Float64, "Float64"),
            (pl.Boolean, "Boolean"),
        ],
    )
    def test_from_polars_simple(
        self, pl_dtype: type[pl.DataType], expected_base_type: str
    ) -> None:
        """Convert simple Polars types to DataTypes."""
        dt = DataTypes.from_dtype(pl_dtype)
        assert dt.base_type.value == expected_base_type

    def test_from_polars_array(self) -> None:
        """Convert Polars Array to DataTypes."""
        pl_array = pl.Array(pl.Float32, 7)
        dt = DataTypes.from_dtype(pl_array)
        assert dt.base_type.value == "Array"
        assert dt.shape == (7,)
        assert dt.inner is not None
        assert dt.inner.base_type.value == "Float32"

    def test_from_polars_list(self) -> None:
        """Convert Polars List to DataTypes."""
        pl_list = pl.List(pl.Int64)
        dt = DataTypes.from_dtype(pl_list)
        assert dt.base_type.value == "List"
        assert dt.inner is not None
        assert dt.inner.base_type.value == "Int64"

    def test_from_polars_struct(self) -> None:
        """Convert Polars Struct to DataTypes."""
        pl_struct = pl.Struct([pl.Field("name", pl.String), pl.Field("age", pl.Int64)])
        dt = DataTypes.from_dtype(pl_struct)
        assert dt.base_type.value == "Struct"
        assert dt.fields is not None
        assert len(dt.fields) == 2
        assert dt.fields["name"].base_type.value == "String"
        assert dt.fields["age"].base_type.value == "Int64"

    def test_to_polars_list(self) -> None:
        """Convert DataTypes List to Polars."""
        dt = DataTypes.LIST(DataTypes.STRING)
        pl_dtype = dt.to_dtype()
        assert isinstance(pl_dtype, pl.List)

    def test_to_polars_array(self) -> None:
        """Convert DataTypes Array to Polars."""
        dt = DataTypes.ARRAY(DataTypes.INT64, shape=4)
        pl_dtype = dt.to_dtype()
        assert isinstance(pl_dtype, pl.Array)
        assert pl_dtype.shape == (4,)

    def test_to_polars_struct(self) -> None:
        """Convert DataTypes Struct to Polars."""
        dt = DataTypes.STRUCT(fields={"x": DataTypes.FLOAT64, "y": DataTypes.FLOAT64})
        pl_dtype = dt.to_dtype()
        assert isinstance(pl_dtype, pl.Struct)
        assert len(pl_dtype.fields) == 2

    def test_polars_roundtrip_complex(self) -> None:
        """Complex types can round-trip through Polars."""
        original_pl = pl.Struct(
            [
                pl.Field("id", pl.Int64),
                pl.Field("coords", pl.Array(pl.Float64, 2)),
                pl.Field("tags", pl.List(pl.String)),
            ]
        )
        dt = DataTypes.from_dtype(original_pl)
        back_to_pl = dt.to_dtype()

        assert isinstance(back_to_pl, pl.Struct)
        assert len(back_to_pl.fields) == 3


class TestComplexTypes:
    """Test complex nested type structures."""

    def test_list_of_structs(self) -> None:
        """List can contain Structs."""
        struct = DataTypes.STRUCT(fields={"a": DataTypes.INT64})
        list_of_structs = DataTypes.LIST(struct)

        deserialised = DataTypes.model_validate_json(list_of_structs.value)
        assert deserialised.inner is not None
        assert deserialised.inner.base_type.value == "Struct"

    def test_array_of_structs(self) -> None:
        """Array can contain Structs."""
        struct = DataTypes.STRUCT(fields={"value": DataTypes.FLOAT64})
        array_of_structs = DataTypes.ARRAY(struct, shape=3)

        deserialised = DataTypes.model_validate_json(array_of_structs.value)
        assert deserialised.shape == (3,)
        assert deserialised.inner is not None
        assert deserialised.inner.base_type.value == "Struct"

    def test_deeply_nested(self) -> None:
        """Multiple levels of nesting serialise correctly."""
        nested = DataTypes.STRUCT(
            fields={
                "metadata": DataTypes.STRUCT(
                    fields={"id": DataTypes.STRING, "version": DataTypes.INT32}
                ),
                "data": DataTypes.LIST(DataTypes.ARRAY(DataTypes.FLOAT64, shape=10)),
            }
        )

        deserialised = DataTypes.model_validate_json(nested.value)
        assert deserialised.fields is not None
        assert "metadata" in deserialised.fields
        assert "data" in deserialised.fields

        metadata = deserialised.fields["metadata"]
        assert metadata.fields is not None
        assert len(metadata.fields) == 2

        data = deserialised.fields["data"]
        assert data.inner is not None
        assert data.inner.shape == (10,)
