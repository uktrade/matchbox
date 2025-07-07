import polars as pl
import pyarrow as pa
import pytest

from matchbox.common.hash import (
    HashMethod,
    IntMap,
    hash_arrow_table,
    hash_rows,
)


def test_intmap_basic():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    assert len({a, b, c}) == 3
    assert max(a, b, c) < 0


def test_intmap_same():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    im2 = IntMap(salt=10)
    x = im2.index(1, 2)
    y = im2.index(3, 4)
    z = im2.index(a, b)

    assert (a, b, c) == (x, y, z)


def test_intmap_different():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    im2 = IntMap(salt=11)
    x = im2.index(1, 2)
    y = im2.index(3, 4)
    z = im2.index(a, b)

    for v1, v2 in zip([a, b, c], [x, y, z], strict=True):
        assert v1 != v2


def test_intmap_unordered():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2, 3)
    b = im1.index(3, 1, 2)

    assert a == b


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_hash_rows(method: HashMethod):
    data = pl.DataFrame(
        {
            "string_col": ["abc", "def", "ghi"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "struct_col": [{"a": 1, "b": "x"}, {"a": 2, "b": None}, {"a": 3, "b": "z"}],
            "binary_col": [b"data1", b"data2", b"data3"],
        }
    )

    assert isinstance(data["string_col"].dtype, pl.String)
    assert isinstance(data["int_col"].dtype, pl.Int64)
    assert isinstance(data["float_col"].dtype, pl.Float64)
    assert isinstance(data["struct_col"].dtype, pl.Struct)
    assert isinstance(data["binary_col"].dtype, pl.Binary)

    hash_rows(data, columns=data.columns, method=method)


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
class TestArrowTableHashing:
    """Test basic Arrow table hashing with order invariance."""

    @classmethod
    def setup_class(cls):
        """Set up test data for basic hashing tests."""
        cls.tables = {
            "original": pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "field_reordered": pa.Table.from_pydict({"b": [4, 5, 6], "a": [1, 2, 3]}),
            "row_reordered": pa.Table.from_pydict({"a": [3, 2, 1], "b": [6, 5, 4]}),
            "both_reordered": pa.Table.from_pydict({"b": [6, 5, 4], "a": [3, 2, 1]}),
            "empty": pa.Table.from_pydict({"a": [], "b": []}),
            "different_data": pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 7]}),
            "content_swapped": pa.Table.from_pydict({"b": [1, 2, 3], "a": [4, 5, 6]}),
        }

    def test_order_invariance(self, method: HashMethod):
        """Test that field and row order don't affect hash values."""
        # Calculate all hashes
        hashes = {
            name: hash_arrow_table(table, method=method)
            for name, table in self.tables.items()
        }

        # All hashes should be bytes
        assert all(isinstance(h, bytes) for h in hashes.values())

        # These should all hash identically despite different ordering
        equivalent_tables = [
            "original",
            "field_reordered",
            "row_reordered",
            "both_reordered",
        ]
        equivalent_hashes = [hashes[name] for name in equivalent_tables]
        assert all(h == equivalent_hashes[0] for h in equivalent_hashes), (
            "Tables with same data but different ordering should hash identically"
        )

        # These should all hash differently
        different_tables = ["empty", "different_data", "content_swapped"]
        for table_name in different_tables:
            assert hashes["original"] != hashes[table_name], (
                f"Table '{table_name}' should hash differently from original"
            )

    def test_consistency_across_calls(self, method: HashMethod):
        """Test that identical tables always produce the same hash."""
        table = self.tables["original"]

        hash1 = hash_arrow_table(table, method=method)
        hash2 = hash_arrow_table(table, method=method)

        assert hash1 == hash2, "Identical calls should produce identical hashes"

    def test_list_fields_order_invariance(self, method: HashMethod):
        """Test that order within list fields doesn't affect hash."""
        table_ordered = pa.Table.from_pydict(
            {
                "a": [1, 2, 3],
                "b": [[1, 2], [3, 4], [5, 6]],
            }
        )
        table_list_reordered = pa.Table.from_pydict(
            {
                "a": [1, 2, 3],
                "b": [[2, 1], [4, 3], [6, 5]],  # List elements reordered
            }
        )

        hash_ordered = hash_arrow_table(table_ordered, method=method)
        hash_reordered = hash_arrow_table(table_list_reordered, method=method)

        assert hash_ordered == hash_reordered, (
            "Order of elements within list fields should not affect hash"
        )

    def test_binary_fields_handling(self, method: HashMethod):
        """Test that binary fields including non-UTF-8 bytes are handled correctly."""
        table_with_binary = pa.Table.from_pydict(
            {
                "a": [1, 2, 3],
                "b": [b"abc", None, bytes([255, 254, 253])],  # Include non-UTF-8 bytes
            }
        )

        hash_result = hash_arrow_table(table_with_binary, method=method)
        assert isinstance(hash_result, bytes), "Should successfully hash binary data"


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
class TestStructHashing:
    """Test hashing of struct/JSON data within Arrow tables."""

    @classmethod
    def setup_class(cls):
        """Set up test data for struct hashing tests."""
        cls.tables = {
            "basic_struct": pa.Table.from_pydict(
                {
                    "id": [1, 2, 3],
                    "metadata": [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 25},
                        {"name": "Charlie", "age": 35},
                    ],
                }
            ),
            "struct_fields_reordered": pa.Table.from_pydict(
                {
                    "id": [1, 2, 3],
                    "metadata": [
                        {"age": 30, "name": "Alice"},
                        {"age": 25, "name": "Bob"},
                        {"age": 35, "name": "Charlie"},
                    ],
                }
            ),
            "struct_data_changed": pa.Table.from_pydict(
                {
                    "id": [1, 2, 3],
                    "metadata": [
                        {"name": "Alice", "age": 31},  # Changed age
                        {"name": "Bob", "age": 25},
                        {"name": "Charlie", "age": 35},
                    ],
                }
            ),
            "nested_struct": pa.Table.from_pydict(
                {
                    "id": [1, 2, 3],
                    "metadata": [
                        {
                            "name": "Alice",
                            "details": {"city": "New York", "active": True},
                        },
                        {"name": "Bob", "details": {"city": "Boston", "active": False}},
                        {
                            "name": "Charlie",
                            "details": {"city": "Chicago", "active": True},
                        },
                    ],
                }
            ),
        }

    def test_struct_field_order_invariance(self, method: HashMethod):
        """Test that order of fields within structs doesn't affect hash."""
        table_basic = self.tables["basic_struct"]
        table_reordered = self.tables["struct_fields_reordered"]

        # Verify we're actually testing struct data
        df = pl.from_arrow(table_basic)
        assert isinstance(df["metadata"].dtype, pl.Struct), (
            "Test should be working with struct data type"
        )

        hash_basic = hash_arrow_table(table_basic, method=method)
        hash_reordered = hash_arrow_table(table_reordered, method=method)

        assert hash_basic == hash_reordered, (
            "Struct field order should not affect hash value"
        )

    def test_struct_content_sensitivity(self, method: HashMethod):
        """Test that changes in struct content produce different hashes."""
        hash_basic = hash_arrow_table(self.tables["basic_struct"], method=method)
        hash_changed = hash_arrow_table(
            self.tables["struct_data_changed"], method=method
        )
        hash_nested = hash_arrow_table(self.tables["nested_struct"], method=method)

        assert hash_basic != hash_changed, (
            "Changed struct content should produce different hash"
        )
        assert hash_basic != hash_nested, (
            "Different struct structure should produce different hash"
        )

    def test_nested_struct_consistency(self, method: HashMethod):
        """Test that nested structs are handled consistently."""
        table = self.tables["nested_struct"]

        hash1 = hash_arrow_table(table, method=method)
        hash2 = hash_arrow_table(table, method=method)

        assert hash1 == hash2, "Nested structs should hash consistently"


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
class TestNormalisation:
    """Test ID normalisation functionality in Arrow table hashing."""

    @classmethod
    def setup_class(cls):
        """Set up test data for normalisation tests."""
        cls.tables = {
            "original": pa.Table.from_pydict(
                {
                    "left_id": [1, 2, 3],
                    "right_id": [4, 5, 6],
                    "probability": [0.8, 0.9, 0.7],
                }
            ),
            "ids_swapped": pa.Table.from_pydict(
                {
                    "left_id": [4, 5, 6],
                    "right_id": [1, 2, 3],
                    "probability": [0.8, 0.9, 0.7],
                }
            ),
            "data_changed": pa.Table.from_pydict(
                {
                    "left_id": [1, 2, 3],
                    "right_id": [4, 5, 6],
                    "probability": [0.8, 0.9, 0.8],  # Changed last probability
                }
            ),
            "rows_reordered": pa.Table.from_pydict(
                {
                    "left_id": [2, 1, 3],
                    "right_id": [5, 4, 6],
                    "probability": [0.9, 0.8, 0.7],
                }
            ),
            "with_nulls_a": pa.Table.from_pydict(
                {
                    "left_id": [1, None, 3],
                    "right_id": [None, 5, 6],
                    "probability": [0.8, 0.9, 0.7],
                }
            ),
            "with_nulls_b": pa.Table.from_pydict(
                {
                    "left_id": [None, 5, 6],
                    "right_id": [1, None, 3],
                    "probability": [0.8, 0.9, 0.7],
                }
            ),
            "with_duplicates": pa.Table.from_pydict(
                {
                    "left_id": [1, 1, 2],
                    "right_id": [1, 2, 1],
                    "probability": [0.8, 0.9, 0.7],
                }
            ),
            "empty": pa.Table.from_pydict(
                {
                    "left_id": [],
                    "right_id": [],
                    "probability": [],
                }
            ),
        }

    def test_normalisation_disabled_by_default(self, method: HashMethod):
        """Test that without normalisation, ID order affects hash."""
        table_original = self.tables["original"]
        table_swapped = self.tables["ids_swapped"]

        hash_original = hash_arrow_table(table_original, method=method)
        hash_swapped = hash_arrow_table(table_swapped, method=method)

        assert hash_original != hash_swapped, (
            "Without normalisation, swapped IDs should produce different hashes"
        )

    def test_normalisation_makes_id_order_irrelevant(self, method: HashMethod):
        """Test that normalisation makes ID column order irrelevant."""
        sorted_list_cols = ["left_id", "right_id"]

        tables_to_test = ["original", "ids_swapped", "rows_reordered"]
        hashes = []

        for table_name in tables_to_test:
            table = self.tables[table_name]
            hash_val = hash_arrow_table(
                table, method=method, as_sorted_list=sorted_list_cols
            )
            hashes.append(hash_val)

        # All should be equal despite different ID arrangements
        assert all(h == hashes[0] for h in hashes), (
            "Normalisation should make ID order irrelevant"
        )

        # But different data should still produce different hash
        hash_different = hash_arrow_table(
            self.tables["data_changed"],
            method=method,
            as_sorted_list=sorted_list_cols,
        )
        assert hashes[0] != hash_different, (
            "Different data should still produce different hash even with normalisation"
        )

    def test_normalisation_with_multiple_columns(self, method: HashMethod):
        """Test normalisation across more than two columns."""
        table_abc = pa.Table.from_pydict(
            {
                "person_a": [1, 2, 3],
                "person_b": [4, 5, 6],
                "person_c": [7, 8, 9],
                "score": [0.8, 0.9, 0.7],
            }
        )

        # Same relationships but people in different columns
        table_cab = pa.Table.from_pydict(
            {
                "person_a": [7, 8, 9],  # person_c values
                "person_b": [1, 2, 3],  # person_a values
                "person_c": [4, 5, 6],  # person_b values
                "score": [0.8, 0.9, 0.7],
            }
        )

        sorted_list_cols = ["person_a", "person_b", "person_c"]
        hash_abc = hash_arrow_table(
            table_abc, method=method, as_sorted_list=sorted_list_cols
        )
        hash_cab = hash_arrow_table(
            table_cab, method=method, as_sorted_list=sorted_list_cols
        )

        assert hash_abc == hash_cab, (
            "Multi-column normalisation should handle arbitrary column reordering"
        )

    def test_normalisation_handles_nulls(self, method: HashMethod):
        """Test that normalisation works correctly with null values."""
        sorted_list_cols = ["left_id", "right_id"]

        hash_a = hash_arrow_table(
            self.tables["with_nulls_a"],
            method=method,
            as_sorted_list=sorted_list_cols,
        )
        hash_b = hash_arrow_table(
            self.tables["with_nulls_b"],
            method=method,
            as_sorted_list=sorted_list_cols,
        )

        assert hash_a == hash_b, "Normalisation should handle null values correctly"

    def test_normalisation_edge_cases(self, method: HashMethod):
        """Test normalisation with edge cases like duplicates and empty tables."""
        sorted_list_cols = ["left_id", "right_id"]

        # Test with duplicate values
        table_dupes = self.tables["with_duplicates"]
        hash_dupes = hash_arrow_table(
            table_dupes, method=method, as_sorted_list=sorted_list_cols
        )
        assert isinstance(hash_dupes, bytes), (
            "Should handle duplicate values in sorted_listd columns"
        )

        # Test with empty table
        table_empty = self.tables["empty"]
        hash_empty = hash_arrow_table(
            table_empty, method=method, as_sorted_list=sorted_list_cols
        )
        assert hash_empty == b"empty_table_hash", (
            "Empty tables should return consistent special hash value"
        )
