import polars as pl
import pyarrow as pa
import pytest

from matchbox.common.hash import HashMethod, IntMap, hash_arrow_table, hash_rows


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
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_hash_arrow_table(method: HashMethod):
    a = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    # Field order should not matter
    b = pa.Table.from_pydict(
        {
            "b": [4, 5, 6],
            "a": [1, 2, 3],
        }
    )
    # Row order should not matter
    c = pa.Table.from_pydict(
        {
            "a": [3, 2, 1],
            "b": [6, 5, 4],
        }
    )
    # Field and row order should not matter
    d = pa.Table.from_pydict(
        {
            "b": [6, 5, 4],
            "a": [3, 2, 1],
        }
    )
    # Empty table should have a different hash
    e = pa.Table.from_pydict(
        {
            "a": [],
            "b": [],
        }
    )
    # Different row data should have a different hash
    f = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 7],
        }
    )
    # If field name change their order, the hash should change
    g = pa.Table.from_pydict(
        {
            "b": [1, 2, 3],
            "a": [4, 5, 6],
        }
    )
    # List fields are handled
    h = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [[1, 2], [3, 4], [5, 6]],
        }
    )
    # List order doesn't matter
    i = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [[2, 1], [4, 3], [6, 5]],
        }
    )
    # Binary fields are handled, including non-UTF-8 bytes
    j = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [b"abc", None, bytes([255, 254, 253])],
        }
    )

    h_a = hash_arrow_table(a, method=method)
    h_a1 = hash_arrow_table(a, method=method)
    h_b = hash_arrow_table(b, method=method)
    h_c = hash_arrow_table(c, method=method)
    h_d = hash_arrow_table(d, method=method)
    h_e = hash_arrow_table(e, method=method)
    h_f = hash_arrow_table(f, method=method)
    h_g = hash_arrow_table(g, method=method)
    h_h = hash_arrow_table(h, method=method)
    h_i = hash_arrow_table(i, method=method)
    h_j = hash_arrow_table(j, method=method)

    # Basic type check
    assert isinstance(h_a, bytes)
    # Basic invariance checks
    assert h_a == h_a1 == h_b == h_c == h_d
    # Different data = different hashes
    assert h_a != h_e
    assert h_a != h_f
    assert h_a != h_g
    assert h_a != h_j
    # List type table should be consistent regardless of field order
    assert h_h == h_i


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_struct_json_hashing(method: HashMethod):
    """Test that struct/JSON data can be properly hashed."""

    # Basic struct test
    a = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )

    assert isinstance(pl.from_arrow(a)["metadata"].dtype, pl.Struct)

    # Same data but different struct serialization
    b = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"age": 30, "name": "Alice"},
                {"age": 25, "name": "Bob"},
                {"age": 35, "name": "Charlie"},
            ],
        }
    )

    # Different data in structs
    c = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "age": 31},  # Changed age
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )

    # Nested structs
    d = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "details": {"city": "New York", "active": True}},
                {"name": "Bob", "details": {"city": "Boston", "active": False}},
                {"name": "Charlie", "details": {"city": "Chicago", "active": True}},
            ],
        }
    )

    # Test basic struct hashing
    h_a = hash_arrow_table(a, method=method)
    h_a1 = hash_arrow_table(a, method=method)
    h_b = hash_arrow_table(b, method=method)
    h_c = hash_arrow_table(c, method=method)
    h_d = hash_arrow_table(d, method=method)

    # Basic type check
    assert isinstance(h_a, bytes)

    # Basic equality check
    assert h_a == h_a1 == h_b
    # Difference checks
    assert h_a != h_c
    assert h_a != h_d
