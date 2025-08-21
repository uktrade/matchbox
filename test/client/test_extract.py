import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine, create_engine

from matchbox.client.extract import key_field_map
from matchbox.common.arrow import SCHEMA_QUERY, table_to_buffer
from matchbox.common.exceptions import MatchboxSourceNotFoundError
from matchbox.common.factories.sources import source_from_tuple


def test_key_field_map(
    sqlite_warehouse: Engine,
    matchbox_api: MockRouter,
):
    # Make dummy data
    sqlite_memory_warehouse = create_engine("sqlite:///:memory:")

    foo = source_from_tuple(
        name="foo",
        location_name="sqlite",
        engine=sqlite_warehouse,
        data_keys=[1, 2, 3],
        data_tuple=({"col": 0}, {"col": 1}, {"col": 2}),
    )
    bar = source_from_tuple(
        name="bar",
        location_name="sqlite_memory",
        engine=sqlite_memory_warehouse,
        data_keys=["a", "b", "c"],
        data_tuple=({"col": 10}, {"col": 11}, {"col": 12}),
    )

    foo.write_to_location(client=sqlite_warehouse, set_client=True)
    bar.write_to_location(client=sqlite_memory_warehouse, set_client=True)

    # Because of FULL OUTER JOIN, we expect some values to be null, and some explosions
    expected_foo_bar_mapping = pl.DataFrame(
        [
            {"id": 1, "foo_key": "1", "bar_key": "a"},
            {"id": 2, "foo_key": "2", "bar_key": None},
            {"id": 3, "foo_key": "3", "bar_key": "b"},
            {"id": 3, "foo_key": "3", "bar_key": "c"},
        ]
    )

    # When selecting single source, we won't explode
    expected_foo_mapping = expected_foo_bar_mapping.select(["id", "foo_key"]).unique()

    # Mock API
    matchbox_api.get("/sources").mock(
        return_value=Response(
            200,
            json=[
                foo.source_config.model_dump(mode="json"),
                bar.source_config.model_dump(mode="json"),
            ],
        )
    )

    # Create mock table for foo source (for single-source queries)
    indices_foo = pa.array([0, 0, 0], type=pa.uint32())
    dictionary_foo = pa.array(["foo"], type=pa.large_string())
    source_dict_foo = pa.DictionaryArray.from_arrays(indices_foo, dictionary_foo)

    foo_table = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(["1", "2", "3"], type=pa.large_string()),
            source_dict_foo,
        ],
        schema=SCHEMA_QUERY,
    )

    matchbox_api.get("/query", params={"sources": ["foo"]}).mock(
        return_value=Response(
            200,
            content=table_to_buffer(foo_table).read(),
        )
    )

    # Create combined mock table for multi-source queries
    combined_indices = pa.array([0, 0, 0, 1, 1, 1], type=pa.uint32())
    combined_dictionary = pa.array(["foo", "bar"], type=pa.large_string())
    combined_source_dict = pa.DictionaryArray.from_arrays(
        combined_indices, combined_dictionary
    )

    combined_table = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3, 1, 3, 3], type=pa.int64()),
            pa.array(["1", "2", "3", "a", "b", "c"], type=pa.large_string()),
            combined_source_dict,
        ],
        schema=SCHEMA_QUERY,
    )

    # Mock for multi-source query (no filter or both sources)
    matchbox_api.get("/query", params={"sources": ["foo", "bar"]}).mock(
        return_value=Response(
            200,
            content=table_to_buffer(combined_table).read(),
        )
    )

    # Case 0: No sources are found
    with pytest.raises(MatchboxSourceNotFoundError):
        key_field_map(resolution="companies", source_filter=["nonexistent"])

    with pytest.raises(MatchboxSourceNotFoundError):
        key_field_map(resolution="companies", location_names=["nonexistent"])

    # Case 1: Retrieve single table
    # With URI filter
    foo_mapping = key_field_map(resolution="companies", location_names=["sqlite"])

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # With source filter
    foo_mapping = key_field_map(resolution="companies", source_filter=["foo"])

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # With both filters
    foo_mapping = key_field_map(
        resolution="companies",
        source_filter=["foo"],
        location_names="sqlite",
    )

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # Case 2: Retrieve multiple tables
    # With no filter
    foo_bar_mapping = key_field_map(resolution="companies")

    assert_frame_equal(
        pl.from_arrow(foo_bar_mapping),
        expected_foo_bar_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # With source filter
    foo_bar_mapping = key_field_map(
        resolution="companies", source_filter=["foo", "bar"]
    )

    assert_frame_equal(
        pl.from_arrow(foo_bar_mapping),
        expected_foo_bar_mapping,
        check_row_order=False,
        check_column_order=False,
    )
