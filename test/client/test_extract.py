import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine, create_engine

from matchbox.client.extract import primary_keys_map
from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.exceptions import MatchboxSourceNotFoundError
from matchbox.common.factories.sources import source_from_tuple


def test_primary_keys_map(
    sqlite_warehouse: Engine,
    matchbox_api: MockRouter,
):
    # Make dummy data
    foo = source_from_tuple(
        full_name="foo",
        engine=sqlite_warehouse,
        data_identifiers=[1, 2, 3],
        data_tuple=({"col": 0}, {"col": 1}, {"col": 2}),
    )
    bar = source_from_tuple(
        full_name="bar",
        engine=create_engine("sqlite:///:memory:"),
        data_identifiers=["a", "b", "c"],
        data_tuple=({"col": 10}, {"col": 11}, {"col": 12}),
    )

    foo.to_warehouse(sqlite_warehouse)
    bar.to_warehouse(sqlite_warehouse)

    # Because of FULL OUTER JOIN, we expect some values to be null, and some explosions
    expected_foo_bar_mapping = pl.DataFrame(
        [
            {"id": 1, "foo_identifier": "1", "bar_identifier": "a"},
            {"id": 2, "foo_identifier": "2", "bar_identifier": None},
            {"id": 3, "foo_identifier": "3", "bar_identifier": "b"},
            {"id": 3, "foo_identifier": "3", "bar_identifier": "c"},
        ]
    )

    # When selecting single source, we won't explode
    expected_foo_mapping = expected_foo_bar_mapping.select(
        ["id", "foo_identifier"]
    ).unique()

    # Mock API
    matchbox_api.get("/sources").mock(
        return_value=Response(
            200,
            json=[
                foo.config.model_dump(),
                bar.config.model_dump(),
            ],
        )
    )

    matchbox_api.get("/query", params={"full_name": "foo"}).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"source_identifier": "1", "id": 1},
                        {"source_identifier": "2", "id": 2},
                        {"source_identifier": "3", "id": 3},
                    ],
                    schema=SCHEMA_MB_IDS,
                )
            ).read(),
        )
    )

    matchbox_api.get("/query", params={"full_name": "bar"}).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"source_identifier": "a", "id": 1},
                        {"source_identifier": "b", "id": 3},
                        {"source_identifier": "c", "id": 3},
                    ],
                    schema=SCHEMA_MB_IDS,
                )
            ).read(),
        )
    )

    # Case 0: no sources are found
    with pytest.raises(MatchboxSourceNotFoundError):
        primary_keys_map(resolution="companies", location_match="postgresql://")

    # Case 1: apply engine filter, and retrieve single table
    foo_mapping = primary_keys_map(
        resolution="companies", location_match=str(sqlite_warehouse.url)
    )

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # Case 2: without engine filter, and retrieve multiple tables
    foo_bar_mapping = primary_keys_map(resolution="companies")

    assert_frame_equal(
        pl.from_arrow(foo_bar_mapping),
        expected_foo_bar_mapping,
        check_row_order=False,
        check_column_order=False,
    )
