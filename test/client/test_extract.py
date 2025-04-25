import polars as pl
import pyarrow as pa
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine, create_engine, text

from matchbox.client.extract import sql_interface
from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.factories.sources import source_from_tuple


def test_sql_interface(
    sqlite_warehouse: Engine,
    matchbox_api: MockRouter,
):
    # Make dummy data
    foo = source_from_tuple(
        full_name="foo",
        engine=sqlite_warehouse,
        data_pks=[1, 2, 3],
        data_tuple=({"col": 0}, {"col": 1}, {"col": 2}),
    )
    bar = source_from_tuple(
        full_name="bar",
        engine=sqlite_warehouse,
        data_pks=["a", "b", "c"],
        data_tuple=({"col": 10}, {"col": 11}, {"col": 12}),
    )
    different_engine_source = source_from_tuple(
        full_name="baz",
        engine=create_engine("sqlite:///:memory:"),
        data_pks=["a", "b", "c"],
        data_tuple=({"col": 10}, {"col": 11}, {"col": 12}),
    )

    foo.to_warehouse(sqlite_warehouse)
    bar.to_warehouse(sqlite_warehouse)

    # Because of FULL OUTER JOIN, we expect some values to be null, and some explosions
    df_expected = pl.DataFrame(
        [
            {"id": 1, "foo_pk": 1, "bar_pk": "a", "foo_col": 0, "bar_col": 10},
            {"id": 2, "foo_pk": 2, "bar_pk": None, "foo_col": 1, "bar_col": None},
            {"id": 3, "foo_pk": 3, "bar_pk": "b", "foo_col": 2, "bar_col": 11},
            {"id": 3, "foo_pk": 3, "bar_pk": "c", "foo_col": 2, "bar_col": 12},
        ]
    )

    # Mock API
    matchbox_api.get("/sources").mock(
        return_value=Response(
            200,
            json=[
                foo.source.model_dump(),
                bar.source.model_dump(),
                different_engine_source.source.model_dump(),
            ],
        )
    )

    # We don't expect a query call for the third source
    matchbox_api.get("/query", params={"full_name": "foo"}).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"source_pk": "1", "id": 1},
                        {"source_pk": "2", "id": 2},
                        {"source_pk": "3", "id": 3},
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
                        {"source_pk": "a", "id": 1},
                        {"source_pk": "b", "id": 3},
                        {"source_pk": "c", "id": 3},
                    ],
                    schema=SCHEMA_MB_IDS,
                )
            ).read(),
        )
    )

    # Call extracting method
    mapping, column_names, view_definition = sql_interface(
        resolution_name="companies",
        engine=sqlite_warehouse,
        mapping_table="companies_matches",
    )

    # Check column names are as expected
    col_n_df = column_names.to_pandas()
    assert set(col_n_df["table_name"].tolist()) == {"foo", "bar"}
    assert set(col_n_df["column_name"].tolist()) == {"col", "pk"}
    assert set(col_n_df["db_pk"].tolist()) == {"pk"}
    assert set(col_n_df["combined_name"].tolist()) == {
        "foo_col",
        "bar_col",
        "foo_pk",
        "bar_pk",
    }

    # Write results to the warehouse
    with sqlite_warehouse.connect() as conn:
        pl.from_arrow(mapping).write_database("companies_matches", conn)
        pl.from_arrow(column_names).write_database("companies_columns", conn)
        conn.execute(text(f"CREATE VIEW companies AS {view_definition}"))
        conn.commit()

        # Check SQL interface works as intended
        test_query = "SELECT * FROM companies"
        df_actual = pl.read_database(test_query, connection=conn)

        assert_frame_equal(
            df_actual, df_expected, check_row_order=False, check_column_order=False
        )
