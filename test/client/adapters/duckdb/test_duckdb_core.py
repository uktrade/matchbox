import pyarrow as pa
from sqlalchemy import Engine

from matchbox.client.adapters.duckdb import MatchboxLocalDuckDB
from matchbox.common.factories.scenarios import setup_scenario


def test_drop_leaves_no_dynamic_tables(
    matchbox_local_duckdb: MatchboxLocalDuckDB, sqla_sqlite_warehouse: Engine
) -> None:
    """drop() sweeps RawData/QueryCache's dynamically named physical tables."""
    with setup_scenario(
        matchbox_local_duckdb, "index", warehouse=sqla_sqlite_warehouse
    ) as dag_testkit:
        crn = dag_testkit.sources["crn"].path
        matchbox_local_duckdb.insert_raw_data(
            crn, pa.table({"key": ["k1"], "name": ["Alice"]})
        )
        matchbox_local_duckdb.cache_query(
            "key1", pa.table({"id": [1]}), depends_on=[crn]
        )

        matchbox_local_duckdb.drop(certain=True)

        with matchbox_local_duckdb._engine.connect() as conn:
            names = {
                row[0]
                for row in conn.exec_driver_sql(
                    "SELECT table_name FROM duckdb_tables()"
                ).fetchall()
            }
        prefixes = ("mb_raw_data__", "mb_query_cache__")
        assert not any(n.startswith(prefixes) for n in names)


def test_dump_restore_round_trips_dynamic_tables(
    matchbox_local_duckdb: MatchboxLocalDuckDB, sqla_sqlite_warehouse: Engine
) -> None:
    """dump()/restore() carry RawData/QueryCache row content, not just pointers."""
    with setup_scenario(
        matchbox_local_duckdb, "index", warehouse=sqla_sqlite_warehouse
    ) as dag_testkit:
        crn = dag_testkit.sources["crn"].path
        matchbox_local_duckdb.insert_raw_data(
            crn, pa.table({"key": ["k1"], "name": ["Alice"]})
        )
        matchbox_local_duckdb.cache_query(
            "key1", pa.table({"id": [1]}), depends_on=[crn]
        )
        snapshot = matchbox_local_duckdb.dump()

    matchbox_local_duckdb.restore(snapshot)

    assert matchbox_local_duckdb.get_raw_data(crn).to_pylist() == [
        {"key": "k1", "name": "Alice"}
    ]
    assert matchbox_local_duckdb.get_cached_query("key1").to_pylist() == [{"id": 1}]
