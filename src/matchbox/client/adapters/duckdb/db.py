"""Engine, session, and DDL management for the local DuckDB adapter.

Engine and session state live on the adapter instance, not a module-level
global, since a process can hold several independent local stores at once.
"""

from pathlib import Path

from sqlalchemy import Table, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable

from matchbox.client.adapters.duckdb import orm

# (table, PK column) pairs needing a duckdb sequence - no SERIAL/BIGSERIAL.
# Composite-PK and foreign-keyed-PK tables need no sequence, so are absent.
SEQUENCE_COLUMNS: list[tuple[Table, str]] = [
    (orm.Steps.__table__, "step_id"),
    (orm.SourceConfigs.__table__, "source_config_id"),
    (orm.Clusters.__table__, "cluster_id"),
    (orm.ClusterSourceKey.__table__, "key_id"),
    (orm.ModelEdges.__table__, "result_id"),
    (orm.QueryCache.__table__, "cache_id"),
]


def create_local_engine(path: Path | None) -> Engine:
    """Create a duckdb engine, in-memory and ephemeral if no path is given."""
    url = f"duckdb:///{path}" if path is not None else "duckdb:///:memory:"
    return create_engine(url).execution_options(schema_translate_map={"mb": None})


def create_db(engine: Engine, order: list[Table]) -> None:
    """Create tables in order, without FK constraints.

    Duckdb has no ON DELETE CASCADE, so the adapter enforces integrity
    itself and FK constraints are stripped rather than rendered. Sequence
    defaults are created first, then attached once the table exists.
    """
    with engine.begin() as conn:
        for tbl, col_name in SEQUENCE_COLUMNS:
            if tbl in order:
                conn.exec_driver_sql(
                    f'CREATE SEQUENCE IF NOT EXISTS "{tbl.name}_{col_name}_seq"'
                )

        for tbl in order:
            ddl = str(
                CreateTable(tbl, include_foreign_key_constraints=[]).compile(
                    dialect=conn.dialect
                )
            )
            conn.exec_driver_sql(ddl)

        for tbl, col_name in SEQUENCE_COLUMNS:
            if tbl in order:
                conn.exec_driver_sql(
                    f'ALTER TABLE "{tbl.name}" ALTER COLUMN "{col_name}" '
                    f"SET DEFAULT nextval('{tbl.name}_{col_name}_seq')"
                )


def drop_db(engine: Engine, tables: list[Table]) -> None:
    """Drop the given tables, if they exist."""
    with engine.begin() as conn:
        for tbl in tables:
            conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{tbl.name}"')


def drop_dynamic_cache_tables(engine: Engine, prefixes: tuple[str, ...]) -> None:
    """Drop tables named with one of prefixes - RawData/QueryCache's dynamic ones."""
    with engine.begin() as conn:
        names = [
            row[0]
            for row in conn.exec_driver_sql(
                "SELECT table_name FROM duckdb_tables()"
            ).fetchall()
            if row[0].startswith(prefixes)
        ]
        for name in names:
            conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{name}"')
