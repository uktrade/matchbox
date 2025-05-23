"""General utilities for the PostgreSQL backend."""

import base64
import contextlib
import cProfile
import io
import pstats
import uuid
from typing import Generator

import pyarrow as pa
from adbc_driver_manager import ProgrammingError as ADBCProgrammingError
from adbc_driver_postgresql import dbapi as adbc_dbapi
from pyarrow import Table as ArrowTable
from sqlalchemy import Column, MetaData, Table, func, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.sql import Select

from matchbox.common.exceptions import (
    MatchboxDatabaseWriteError,
)
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeType,
)
from matchbox.server.base import MatchboxBackends, MatchboxSnapshot
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ResolutionFrom,
    Resolutions,
)

# Retrieval


def get_resolution_graph() -> ResolutionGraph:
    """Retrieves the resolution graph."""
    G = ResolutionGraph(nodes=set(), edges=set())
    with MBDB.get_session() as session:
        for resolution in session.query(Resolutions).all():
            G.nodes.add(
                ResolutionNode(
                    id=resolution.resolution_id,
                    name=resolution.name,
                    type=ResolutionNodeType(resolution.type),
                )
            )

        for edge in (
            session.query(ResolutionFrom).filter(ResolutionFrom.level == 1).all()
        ):
            G.edges.add(ResolutionEdge(parent=edge.parent, child=edge.child))

    return G


# Data management


def dump() -> MatchboxSnapshot:
    """Dumps the entire database to a snapshot.

    Returns:
        A MatchboxSnapshot object of type "postgres" with the database's
            current state.
    """
    data = {}

    with MBDB.get_session() as session:
        for table in MBDB.sorted_tables:
            # Query all records from this table
            records = session.execute(select(table)).mappings().all()

            # Convert each record to a dictionary
            table_data = []
            for record in records:
                record_dict = dict(record)
                for k, v in record_dict.items():
                    # Store bytes as nested dictionary with encoding format
                    if isinstance(v, bytes):
                        record_dict[k] = {"base64": base64.b64encode(v).decode("ascii")}

                table_data.append(record_dict)

            data[table.name] = table_data

    return MatchboxSnapshot(backend_type=MatchboxBackends.POSTGRES, data=data)


def restore(snapshot: MatchboxSnapshot, batch_size: int) -> None:
    """Restores the database from a snapshot.

    Args:
        snapshot: A MatchboxSnapshot object of type "postgres" with the
            database's state
        batch_size: The number of records to insert in each batch

    Raises:
        ValueError: If the snapshot is missing data
    """
    with MBDB.get_session() as session:
        # Process tables in order
        for table in MBDB.sorted_tables:
            if table.name not in snapshot.data:
                raise ValueError(f"Invalid: Table {table.name} not found in snapshot.")

            records = snapshot.data[table.name]

            if not records:
                continue

            # Process records for insertion
            processed_records = []
            for record in records:
                processed_record = {}

                for key, value in record.items():
                    # Check if the value is a dictionary with encoding format
                    if isinstance(value, dict) and "base64" in value:
                        processed_record[key] = base64.b64decode(value["base64"])
                    else:
                        processed_record[key] = value

                processed_records.append(processed_record)

            # Insert
            for i in range(0, len(processed_records), batch_size):
                batch = processed_records[i : i + batch_size]
                session.execute(insert(table), batch)
                session.flush()

            # Re-sync primary key sequences
            for pk in table.primary_key:
                lock_stmt = text(
                    f"lock table {table.schema}.{table.name} in exclusive mode;"
                )
                sync_statement = select(
                    func.setval(
                        func.pg_get_serial_sequence(
                            text(f"'{table.schema}.{table.name}'"),
                            text(f"'{pk.name}'"),
                        ),
                        func.coalesce(func.max(text(pk.name)), 0),
                    )
                ).select_from(text(f"{table.schema}.{table.name}"))

                session.execute(lock_stmt)
                session.execute(sync_statement)

        session.commit()


# SQLAlchemy profiling


@contextlib.contextmanager
def sqa_profiled():
    """SQLAlchemy profiler.

    Taken directly from their docs:
    https://docs.sqlalchemy.org/en/20/faq/performance.html#query-profiling
    """
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    print(s.getvalue())


# Misc


def compile_sql(stmt: Select) -> str:
    """Compiles a SQLAlchemy statement into a string.

    Args:
        stmt: The SQLAlchemy statement to compile.

    Returns:
        The compiled SQL statement as a string.
    """
    return str(
        stmt.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def _copy_to_table(
    table_name: str,
    schema_name: str,
    data: ArrowTable,
    connection: adbc_dbapi.Connection,
    max_chunksize: int | None = None,
):
    batch_reader = pa.RecordBatchReader.from_batches(
        data.schema, data.to_batches(max_chunksize=max_chunksize)
    )

    with connection.cursor() as cursor:
        cursor.adbc_ingest(
            table_name=table_name,
            data=batch_reader,
            mode="append",
            db_schema_name=schema_name,
        )
        connection.commit()


def large_ingest(
    data: pa.Table,
    table_class: DeclarativeMeta,
    max_chunksize: int | None = None,
    upsert_keys: list[str] | None = None,
    update_columns: list[str] | None = None,
):
    """Append a PyArrow table to a PostgreSQL table using ADBC.

    It will either copy directly (and error if primary key constraints are violated),
    or it can be run in upsert mode by using a staging table, which is slower.

    Args:
        data: A PyArrow table to write.
        table_class: The SQLAlchemy ORM class for the table to write to.
        max_chunksize: Size of data chunks to be read and copied.
        upsert_keys: Columns used as keys for "on conflict do update".
            If passed, it will run ingest in slower upsert mode.
            If not passed and `update_columns` is passed, defaults to primary keys.
        update_columns: Columns to update when upserting.
            If passed, it will run ingest in slower upsert mode.
            If not passed and `upsert_keys` is passed, defaults to all other columns.
    """
    with (
        MBDB.get_adbc_connection() as conn,
        MBDB.get_session() as session,
    ):
        table: Table = table_class.__table__
        metadata = table.metadata

        table_columns = [c.name for c in table.columns]
        col_diff = set(data.column_names) - set(table_columns)
        if len(col_diff) > 0:
            raise ValueError(f"Table {table.name} does not have columns {col_diff}")

        if not update_columns and not upsert_keys:
            try:
                _copy_to_table(
                    table_name=table.name,
                    schema_name=table.schema,
                    data=data,
                    connection=conn,
                    max_chunksize=max_chunksize,
                )
            except ADBCProgrammingError as e:
                raise MatchboxDatabaseWriteError from e

        # Upsert mode (slower)
        else:
            keys_names = [c.name for c in table.primary_key.columns]

            # Validate upsert arguments
            if len(set(update_columns or []) & set(upsert_keys or [])) > 0:
                raise ValueError("Cannot update a custom upsert key")

            if len(set(update_columns or []) & set(keys_names)) > 0:
                raise ValueError(
                    "Cannot update a primary key without "
                    "setting a different custom upsert key"
                )

            # If necessary, set defaults for upsert variables
            upsert_keys = upsert_keys or keys_names

            if not update_columns:
                update_columns = [c for c in table_columns if c not in upsert_keys]
            try:
                # Create temp table
                temp_table_name = f"{table.name}_tmp_{uuid.uuid4().hex}"
                temp_cols = [
                    Column(c.name, c.type, primary_key=c.primary_key)
                    for c in table.columns
                ]
                temp_table = Table(temp_table_name, metadata, *temp_cols)
                temp_table.create(session.bind)

                # Add new records to temp table
                _copy_to_table(
                    table_name=temp_table_name,
                    schema_name=table.schema,
                    data=data,
                    connection=conn,
                    max_chunksize=max_chunksize,
                )

                # Copy new records to original table
                insert_stmt = insert(table).from_select(
                    [c.name for c in temp_table.columns], temp_table.select()
                )
                upsert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=upsert_keys,
                    set_={c: getattr(insert_stmt.excluded, c) for c in update_columns},
                )

                session.execute(upsert_stmt)
                session.commit()

            except (ADBCProgrammingError, IntegrityError) as e:
                raise MatchboxDatabaseWriteError from e

            finally:
                # Drop temp table
                temp_table.drop(session.bind, checkfirst=True)


@contextlib.contextmanager
def ingest_to_temporary_table(
    table_name: str,
    schema_name: str,
    data: ArrowTable,
    max_chunksize: int | None = None,
) -> Generator[Table, None, None]:
    """Context manager to ingest Arrow data to a temporary table.

    Let ADBC create the table based on the PyArrow schema,
    ingests the data, and drops the table automatically on exit.

    Args:
        table_name: Base name for the temporary table
        schema_name: Schema where the temporary table will be created
        data: PyArrow table containing the data to ingest
        max_chunksize: Optional maximum chunk size for batches

    Returns:
        A SQLAlchemy Table object representing the temporary table
    """
    with (
        MBDB.get_adbc_connection() as conn,
        MBDB.get_session() as session,
    ):
        temp_table_name = f"{table_name}_tmp_{uuid.uuid4().hex}"

        try:
            # Create the table with ADBC
            batch_reader = pa.RecordBatchReader.from_batches(
                data.schema, data.to_batches(max_chunksize=max_chunksize)
            )

            with conn.cursor() as cursor:
                cursor.adbc_ingest(
                    table_name=temp_table_name,
                    data=batch_reader,
                    mode="create",
                    db_schema_name=schema_name,
                )
                conn.commit()

            # Reflect the table that ADBC created
            metadata = MetaData()
            temp_table = Table(
                temp_table_name,
                metadata,
                schema=schema_name,
                autoload_with=session.bind,
            )

            # Yield the temporary table
            yield temp_table

        finally:
            # Drop the table when done
            with conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS {schema_name}.{temp_table_name}")
                conn.commit()
