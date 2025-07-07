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
from adbc_driver_postgresql.dbapi import Connection as ADBCConnection
from pyarrow import Table as ArrowTable
from sqlalchemy import Column, MetaData, Table, func, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.pool import PoolProxiedConnection
from sqlalchemy.sql import Select
from sqlalchemy.sql.type_api import TypeEngine

from matchbox.common.exceptions import (
    MatchboxDatabaseWriteError,
)
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeType,
)
from matchbox.common.logging import logger
from matchbox.server.base import MatchboxBackends, MatchboxSnapshot
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import ResolutionFrom, Resolutions

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
    connection: ADBCConnection,
    data: ArrowTable,
    max_chunksize: int | None = None,
):
    """Copy data to table using ADBC with isolated connection."""
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


def large_append(
    data: pa.Table,
    table_class: DeclarativeMeta,
    adbc_connection: PoolProxiedConnection,
    max_chunksize: int | None = None,
):
    """Append a PyArrow table to a PostgreSQL table using ADBC.

    This function does not support upserting and will error if keys clash.
    This method does not auto-commit, which is the responsibility of the caller.

    Args:
        data: A PyArrow table to write.
        table_class: The SQLAlchemy ORM class for the table to write to.
        adbc_connection: An ADBC connection from the pool.
            This is returned by MBDB.get_adbc_connection() and needs to be used via a
            context manager.
        max_chunksize: Size of data chunks to be read and copied.
    """
    table: Table = table_class.__table__

    table_columns = [c.name for c in table.columns]
    col_diff = set(data.column_names) - set(table_columns)
    if len(col_diff) > 0:
        raise ValueError(f"Table {table.name} does not have columns {col_diff}")
    try:
        _copy_to_table(
            table_name=table.name,
            schema_name=table.schema,
            connection=adbc_connection,
            data=data,
            max_chunksize=max_chunksize,
        )
    except ADBCProgrammingError as e:
        raise MatchboxDatabaseWriteError from e


@contextlib.contextmanager
def ingest_to_temporary_table(
    table_name: str,
    schema_name: str,
    data: ArrowTable,
    column_types: dict[str, type[TypeEngine]],
    max_chunksize: int | None = None,
) -> Generator[Table, None, None]:
    """Context manager to ingest Arrow data to a temporary table with explicit types.

    Args:
        table_name: Base name for the temporary table
        schema_name: Schema where the temporary table will be created
        data: PyArrow table containing the data to ingest
        column_types: Map of column names to SQLAlchemy types
        max_chunksize: Optional maximum chunk size for batches

    Returns:
        A SQLAlchemy Table object representing the temporary table
    """
    temp_table_name = f"{table_name}_tmp_{uuid.uuid4().hex}"

    # Validate that all data columns have type mappings
    missing_columns = set(data.column_names) - set(column_types.keys())
    if missing_columns:
        raise ValueError(f"Missing type mappings for columns: {missing_columns}")

    try:
        # Create SQLAlchemy Table from explicit type mapping
        metadata = MetaData(schema=schema_name)
        columns = [
            Column(column_name, column_type())
            for column_name, column_type in column_types.items()
            if column_name in data.column_names
        ]
        temp_table = Table(temp_table_name, metadata, *columns)

        with MBDB.get_session() as session:
            temp_table.create(session.bind)
            session.commit()

        # Ingest data into the temporary table
        with MBDB.get_adbc_connection() as connection:
            _copy_to_table(
                table_name=temp_table_name,
                schema_name=schema_name,
                connection=connection,
                data=data,
                max_chunksize=max_chunksize,
            )
            connection.commit()

        yield temp_table

    finally:
        # Step 3: Clean up
        try:
            with MBDB.get_session() as session:
                temp_table.drop(session.bind, checkfirst=True)
                session.commit()
        except Exception as e:
            logger.warning(f"Failed to drop temp table {temp_table_name}: {e}")
