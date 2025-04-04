"""Common database utilities for Matchbox."""

from typing import Any, Iterator, Literal, TypeVar, get_args, overload

import polars as pl
import sqlglot
from adbc_driver_postgresql import dbapi as adbc_dbapi
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.sql.selectable import Select
from sqlglot.dialects import DIALECTS

ReturnTypeStr = Literal["arrow", "pandas", "polars"]
QueryReturnType = ArrowTable | PandasDataFrame | PolarsDataFrame

T = TypeVar("T")


def detect_dialect(connection: Engine | adbc_dbapi.Connection) -> str:
    """Detect the SQLGlot dialect from the connection.

    Args:
        connection (Engine | adbc_dbapi.Connection): The SQLAlchemy Engine or ADBC
            connection object.

    Returns:
        str: The detected SQLGlot dialect.

    Raises:
        ValueError: If the dialect cannot be detected.
    """
    if isinstance(connection, Engine):
        dialect_name: str = connection.dialect.name
    else:
        dialect_name: str = connection.adbc_get_info()["driver_name"]

    detected = [d.lower() for d in DIALECTS if d.lower() in dialect_name.lower()]
    if len(detected) == 1:
        return detected[0]
    else:
        raise ValueError(
            "Could not detect single SQLGlot dialect from driver or dialect name: "
            f"{dialect_name}, {', '.join(detected)}, {DIALECTS}"
        )


def compile_sql(
    stmt: Select,
    connection: Engine | adbc_dbapi.Connection,
) -> str:
    """Compile a SQLAlchemy statement to a SQL string.

    When compiling for ADBC connections, we use PostgreSQL dialect as the default
    because it adheres more closely to SQL standards and provides better compatibility
    when transpiling to other dialects

    Args:
        stmt (Select): A SQLAlchemy Select statement to be compiled.
        connection (Engine | adbc_dbapi.Connection): A SQLAlchemy Engine object or
            ADBC connection.

    Returns:
        str: The compiled SQL string.
    """
    dialect = (
        connection.dialect if isinstance(connection, Engine) else postgresql.dialect()
    )

    compiled_stmt = stmt.compile(
        dialect=dialect,
        compile_kwargs={"literal_binds": True},
    )

    if isinstance(connection, Engine):
        # SQLAlchemy compiles for SQLAlchemy
        return str(compiled_stmt)
    else:
        # SQLGlot transpiles for ADBC
        target_dialect = detect_dialect(connection=connection)
        return sqlglot.transpile(
            sql=str(compiled_stmt),
            read="postgres",
            write=target_dialect,
        )[0]


@overload
def sql_to_df(
    stmt: Select,
    connection: Engine | adbc_dbapi.Connection,
    return_type: Literal["arrow", "pandas", "polars"],
    *,
    return_batches: Literal[False] = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnType: ...


@overload
def sql_to_df(
    stmt: Select,
    connection: Engine | adbc_dbapi.Connection,
    return_type: Literal["arrow", "pandas", "polars"],
    *,
    return_batches: Literal[True],
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[QueryReturnType]: ...


def sql_to_df(
    stmt: Select | str,
    connection: Engine | adbc_dbapi.Connection,
    return_type: ReturnTypeStr = "pandas",
    *,
    return_batches: bool = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnType | Iterator[QueryReturnType]:
    """Executes the given SQLAlchemy statement or SQL string using Polars.

    Args:
        stmt (Select | str): A SQLAlchemy Select statement or SQL string to be executed.
        connection (Engine | adbc_dbapi.Connection): A SQLAlchemy Engine object or
            ADBC connection.
        return_type (str): The type of the return value. One of "arrow", "pandas",
            or "polars".
        return_batches (bool): If True, return an iterator that yields each batch
            separately. If False, return a single DataFrame with all results.
            Default is False.
        batch_size (int | None): Indicate the size of each batch when processing
            data in batches. Default is None.
        schema_overrides (dict[str, Any] | None): A dictionary mapping column names
            to dtypes. Default is None.
        execute_options (dict[str, Any] | None): These options will be passed through
            into the underlying query execution method as kwargs. Default is None.

    Returns:
        If return_batches is False: A dataframe of the query results in the specified
            format.
        If return_batches is True: An iterator of dataframes in the specified format.

    Raises:
        ValueError: If the connection is not properly configured or if an unsupported
            return type is specified.
    """
    if return_type not in get_args(ReturnTypeStr):
        raise ValueError(f"return_type of {return_type} not valid")

    if isinstance(stmt, str):
        sql_query = stmt
    else:
        sql_query = compile_sql(stmt, connection)

    def to_format(results: PolarsDataFrame) -> QueryReturnType:
        if return_type == "polars":
            return results
        elif return_type == "arrow":
            return results.to_arrow()
        elif return_type == "pandas":
            return results.to_pandas()

    res = pl.read_database(
        query=sql_query,
        connection=connection,
        iter_batches=bool(batch_size),
        batch_size=batch_size,
        schema_overrides=schema_overrides,
        execute_options=execute_options,
    )

    if not bool(batch_size):
        return to_format(res)

    if not return_batches:
        return to_format(pl.concat(res))

    return (to_format(batch) for batch in res)


def get_schema_table_names(full_name: str) -> tuple[str, str]:
    """Takes a string table name and returns the unquoted schema and table as a tuple.

    Args:
        full_name: A string indicating a table's full name

    Returns:
        (schema, table): A tuple of schema and table name. If schema
            cannot be inferred, returns None.

    Raises:
        ValueError: When the function can't detect either a
            schema.table or table format in the input
    """
    schema_name_list = full_name.replace('"', "").split(".")

    if len(schema_name_list) == 1:
        schema = None
        table = schema_name_list[0]
    elif len(schema_name_list) == 2:
        schema = schema_name_list[0]
        table = schema_name_list[1]
    else:
        raise ValueError(f"Could not identify schema and table in {full_name}.")

    return schema, table


def fullname_to_prefix(fullname: str) -> str:
    """Converts a full name to a prefix for column names."""
    db_schema, db_table = get_schema_table_names(fullname)
    if db_schema:
        return f"{db_schema}_{db_table}_"
    return f"{db_table}_"
