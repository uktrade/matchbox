"""Common database utilities for Matchbox."""

from typing import Any, Iterator, Literal, TypeVar, overload

import polars as pl
import pyarrow as pa
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.selectable import Select

ReturnTypeStr = Literal["arrow", "pandas", "polars"]

T = TypeVar("T")


def _convert_large_binary_to_binary(table: pa.Table) -> pa.Table:
    """Converts Arrow large_binary fields to binary type."""
    new_fields = []
    for field in table.schema:
        if pa.types.is_large_binary(field.type):
            new_fields.append(field.with_type(pa.binary()))
        else:
            new_fields.append(field)

    new_schema = pa.schema(new_fields)
    return table.cast(new_schema)


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["arrow"],
    *,
    iter_batches: Literal[False] = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> pa.Table: ...


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["arrow"],
    *,
    iter_batches: Literal[True],
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[pa.Table]: ...


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["pandas"],
    *,
    iter_batches: Literal[False] = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> DataFrame: ...


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["pandas"],
    *,
    iter_batches: Literal[True],
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[DataFrame]: ...


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["polars"],
    *,
    iter_batches: Literal[False] = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> PolarsDataFrame: ...


@overload
def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: Literal["polars"],
    *,
    iter_batches: Literal[True],
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[PolarsDataFrame]: ...


def sql_to_df(
    stmt: Select,
    engine: Engine,
    return_type: ReturnTypeStr = "pandas",
    *,
    iter_batches: bool = False,
    batch_size: int | None = None,
    schema_overrides: dict[str, Any] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> (
    pa.Table
    | DataFrame
    | PolarsDataFrame
    | Iterator[pa.Table]
    | Iterator[DataFrame]
    | Iterator[PolarsDataFrame]
):
    """Executes the given SQLAlchemy statement using Polars.

    Args:
        stmt (Select): A SQLAlchemy Select statement to be executed.
        engine (Engine): A SQLAlchemy Engine object for the database connection.
        return_type (str): The type of the return value. One of "arrow", "pandas",
            or "polars".
        iter_batches (bool): If True, return an iterator that yields each batch
            separately. If False, return a single DataFrame with all results.
            Default is False.
        batch_size (int | None): Indicate the size of each batch when processing
            data in batches. Default is None.
        schema_overrides (dict[str, Any] | None): A dictionary mapping column names
            to dtypes. Default is None.
        execute_options (dict[str, Any] | None): These options will be passed through
            into the underlying query execution method as kwargs. Default is None.

    Returns:
        If iter_batches is False: A dataframe of the query results in the specified
            format.
        If iter_batches is True: An iterator of dataframes in the specified format.

    Raises:
        ValueError: If the engine URL is not properly configured or if an unsupported
            return type is specified.
    """
    # Compile the SQLAlchemy statement to SQL string
    compiled_stmt = stmt.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )
    sql_query = str(compiled_stmt)

    # Extract the connection URL from the engine
    url: str | URL = engine.url
    if isinstance(url, URL):
        url = url.render_as_string(hide_password=False)
    if not isinstance(url, str):
        raise ValueError("Unable to obtain a valid connection string from the engine.")

    if iter_batches:
        pl_batches = pl.read_database(
            query=sql_query,
            connection=engine,
            iter_batches=True,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

        if return_type == "polars":
            return pl_batches
        elif return_type == "arrow":
            return (
                _convert_large_binary_to_binary(batch.to_arrow())
                for batch in pl_batches
            )
        elif return_type == "pandas":
            return (batch.to_pandas() for batch in pl_batches)
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")
    else:
        # Use the most efficient method for a single result
        # Fall back if the URI method fails
        try:
            pl_result = pl.read_database_uri(
                query=sql_query,
                uri=url,
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )
        except Exception:
            pl_result = pl.read_database(
                query=sql_query,
                connection=engine,
                batch_size=batch_size,
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )

        if return_type == "polars":
            return pl_result
        elif return_type == "arrow":
            arrow_table = pl_result.to_arrow()
            return _convert_large_binary_to_binary(table=arrow_table)
        elif return_type == "pandas":
            return pl_result.to_pandas()
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")


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
