"""Common database utilities for Matchbox."""

from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
)

import polars as pl
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    from adbc_driver_postgresql.dbapi import Connection as AdbcConnection
else:
    AdbcConnection = Any


class QueryReturnType(StrEnum):
    """Enumeration of dataframe types to return from query."""

    PANDAS = "pandas"
    POLARS = "polars"
    ARROW = "arrow"


QueryReturnClass: TypeAlias = ArrowTable | PandasDataFrame | PolarsDataFrame

T = TypeVar("T")


@overload
def sql_to_df(
    stmt: str,
    connection: Engine | AdbcConnection,
    return_type: QueryReturnType,
    *,
    return_batches: Literal[False] = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnClass: ...


@overload
def sql_to_df(
    stmt: str,
    connection: Engine | AdbcConnection,
    return_type: QueryReturnType,
    *,
    return_batches: Literal[True],
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[QueryReturnClass]: ...


def sql_to_df(
    stmt: str,
    connection: Engine | AdbcConnection,
    return_type: QueryReturnType = QueryReturnType.PANDAS,
    *,
    return_batches: bool = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> QueryReturnClass | Iterator[QueryReturnClass]:
    """Executes the given SQLAlchemy statement or SQL string using Polars.

    Args:
        stmt: A SQL string to be executed.
        connection: A SQLAlchemy Engine object or
            ADBC connection.
        return_type: The type of the return value. One of "arrow", "pandas",
            or "polars".
        return_batches: If True, return an iterator that yields each batch
            separately. If False, return a single DataFrame with all results.
            Default is False.
        batch_size: Indicate the size of each batch when processing
            data in batches. Default is None.
        rename: A dictionary mapping old column
            names to new column names, or a callable that takes a DataFrame and
            returns a DataFrame with renamed columns. Default is None.
        schema_overrides: A dictionary mapping column names
            to dtypes. Default is None.
        execute_options: These options will be passed through
            into the underlying query execution method as kwargs. Default is None.

    Returns:
        If return_batches is False: A dataframe of the query results in the specified
            format.
        If return_batches is True: An iterator of dataframes in the specified format.

    Raises:
        ValueError:

            * If the connection is not properly configured or if an unsupported
                return type is specified.
            * If batch_size and return_batches are either both set or both unset.

    """
    if not batch_size and return_batches:
        raise ValueError("A batch size must be specified if return_batches is True")

    if batch_size and not return_batches:
        raise ValueError("Cannot set a batch size if return_batches if False")

    def _to_format(results: PolarsDataFrame) -> QueryReturnClass:
        """Convert the results to the specified format."""
        if rename:
            results = results.rename(rename)

        match return_type:
            case QueryReturnType.POLARS:
                return results
            case QueryReturnType.PANDAS:
                return results.to_pandas()
            case QueryReturnType.ARROW:
                return results.to_arrow()
            case _:
                raise ValueError("Unknown return type specified.")

    res = pl.read_database(
        query=stmt,
        connection=connection,
        iter_batches=bool(batch_size),
        batch_size=batch_size,
        schema_overrides=schema_overrides,
        execute_options=execute_options,
    )

    if return_batches:
        return (_to_format(batch) for batch in res)

    return _to_format(res)
