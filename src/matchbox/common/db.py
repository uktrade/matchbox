"""Common database utilities for Matchbox."""

from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, overload

import connectorx as cx
import pyarrow as pa
from pandas import DataFrame
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.selectable import Select

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any

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
    stmt: Select, engine: Engine, return_type: Literal["arrow"]
) -> pa.Table: ...


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["pandas"]
) -> DataFrame: ...


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["polars"]
) -> PolarsDataFrame: ...


def sql_to_df(
    stmt: Select, engine: Engine, return_type: ReturnTypeStr = "pandas"
) -> pa.Table | DataFrame | PolarsDataFrame:
    """Executes the given SQLAlchemy statement using connectorx.

    Args:
        stmt (Select): A SQLAlchemy Select statement to be executed.
        engine (Engine): A SQLAlchemy Engine object for the database connection.
        return_type (str): The type of the return value. One of "arrow", "pandas",
            or "polars".

    Returns:
        A dataframe of the query results.

    Raises:
        ValueError: If the engine URL is not properly configured.
    """
    compiled_stmt = stmt.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )
    sql_query = str(compiled_stmt)

    url: Union[str, URL] = engine.url

    if isinstance(url, URL):
        url = url.render_as_string(hide_password=False)

    if not isinstance(url, str):
        raise ValueError("Unable to obtain a valid connection string from the engine.")

    result = cx.read_sql(conn=url, query=sql_query, return_type=return_type)

    if return_type == "arrow":
        return _convert_large_binary_to_binary(table=result)

    return result


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

    return (schema, table)


def fullname_to_prefix(fullname: str) -> str:
    """Converts a full name to a prefix for column names."""
    return fullname.replace('"', "").replace(".", "_") + "_"
