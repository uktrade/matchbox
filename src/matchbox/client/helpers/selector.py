"""Functions to select and retrieve data from the Matchbox server."""

import duckdb
import polars as pl
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchbox.client import _handler
from matchbox.common.dtos import Match
from matchbox.common.graph import (
    DEFAULT_RESOLUTION,
    ResolutionName,
    SourceResolutionName,
)


def match(
    *targets: list[SourceResolutionName],
    source: SourceResolutionName,
    key: str,
    resolution: ResolutionName = DEFAULT_RESOLUTION,
    threshold: int | None = None,
) -> list[Match]:
    """Matches IDs against the selected backend.

    Args:
        targets: Source resolutions to find keys in
        source: The source resolution the provided key belongs to
        key: The value to match from the source. Usually a primary key
        resolution (optional): The resolution to use to resolve matches against
            If not set, it will look for a default resolution.
        threshold (optional): The threshold to use for creating clusters.
            If None, uses the resolutions' default threshold
            If an integer, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors

    Examples:
        ```python
        mb.match(
            "datahub_companies",
            "hmrc_exporters",
            source="companies_house",
            key="8534735",
            resolution="last_linker",
        )
        ```
    """
    # Validate arguments
    for name in targets + (source,):
        _ = _handler.get_source_resolution(name=name)

    return _handler.match(
        targets=targets,
        source=source,
        key=key,
        resolution=resolution,
        threshold=threshold,
    )


def clean(data: pl.DataFrame, cleaning_dict: dict[str, str] | None) -> pl.DataFrame:
    """Clean data using DuckDB with the provided cleaning SQL.

    * ID is passed through automatically
    * If present, leaf_id and key are passed through automatically
    * Columns not mentioned in the cleaning_dict are passed through unchanged
    * Each key in cleaning_dict is an alias for a SQL expression

    Args:
        data: Raw polars dataframe to clean
        cleaning_dict: A dictionary mapping field aliases to SQL expressions.
            The SQL expressions can reference columns in the data using their names.
            If None, no cleaning is applied and the original data is returned.
            `SourceConfig.f()` can be used to help reference qualified fields.

    Returns:
        Cleaned polars dataframe

    Examples:
        Column passthrough behavior:

        ```python
        data = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["John", "Jane", "Bob"],
                "age": [25, 30, 35],
                "city": ["London", "Hull", "Stratford-upon-Avon"],
            }
        )
        cleaning_dict = {
            "full_name": "name"  # Only references 'name' column
        }
        result = clean(data, cleaning_dict)
        # Result columns: id, full_name, age, city
        # 'name' is dropped because it was used in cleaning_dict
        # 'age' and 'city' are passed through unchanged
        ```

        Multiple column references:

        ```python
        data = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "first": ["John", "Jane", "Bob"],
                "last": ["Doe", "Smith", "Johnson"],
                "salary": [50000, 60000, 55000],
            }
        )
        cleaning_dict = {
            "name": "first || ' ' || last",  # References both 'first' and 'last'
            "high_earner": "salary > 55000",
        }
        result = clean(data, cleaning_dict)
        # Result columns: id, name, high_earner
        # 'first', 'last', and 'salary' are dropped (used in expressions)
        # No other columns to pass through
        ```

        Special columns (leaf_id, key) handling:

        ```python
        data = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "leaf_id": ["a", "b", "c"],
                "key": ["x", "y", "z"],
                "value": [10, 20, 30],
                "status": ["active", "inactive", "pending"],
            }
        )
        cleaning_dict = {"processed_value": "value * 2"}
        result = clean(data, cleaning_dict)
        # Result columns: id, leaf_id, key, processed_value, status
        # 'id', 'leaf_id', 'key' always included automatically
        # 'value' dropped (used in expression), 'status' passed through
        ```

        No cleaning (returns original data):

        ```python
        data = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "score": [95, 87]})
        result = clean(data, None)
        # Returns exact same dataframe with all original columns
        ```
    """
    if cleaning_dict is None:
        return data

    def _add_column(name: str) -> expressions.Alias:
        """Select and alias a column with the given name."""
        return expressions.Alias(
            this=expressions.Column(
                this=expressions.Identifier(this=name, quoted=False)
            ),
            alias=expressions.Identifier(this=name, quoted=False),
        )

    # Always select 'id'
    to_select: list[expressions.Expression] = [_add_column("id")]

    # Add optional columns if they exist
    for col in ["leaf_id", "key"]:
        if col in data.columns:
            to_select.append(_add_column(col))

    # Parse and add each SQL expression from cleaning_dict
    query_column_names: set[str] = set()
    for alias, sql in cleaning_dict.items():
        stmt = parse_one(sql, dialect="duckdb")

        # Get column name used in the expression
        for node in stmt.walk():
            if isinstance(node, expressions.Column):
                query_column_names.add(node.name)

        # Add to the list of expressions to select
        to_select.append(expressions.alias_(stmt, alias))

    # Add all column names not used in the query
    for column in set(data.columns) - query_column_names - {"id", "leaf_id", "key"}:
        to_select.append(
            expressions.Alias(
                this=expressions.Column(
                    this=expressions.Identifier(this=column, quoted=False)
                ),
                alias=expressions.Identifier(this=column, quoted=False),
            )
        )

    query = sqlglot_select(*to_select, dialect="duckdb").from_("data")

    with duckdb.connect(":memory:") as conn:
        conn.register("data", data)
        return conn.execute(query.sql(dialect="duckdb")).pl()
