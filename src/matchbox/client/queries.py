"""Definition of model inputs."""

from typing import TYPE_CHECKING, Any, Self

import duckdb
import polars as pl
from polars import DataFrame as PolarsDataFrame
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchbox.client import _handler
from matchbox.client.models import Model
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.sources import Source
from matchbox.common.db import QueryReturnClass, QueryReturnType
from matchbox.common.dtos import (
    QueryCombineType,
    QueryConfig,
)

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
else:
    DAG = Any


class Query:
    """Queriable input to a model."""

    def __init__(
        self,
        *sources: Source,
        dag: DAG,
        model: Model | None = None,
        combine_type: QueryCombineType = QueryCombineType.CONCAT,
        threshold: float | None = None,
        cleaning: dict[str, str] | None = None,
    ):
        """Initialise query.

        Args:
            sources: List of sources to query from
            dag: DAG containing sources and models.
            model (optional): Model to use to resolve sources. It can only be missing
                if querying from a single source.
            combine_type (optional): How to combine the data from different sources.
                Default is `concat`.

                * If `concat`, concatenate all sources queried without any merging.
                    Multiple rows per ID, with null values where data isn't available
                * If `explode`, outer join on Matchbox ID. Multiple rows per ID,
                    with one for every unique combination of data requested
                    across all sources
                * If `set_agg`, join on Matchbox ID, group on Matchbox ID, then
                    aggregate to nested lists of unique values. One row per ID,
                    but all requested data is in nested arrays

            threshold (optional): The threshold to use for creating clusters
                If None, uses the resolutions' default threshold
                If an integer, uses that threshold for the specified resolution, and the
                resolution's cached thresholds for its ancestors

            cleaning (optional): A dictionary mapping an output column name to a SQL
                expression that will populate a new column.
        """
        self.dag = dag
        self.sources = sources
        self.model = model
        self.config = QueryConfig(
            source_resolutions=[source.name for source in sources],
            model_resolution=model.name if model else None,
            combine_type=combine_type,
            threshold=int(threshold * 100) if threshold else None,
            cleaning=cleaning,
        )

    def run(
        self,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        return_leaf_id: bool = False,
        batch_size: int | None = None,
    ) -> QueryReturnClass:
        """Runs queries against the selected backend.

        Args:
            return_type (optional): Type of dataframe returned, defaults to "polars".
                Other options are "pandas" and "arrow".
            return_leaf_id (optional): Whether matchbox IDs for source clusters should
                be saved as a byproduct in the `leaf_ids` attribute.
            batch_size (optional): The size of each batch when fetching data from the
                warehouse, which helps reduce memory usage and load on the database.
                Default is None.

        Returns: Data in the requested return type

        Raises:
            MatchboxEmptyServerResponse: If no data was returned by the server.
        """
        source_results: list[PolarsDataFrame] = []
        for source in self.sources:
            mb_ids = pl.from_arrow(
                _handler.query(
                    source=source.name,
                    resolution=self.model.name if self.model else None,
                    threshold=self.config.threshold,
                    return_leaf_id=return_leaf_id,
                )
            )

            raw_batches = source.fetch(
                qualify_names=True,
                batch_size=batch_size,
                return_type=QueryReturnType.POLARS,
            )

            processed_batches = [
                b.join(
                    other=mb_ids,
                    left_on=source.qualified_key,
                    right_on="key",
                    how="inner",
                )
                for b in raw_batches
            ]
            source_results.append(pl.concat(processed_batches, how="vertical"))

        # Process all data and return a single result
        tables: list[PolarsDataFrame] = list(source_results)

        # Combine results based on combine_type
        if return_leaf_id:
            concatenated = pl.concat(tables, how="diagonal")
            self.leaf_id = concatenated.select(["id", "leaf_id"])

        if self.config.combine_type == QueryCombineType.CONCAT:
            if return_leaf_id:  # can reuse the concatenated dataframe
                data = concatenated.drop(["leaf_id"])
            else:
                data = pl.concat(tables, how="diagonal")
        else:
            data = tables[0].drop("leaf_id", strict=False)
            for table in tables[1:]:
                data = data.join(
                    table.drop("leaf_id", strict=False),
                    on="id",
                    how="full",
                    coalesce=True,
                )

            data = data.select(["id", pl.all().exclude("id")])

            if self.config.combine_type == QueryCombineType.SET_AGG:
                # Aggregate into lists
                agg_expressions = [
                    pl.col(col).unique() for col in data.columns if col != "id"
                ]
                data = data.group_by("id").agg(agg_expressions)

        data = clean(data, self.config.cleaning)

        match return_type:
            case QueryReturnType.POLARS:
                return data
            case QueryReturnType.PANDAS:
                return data.to_pandas()
            case QueryReturnType.ARROW:
                return data.to_arrow()
            case _:
                raise ValueError(f"Return type {return_type} is invalid")

    def deduper(
        self,
        name: str,
        model_class: Deduper,
        model_settings: DeduperSettings,
        description: str | None = None,
    ) -> Model:
        """Create deduper for data in this query."""
        return self.dag.model(
            name=name,
            description=description,
            model_class=model_class,
            model_settings=model_settings,
            left_query=self,
        )

    def linker(
        self,
        other_query: Self,
        name: str,
        model_class: Linker,
        model_settings: LinkerSettings,
        description: str | None = None,
    ) -> Model:
        """Create linker for data in this query and another query."""
        return self.dag.model(
            name=name,
            description=description,
            model_class=model_class,
            model_settings=model_settings,
            left_query=self,
            right_query=other_query,
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
