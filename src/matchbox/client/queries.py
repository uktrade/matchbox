"""Definition of model inputs."""

import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self

import duckdb
import polars as pl
from polars import DataFrame as PolarsDataFrame
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchbox.client import _handler
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.common.db import QueryReturnClass, QueryReturnType
from matchbox.common.dtos import (
    QueryCombineType,
    QueryConfig,
)
from matchbox.common.transform import truth_float_to_int, truth_int_to_float

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.models import Model
    from matchbox.client.sources import Source
else:
    DAG = Any
    Model = Any
    Source = Any


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
        self.last_run: datetime | None = None
        self.raw_data: PolarsDataFrame | None = None
        self.dag = dag
        self.sources = sources
        self.model = model
        self.combine_type = combine_type
        self.threshold = threshold
        self.cleaning = cleaning

    @property
    def config(self) -> QueryConfig:
        """The query configuration for the current DAG."""
        return QueryConfig(
            source_resolutions=[source.resolution_path for source in self.sources],
            model_resolution=self.model.resolution_path if self.model else None,
            combine_type=self.combine_type,
            threshold=truth_float_to_int(self.threshold) if self.threshold else None,
            cleaning=self.cleaning,
        )

    @classmethod
    def from_config(cls, config: QueryConfig, dag: DAG) -> Self:
        """Create query from config.

        The DAG must have had relevant sources and model added already.

        Args:
            config: The QueryConfig to reconstruct from.
            dag: The DAG containing the sources and model.

        Returns:
            A reconstructed Query instance.
        """
        # Get sources from DAG
        sources = [dag.get_source(res.name) for res in config.source_resolutions]

        # Get model if specified
        model = (
            dag.get_model(config.model_resolution.name)
            if config.model_resolution
            else None
        )

        # Convert threshold back to float
        threshold = truth_int_to_float(config.threshold) if config.threshold else None

        return cls(
            *sources,
            dag=dag,
            model=model,
            combine_type=config.combine_type,
            threshold=threshold,
            cleaning=config.cleaning,
        )

    def run(
        self,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        return_leaf_id: bool = False,
        batch_size: int | None = None,
        full_rerun: bool = False,
        cache_raw: bool = False,
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
            full_rerun: Whether to force a re-run of the query
            cache_raw: Whether to store the pre-cleaned data to iterate on cleaning.

        Returns: Data in the requested return type

        Raises:
            MatchboxEmptyServerResponse: If no data was returned by the server.
        """
        if self.last_run and not full_rerun:
            warnings.warn("Query already run, skipping.", UserWarning, stacklevel=2)
            return self.data

        source_results: list[PolarsDataFrame] = []
        for source in self.sources:
            mb_ids = pl.from_arrow(
                _handler.query(
                    source=source.resolution_path,
                    resolution=self.model.resolution_path if self.model else None,
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
                raw_data = concatenated.drop(["leaf_id"])
            else:
                raw_data = pl.concat(tables, how="diagonal")
        else:
            raw_data = tables[0].drop("leaf_id", strict=False)
            for table in tables[1:]:
                raw_data = raw_data.join(
                    table.drop("leaf_id", strict=False),
                    on="id",
                    how="full",
                    coalesce=True,
                )

            raw_data = raw_data.select(["id", pl.all().exclude("id")])

            if self.config.combine_type == QueryCombineType.SET_AGG:
                # Aggregate into lists
                agg_expressions = [
                    pl.col(col).unique() for col in raw_data.columns if col != "id"
                ]
                raw_data = raw_data.group_by("id").agg(agg_expressions)

        if cache_raw:
            self.raw_data = raw_data

        self.data = _convert_df(
            _clean(
                data=raw_data,
                cleaning_dict=self.config.cleaning,
                key_columns=[source.qualified_key for source in self.sources],
            ),
            return_type=return_type,
        )
        self.last_run = datetime.now()

        return self.data

    def clean(
        self,
        cleaning: dict[str, str] | None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
    ) -> QueryReturnClass:
        """Change cleaning dictionary and re-apply cleaning, if raw data was cached.

        Args:
            cleaning: A dictionary mapping field aliases to SQL expressions.
                The SQL expressions can reference columns in the data using their names.
                If None, no cleaning is applied and the original data is returned.
                `SourceConfig.f()` can be used to help reference qualified fields.
            return_type (optional): Type of dataframe returned, defaults to "polars".
                    Other options are "pandas" and "arrow".
        """
        if self.raw_data is None:
            raise RuntimeError("No raw data is stored in this query.")

        self.data = _convert_df(
            data=_clean(
                data=self.raw_data,
                cleaning_dict=cleaning,
                key_columns=[source.qualified_key for source in self.sources],
            ),
            return_type=return_type,
        )

        self.cleaning = cleaning

        return self.data

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


def _clean(
    data: pl.DataFrame,
    cleaning_dict: dict[str, str] | None,
    key_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Clean data using DuckDB with the provided cleaning SQL.

    * ID is passed through automatically
    * If present, leaf_id is passed through automatically
    * Key columns (specified in key_columns) are passed through automatically
    * Columns not mentioned in the cleaning_dict are dropped
    * Each key in cleaning_dict is an alias for a SQL expression

    Args:
        data: Raw polars dataframe to clean
        cleaning_dict: A dictionary mapping field aliases to SQL expressions.
            The SQL expressions can reference columns in the data using their names.
            If None, no cleaning is applied and the original data is returned.
            `SourceConfig.f()` can be used to help reference qualified fields.
        key_columns: List of key column names to pass through automatically.
            These are typically qualified key columns like "foo_key", "bar_key".

    Returns:
        Cleaned polars dataframe

    Examples:
        Column passthrough behaviour:

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
        # Result columns: id, full_name
        ```

        Special columns (leaf_id, key) handling:

        ```python
        data = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "leaf_id": ["a", "b", "c"],
                "foo_key": ["x", "y", "z"],
                "bar_key": ["p", "q", "r"],
                "value": [10, 20, 30],
                "status": ["active", "inactive", "pending"],
            }
        )
        cleaning_dict = {"processed_value": "value * 2"}
        key_columns = ["foo_key", "bar_key"]
        result = clean(data, cleaning_dict, key_columns)
        # Result columns: id, leaf_id, foo_key, bar_key, processed_value
        # 'id', 'leaf_id', and key columns always included automatically
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

    # Add leaf_id if it exists
    if "leaf_id" in data.columns:
        to_select.append(_add_column("leaf_id"))

    # Add key columns if specified
    if key_columns:
        for key_col in key_columns:
            if key_col in data.columns:
                to_select.append(_add_column(key_col))

    # Parse and add each SQL expression from cleaning_dict
    for alias, sql in cleaning_dict.items():
        stmt = parse_one(sql, dialect="duckdb")
        to_select.append(expressions.alias_(stmt, alias))

    query = sqlglot_select(*to_select, dialect="duckdb").from_("data")

    with duckdb.connect(":memory:") as conn:
        conn.register("data", data)
        return conn.execute(query.sql(dialect="duckdb")).pl()


def _convert_df(data: PolarsDataFrame, return_type: QueryReturnType):
    match return_type:
        case QueryReturnType.POLARS:
            return data
        case QueryReturnType.PANDAS:
            return data.to_pandas()
        case QueryReturnType.ARROW:
            return data.to_arrow()
        case _:
            raise ValueError(f"Return type {return_type} is invalid")
