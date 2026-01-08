"""Definition of model inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Self

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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
from matchbox.common.logging import profile_time
from matchbox.common.transform import threshold_float_to_int, threshold_int_to_float

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
    ) -> None:
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
            source_resolutions=[source.name for source in self.sources],
            model_resolution=self.model.name if self.model else None,
            combine_type=self.combine_type,
            threshold=threshold_float_to_int(self.threshold)
            if self.threshold
            else None,
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
        sources = [dag.get_source(res) for res in config.source_resolutions]

        # Get model if specified
        model = (
            dag.get_model(config.model_resolution) if config.model_resolution else None
        )

        # Convert threshold back to float
        threshold = (
            threshold_int_to_float(config.threshold) if config.threshold else None
        )

        return cls(
            *sources,
            dag=dag,
            model=model,
            combine_type=config.combine_type,
            threshold=threshold,
            cleaning=config.cleaning,
        )

    def data_raw(
        self,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        return_leaf_id: bool = False,
    ) -> QueryReturnClass:
        """Fetches raw query data by joining source data and matchbox matches.

        Args:
            return_type (optional): Type of dataframe returned, defaults to "polars".
                Other options are "pandas" and "arrow".
            return_leaf_id (optional): Whether matchbox IDs for source clusters should
                be saved as a byproduct in the `leaf_ids` attribute.

        Returns: Data in the requested return type

        Raises:
            MatchboxEmptyServerResponse: If no data was returned by the server.
        """
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mb_ids_path = tmpdir / "mb_ids.parquet"
            # Download data from Matchbox server
            writer = None
            for source in self.sources:
                res = _handler.query(
                    source=source.resolution_path,
                    resolution=self.model.resolution_path if self.model else None,
                    threshold=self.config.threshold,
                    return_leaf_id=return_leaf_id,
                )

                res = res.append_column(
                    "source", pa.array([source.name] * res.num_rows)
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        mb_ids_path,
                        res.schema,
                        compression="snappy",
                        use_dictionary=True,
                    )
                writer.write_table(res)

            writer.close()

            # Download sources from warehouse
            lazy_sources = []
            for source in self.sources:
                source.run()

                lazy_sources.append(
                    pl.scan_parquet(source.cache_path)
                    .select(pl.all().name.prefix(f"{source.name}_"))
                    .with_columns(pl.lit(source.name).alias("source"))
                    .rename({source.qualified_key: "key"})
                )

            mb_ids = pl.scan_parquet(mb_ids_path)

            if return_leaf_id:
                self.leaf_id = mb_ids.select("id", "leaf_id").collect()
                mb_ids = mb_ids.drop("leaf_id")

            raw_data = (
                pl.concat(lazy_sources, how="diagonal")
                .join(mb_ids, how="inner", on=("source", "key"))
                .drop("source", "key")
            )

            if self.config.combine_type == QueryCombineType.SET_AGG:
                raw_data = raw_data.group_by("id").agg(pl.all().exclude("id").unique())
            if self.config.combine_type == QueryCombineType.EXPLODE:
                raw_data = raw_data.group_by("id").agg(pl.all().exclude("id"))
                raw_data = raw_data.explode(pl.all().exclude("id")).unique()

            return _convert_df(raw_data.collect(), return_type=return_type)

    @profile_time()
    def data(
        self,
        raw_data: pl.DataFrame | None = None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        return_leaf_id: bool = False,
    ) -> QueryReturnClass:
        """Returns final data from defined query.

        Args:
            raw_data: If passed, will only apply cleaning instead of fetching raw data.
            return_type (optional): Type of dataframe returned, defaults to "polars".
                Other options are "pandas" and "arrow".
            return_leaf_id (optional): Whether matchbox IDs for source clusters should
                be saved as a byproduct in the `leaf_ids` attribute. If pre-fetched raw
                data is passed, this argument is ignored.

        Returns: Data in the requested return type
        """
        if raw_data is None:
            raw_data = self.data_raw(return_leaf_id=return_leaf_id)

        clean_data = _clean(data=raw_data, cleaning_dict=self.config.cleaning)

        return _convert_df(clean_data, return_type=return_type)

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
) -> pl.DataFrame:
    """Clean data using DuckDB with the provided cleaning SQL.

    * ID is passed through automatically
    * If present, leaf_id is passed through automatically
    * Columns not mentioned in the cleaning_dict are dropped
    * Each key in cleaning_dict is an alias for a SQL expression

    Args:
        data: Raw polars dataframe to clean
        cleaning_dict: A dictionary mapping field aliases to SQL expressions.
            The SQL expressions can reference columns in the data using their names.
            If None, no cleaning is applied and the original data is returned.
            `SourceConfig.f()` can be used to help reference qualified fields.

    Returns:
        Cleaned polars dataframe
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

    # Parse and add each SQL expression from cleaning_dict
    for alias, sql in cleaning_dict.items():
        stmt = parse_one(sql, dialect="duckdb")
        to_select.append(expressions.alias_(stmt, alias))

    query = sqlglot_select(*to_select, dialect="duckdb").from_("data")

    with duckdb.connect(":memory:") as conn:
        conn.register("data", data)
        return conn.execute(query.sql(dialect="duckdb")).pl()


def _convert_df(
    data: PolarsDataFrame, return_type: QueryReturnType
) -> QueryReturnClass:
    match return_type:
        case QueryReturnType.POLARS:
            return data
        case QueryReturnType.PANDAS:
            return data.to_pandas()
        case QueryReturnType.ARROW:
            return data.to_arrow()
        case _:
            raise ValueError(f"Return type {return_type} is invalid")
