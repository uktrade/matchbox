"""Definition of model inputs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchbox.client import _handler
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.common.db import QueryReturnClass, QueryReturnType
from matchbox.common.dtos import QueryCombineType, QueryConfig, stable_hash_dict
from matchbox.common.logging import profile_time
from matchbox.common.resolvers import (
    ResolverSettings,
)

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.models.models import Model
    from matchbox.client.resolvers import Resolver
    from matchbox.client.sources import Source
else:
    DAG = Any
    Model = Any
    Resolver = Any
    Source = Any


class Query:
    """Queriable input to a model."""

    def __init__(
        self,
        *sources: Source,
        dag: DAG,
        resolver: Resolver | None = None,
        resolver_overrides: ResolverSettings | dict[str, Any] | None = None,
        combine_type: QueryCombineType = QueryCombineType.CONCAT,
        cleaning: dict[str, str] | None = None,
    ) -> None:
        """Initialise query.

        Args:
            sources: List of sources to query from
            dag: DAG containing sources and models.
            resolver (optional): Resolver to use to resolve sources. It can be missing
                if querying from a single source.
            resolver_overrides (optional): Full replacement settings object to apply
                to this resolver for query-time analysis.
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

            cleaning (optional): A dictionary mapping an output column name to a SQL
                expression that will populate a new column.
        """
        self.raw_data: PolarsDataFrame | None = None
        self.dag = dag
        self.sources = sources
        self.resolver = resolver
        self.resolver_overrides = self._normalise_resolver_overrides(resolver_overrides)
        self.combine_type = combine_type
        self.cleaning = cleaning

    def _normalise_resolver_overrides(
        self,
        resolver_overrides: ResolverSettings | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Validate and normalise resolver overrides as JSON payloads."""
        if resolver_overrides is None:
            return None
        if self.resolver is None:
            raise ValueError("resolver_overrides require a resolver-backed query.")
        method = self.resolver.resolver_class(settings=resolver_overrides)
        return method.settings.model_dump(mode="json")

    def _effective_resolver_settings(self) -> dict[str, Any] | None:
        """Return resolver settings applied to this query."""
        if self.resolver is None:
            return None
        if self.resolver_overrides is not None:
            return self.resolver_overrides
        return self.resolver.resolver_settings.model_dump(mode="json")

    def _resolver_settings_hash(self) -> str | None:
        """Return hash for resolver config + effective query-time settings."""
        if self.resolver is None:
            return None
        return stable_hash_dict(
            {
                "resolver_config": self.resolver.config.model_dump(mode="json"),
                "effective_resolver_settings": self._effective_resolver_settings(),
                "resolver_overrides": self.resolver_overrides,
            }
        )

    def _assert_pipeline_safe(self) -> None:
        """Ensure analysis-only override queries are not used in pipeline builds."""
        if self.resolver_overrides is not None:
            raise ValueError(
                "Queries with resolver_overrides are analysis-only and cannot be "
                "used as model inputs. Create another resolver for inference-speed "
                "settings changes."
            )

    @property
    def config(self) -> QueryConfig:
        """The query configuration for the current DAG."""
        return QueryConfig(
            source_resolutions=tuple(source.name for source in self.sources),
            resolver_resolution=self.resolver.name if self.resolver else None,
            combine_type=self.combine_type,
            resolver_settings_hash=self._resolver_settings_hash(),
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

        # Get resolver if specified
        resolver = (
            dag.get_resolver(config.resolver_resolution)
            if config.resolver_resolution
            else None
        )

        return cls(
            *sources,
            dag=dag,
            resolver=resolver,
            combine_type=config.combine_type,
            cleaning=config.cleaning,
        )

    @overload
    def data_raw(
        self,
        return_type: Literal[QueryReturnType.POLARS] = ...,
        return_leaf_id: bool = False,
    ) -> PolarsDataFrame: ...

    @overload
    def data_raw(
        self,
        return_type: Literal[QueryReturnType.PANDAS] = ...,
        return_leaf_id: bool = False,
    ) -> PandasDataFrame: ...

    @overload
    def data_raw(
        self,
        return_type: Literal[QueryReturnType.ARROW] = ...,
        return_leaf_id: bool = False,
    ) -> ArrowTable: ...

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
                    resolution=self.resolver.resolution_path if self.resolver else None,
                    resolver_overrides=self.resolver_overrides,
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
        self._assert_pipeline_safe()
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
        self._assert_pipeline_safe()
        other_query._assert_pipeline_safe()
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
