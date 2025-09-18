"""Definition of model inputs."""

from typing import TYPE_CHECKING, Any

import polars as pl
from polars import DataFrame as PolarsDataFrame

from matchbox.client import _handler
from matchbox.client.helpers import clean
from matchbox.client.sources import Source
from matchbox.common.db import QueryReturnClass, QueryReturnType
from matchbox.common.dtos import (
    QueryCombineType,
    QueryConfig,
)

if TYPE_CHECKING:
    from matchbox.client.models import Model
else:
    Model = Any


class Query:
    """Queriable input to a model."""

    def __init__(
        self,
        *sources: Source,
        model: Model | None = None,
        combine_type: QueryCombineType = QueryCombineType.CONCAT,
        threshold: float | None = None,
        cleaning: dict[str, str] | None = None,
    ):
        """Initialise query.

        Args:
            sources: List of sources to query from
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
