"""Functions to select and retrieve data from the Matchbox server."""

import itertools
from typing import Generator, Iterator, Literal, get_args

import polars as pl
from polars import DataFrame as PolarsDataFrame
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, create_engine

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.common.db import QueryReturnType, ReturnTypeStr
from matchbox.common.graph import DEFAULT_RESOLUTION
from matchbox.common.logging import logger
from matchbox.common.sources import Match, Source, SourceAddress


class Selector(BaseModel):
    """A selector to choose a source and optionally a subset of columns to select."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    address: SourceAddress
    fields: list[str] | None = None
    engine: Engine


def select(
    *selection: str | dict[str, str],
    engine: Engine | None = None,
) -> list[Selector]:
    """From one engine, builds and verifies a list of selectors.

    Args:
        selection: Full source names and optionally a subset of columns to select
        engine: The engine to connect to the data warehouse hosting the source.
            If not provided, will use a connection string from the
            `MB__CLIENT__DEFAULT_WAREHOUSE` environment variable.

    Returns:
        A list of Selector objects

    Examples:
        ```python
        select("companies_house", "hmrc_exporters", engine=engine)
        ```

        ```python
        select({"companies_house": ["crn"], "hmrc_exporters": ["name"]}, engine=engine)
        ```
    """
    if not engine:
        if default_engine := settings.default_warehouse:
            engine = create_engine(default_engine)
            logger.warning("Using default engine")
        else:
            raise ValueError(
                "An engine needs to be provided if "
                "`MB__CLIENT__DEFAULT_WAREHOUSE` is unset"
            )

    selectors = []
    for s in selection:
        if isinstance(s, str):
            source_address = SourceAddress.compose(engine, s)
            selectors.append(Selector(engine=engine, address=source_address))
        elif isinstance(s, dict):
            for full_name, fields in s.items():
                source_address = SourceAddress.compose(engine, full_name)

                selectors.append(
                    Selector(engine=engine, address=source_address, fields=fields)
                )
        else:
            raise ValueError("Selection specified in incorrect format")

    return selectors


def _process_query_result(
    data: PolarsDataFrame, selector: Selector, mb_ids: PolarsDataFrame, db_pk: str
) -> PolarsDataFrame:
    """Process query results by joining with matchbox IDs and filtering fields.

    Args:
        data: The raw data from the source
        selector: The selector with source and fields information
        mb_ids: The matchbox IDs
        db_pk: The primary key of the source

    Returns:
        The processed table with joined matchbox IDs and filtered fields
    """
    # Join data with matchbox IDs
    joined_table = data.join(
        other=mb_ids,
        left_on=selector.address.format_column(db_pk),
        right_on="source_pk",
        how="inner",
    )

    # Apply field filtering if needed
    if selector.fields:
        keep_cols = ["id"] + [
            selector.address.format_column(f) for f in selector.fields
        ]
        match_cols = [col for col in joined_table.columns if col in keep_cols]
        return joined_table.select(match_cols)
    else:
        return joined_table


def _source_query(
    selector: Selector,
    mb_ids: PolarsDataFrame | None = None,
    return_batches: bool = False,
    batch_size: int | None = None,
    only_indexed: bool = False,
    limit: int | None = None,
) -> tuple[Source, Iterator[PolarsDataFrame]]:
    """From a Selector, query a source and join to matchbox IDs."""
    source = _handler.get_source(selector.address).set_engine(selector.engine)

    indexed_columns = set()
    if source.columns:
        indexed_columns = set([col.name for col in source.columns])

    # If only_indexed is True and source.columns is unset, we will raise
    if only_indexed and selector.fields and not set(selector.fields) <= indexed_columns:
        raise ValueError("Attempting to query unindexed columns.")

    selected_fields = None
    if selector.fields:
        selected_fields = list(set(selector.fields))

    raw_results = source.to_polars(
        fields=selected_fields,
        pks=mb_ids["source_pk"].to_list() if mb_ids is not None else None,
        return_batches=return_batches,
        batch_size=batch_size,
        limit=limit,
    )

    if isinstance(raw_results, PolarsDataFrame):
        raw_results = [raw_results]

    return source, raw_results


def _process_selectors(
    selectors: list[Selector],
    sub_limits: list[int | None],
    resolution_name: str | None,
    threshold: int | None,
    batch_size: int | None,
    only_indexed: bool,
) -> Iterator[PolarsDataFrame]:
    """Helper function to process selectors and return an iterator of results.

    For non-batched queries, turn this into a list.

    For batched queries, yield from it.
    """
    limit_threshold: int = 1_000
    client_side_filtering: bool = False

    if any(
        (sub_limit is not None and sub_limit > limit_threshold)
        for sub_limit in sub_limits
    ):
        logger.warning(
            f"Limit is above {limit_threshold:,} threshold for "
            "filtering warehouse-side. Filtering client-side."
        )
        client_side_filtering = True

    # Create iterators for each selector
    selector_iters: list[Generator[PolarsDataFrame, None, None]] = []

    def _process_batches(
        batches: Iterator[PolarsDataFrame],
        selector: Selector,
        mb_ids: PolarsDataFrame,
        db_pk: str,
    ) -> Generator[PolarsDataFrame, None, None]:
        """Process and transform each batch of results."""
        for batch in batches:
            yield _process_query_result(batch, selector, mb_ids, db_pk=db_pk)

    for selector, sub_limit in zip(selectors, sub_limits, strict=True):
        mb_ids = pl.from_arrow(
            _handler.query(
                source_address=selector.address,
                resolution_name=resolution_name,
                threshold=threshold,
                limit=None if client_side_filtering else sub_limit,
            )
        )
        source, raw_batches = _source_query(
            selector=selector,
            mb_ids=None if client_side_filtering else mb_ids,
            return_batches=True,
            batch_size=batch_size,
            only_indexed=only_indexed,
            limit=sub_limit if client_side_filtering else None,
        )

        # Process and transform each batch
        selector_iters.append(
            _process_batches(
                batches=raw_batches,
                selector=selector,
                mb_ids=mb_ids,
                db_pk=source.db_pk,
            )
        )

    # Chain iterators if multiple selectors
    if len(selector_iters) == 1:
        batches_iter = selector_iters[0]
    else:
        # Interleave batches from different selectors
        batches_iter = itertools.chain.from_iterable(selector_iters)

    return batches_iter


def _query_batched(
    selectors: list[Selector],
    sub_limits: list[int | None],
    resolution_name: str | None,
    threshold: int | None,
    return_type: ReturnTypeStr,
    batch_size: int | None,
    only_indexed: bool,
) -> Iterator[Generator[PolarsDataFrame, None, None]]:
    """Helper function that implements batched query processing.

    Returns an iterator yielding batches of results.
    """
    batches_iter: Iterator[PolarsDataFrame] = _process_selectors(
        selectors=selectors,
        sub_limits=sub_limits,
        resolution_name=resolution_name,
        threshold=threshold,
        batch_size=batch_size,
        only_indexed=only_indexed,
    )

    for batch in batches_iter:
        match return_type:
            case "pandas":
                yield batch.to_pandas()
            case "polars":
                yield batch
            case "arrow":
                yield batch.to_arrow()


def query(
    *selectors: list[Selector],
    resolution_name: str | None = None,
    combine_type: Literal["concat", "explode", "set_agg"] = "concat",
    return_type: ReturnTypeStr = "pandas",
    threshold: int | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    return_batches: bool = False,
    only_indexed: bool = False,
) -> QueryReturnType | Iterator[QueryReturnType]:
    """Runs queries against the selected backend.

    Args:
        selectors: Each selector is the output of `select()`.
            This allows querying sources coming from different engines
        resolution_name (optional): The name of the resolution point to query
            If not set:

            * If querying a single source, it will use the source resolution
            * If querying 2 or more sources, it will look for a default resolution
        combine_type: How to combine the data from different sources.

            * If `concat`, concatenate all sources queried without any merging.
                Multiple rows per ID, with null values where data isn't available
            * If `explode`, outer join on Matchbox ID. Multiple rows per ID,
                with one for every unique combination of data requested
                across all sources
            * If `set_agg`, join on Matchbox ID, group on Matchbox ID, then
                aggregate to nested lists of unique values. One row per ID,
                but all requested data is in nested arrays
        return_type: The form to return data in, one of "pandas" or "arrow"
            Defaults to pandas for ease of use
        threshold (optional): The threshold to use for creating clusters
            If None, uses the resolutions' default threshold
            If an integer, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors
        limit (optional): The number to use in a limit clause. Useful for testing
        batch_size (optional): The size of each batch when fetching data from the
            warehouse, which helps reduce memory usage and load on the database.
            Default is None.
        return_batches (optional): If True, returns an iterator of batches instead of a
            single combined result, which is useful for processing large datasets with
            limited memory. Default is False.
        only_indexed (optional): If True, it will raise an exception when attempting to
            query un-indexed columns, which should never be done if querying for
            subsequent matching. Default is False.

    Returns:
        If return_batches is False:
            Data in the requested return type (DataFrame or ArrowTable)
        If return_batches is True:
            An iterator yielding batches in the requested return type

    Examples:
        ```python
        query(
            select({"companies_house": ["crn", "name"]}, engine=engine),
        )
        ```

        ```python
        query(
            select("companies_house", engine=engine1),
            select("datahub_companies", engine=engine2),
            resolution_name="last_linker",
        )
        ```

        ```python
        # Process large results in batches of 5000 rows
        for batch in query(
            select("companies_house", engine=engine),
            batch_size=5000,
            return_batches=True,
        ):
            batch.head()
        ```
    """
    # Validate arguments
    if combine_type not in ("concat", "explode", "set_agg"):
        raise ValueError(f"combine_type of {combine_type} not valid")

    if return_type not in get_args(ReturnTypeStr):
        raise ValueError(f"return_type of {return_type} not valid")

    if not selectors:
        raise ValueError("At least one selector must be specified")

    selectors: list[Selector] = list(itertools.chain(*selectors))

    if not resolution_name and len(selectors) > 1:
        resolution_name = DEFAULT_RESOLUTION

    # Divide the limit among selectors
    if limit:
        n_selectors = len(selectors)
        sub_limit_base = limit // n_selectors
        sub_limit_remainder = limit % n_selectors
        sub_limits = [sub_limit_base + 1] * sub_limit_remainder + [sub_limit_base] * (
            n_selectors - sub_limit_remainder
        )
    else:
        sub_limits = [None] * len(selectors)

    if return_batches:
        if combine_type != "concat":
            raise logger.warning(
                "Batching is only supported for `combine_type='concat'`"
            )

        # Return an iterator of batches
        return _query_batched(
            selectors=selectors,
            sub_limits=sub_limits,
            resolution_name=resolution_name,
            threshold=threshold,
            return_type=return_type,
            batch_size=batch_size,
            only_indexed=only_indexed,
        )
    else:
        # Process all data and return a single result
        tables: list[PolarsDataFrame] = list(
            _process_selectors(
                selectors=selectors,
                sub_limits=sub_limits,
                resolution_name=resolution_name,
                threshold=threshold,
                batch_size=batch_size,
                only_indexed=only_indexed,
            )
        )

        # Make sure we have some results
        if not tables:
            empty_result = pl.DataFrame()
            match return_type:
                case "pandas":
                    return empty_result.to_pandas()
                case "polars":
                    return empty_result
                case "arrow":
                    return empty_result.to_arrow()

        # Combine results based on combine_type
        if combine_type == "concat":
            result = pl.concat(tables, how="diagonal")
        else:
            result = tables[0]
            for table in tables[1:]:
                result = result.join(table, on="id", how="full", coalesce=True)

            result = result.select(["id", pl.all().exclude("id")])

            if combine_type == "set_agg":
                # Aggregate into lists
                agg_expressions = [
                    pl.col(col).unique() for col in result.columns if col != "id"
                ]
                result = result.group_by("id").agg(agg_expressions)

        # Return in requested format
        match return_type:
            case "pandas":
                return result.to_pandas()
            case "polars":
                return result
            case "arrow":
                return result.to_arrow()


def match(
    *targets: list[Selector],
    source: list[Selector],
    source_pk: str,
    resolution_name: str = DEFAULT_RESOLUTION,
    threshold: int | None = None,
) -> list[Match]:
    """Matches IDs against the selected backend.

    Args:
        targets: Each target is the output of `select()`.
            This allows matching against sources coming from different engines
        source: The output of using `select()` on a single source.
        source_pk: The primary key value to match from the source.
        resolution_name (optional): The resolution name to use for filtering results.
            If not set, it will look for a default resolution.
        threshold (optional): The threshold to use for creating clusters.
            If None, uses the resolutions' default threshold
            If an integer, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors

    Examples:
        ```python
        mb.match(
            select("datahub_companies", engine=engine),
            source=select("companies_house", engine=engine),
            source_pk="8534735",
            resolution_name="last_linker",
        )
        ```
    """
    if len(source) > 1:
        raise ValueError("Only one source can be matched at one time")
    source = source[0].address

    targets: list[Selector] = list(itertools.chain(*targets))
    targets = [t.address for t in targets]

    return _handler.match(
        targets=targets,
        source=source,
        source_pk=source_pk,
        resolution_name=resolution_name,
        threshold=threshold,
    )
