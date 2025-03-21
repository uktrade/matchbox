"""Functions to select and retrieve data from the Matchbox server."""

import itertools
from typing import Iterator, Literal
from warnings import warn

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from pyarrow import compute as pc
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.common.graph import DEFAULT_RESOLUTION
from matchbox.common.logging import logger
from matchbox.common.sources import Match, Source, SourceAddress


class Selector(BaseModel):
    """A selector to choose a source and optionally a subset of columns to select."""

    source: Source
    fields: list[str] | None = None


def select(
    *selection: str | dict[str, str],
    engine: Engine | None = None,
    only_indexed: bool = True,
) -> list[Selector]:
    """From one engine, builds and verifies a list of selectors.

    Args:
        selection: Full source names and optionally a subset of columns to select
        engine: The engine to connect to the data warehouse hosting the source.
            If not provided, will use a connection string from the
            `MB__CLIENT__DEFAULT_WAREHOUSE` environment variable.
        only_indexed: Whether you intend to select indexed columns only. Will raise a
            warning if True and non-indexed columns are selected. Defaults to True.
            Non-indexed columns should only be selected if you're querying data for
            a purpose other than matching

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
            source = _handler.get_source(source_address).set_engine(engine)
            selectors.append(Selector(source=source))
        elif isinstance(s, dict):
            for full_name, fields in s.items():
                source_address = SourceAddress.compose(engine, full_name)
                source = _handler.get_source(source_address).set_engine(engine)

                warehouse_cols = set(source.to_table().columns.keys())
                selected_cols = set(fields)
                if not selected_cols <= warehouse_cols:
                    raise ValueError(
                        f"{selected_cols - warehouse_cols} not in {source_address}"
                    )

                indexed_cols = set(col.name for col in source.columns)
                if (not selected_cols <= indexed_cols) and only_indexed:
                    warn(
                        "You are selecting columns that are not indexed in Matchbox",
                        stacklevel=2,
                    )
                selectors.append(Selector(source=source, fields=fields))
        else:
            raise ValueError("Selection specified in incorrect format")

    return selectors


def _query_batched(
    selectors: list[Selector],
    sub_limits: list[int | None],
    resolution_name: str | None,
    threshold: int | None,
    return_type: Literal["pandas", "arrow"],
    batch_size: int | None,
) -> Iterator[DataFrame] | Iterator[pa.Table]:
    """Helper function that implements batched query processing.

    Returns an iterator yielding batches of results.
    """
    # Create iterators for each selector
    selector_iters = []

    for selector, sub_limit in zip(selectors, sub_limits, strict=True):
        # Get ids from matchbox
        mb_ids = _handler.query(
            source_address=selector.source.address,
            resolution_name=resolution_name,
            threshold=threshold,
            limit=sub_limit,
        )

        fields = None
        if selector.fields:
            fields = list(set(selector.fields))

        # Get batched data
        raw_batches = selector.source.to_arrow(
            fields=fields,
            pks=mb_ids["source_pk"].to_pylist(),
            iter_batches=True,
            batch_size=batch_size,
        )

        # Process and transform each batch
        def process_batches(batches, selector, mb_ids):
            for batch in batches:
                # Join and select columns
                joined_table = batch.join(
                    right_table=mb_ids,
                    keys=selector.source.format_column(selector.source.db_pk),
                    right_keys="source_pk",
                    join_type="inner",
                )

                if selector.fields:
                    keep_cols = ["id"] + [
                        selector.source.format_column(f) for f in selector.fields
                    ]
                    match_cols = [
                        col for col in joined_table.column_names if col in keep_cols
                    ]
                    yield joined_table.select(match_cols)
                else:
                    yield joined_table

        selector_iters.append(process_batches(raw_batches, selector, mb_ids))

    # Chain iterators if multiple selectors
    if len(selector_iters) == 1:
        batches_iter = selector_iters[0]
    else:
        # Interleave batches from different selectors
        batches_iter = itertools.chain(*selector_iters)

    # Convert each batch to the requested return type
    for batch in batches_iter:
        if return_type == "arrow":
            yield batch
        elif return_type == "pandas":
            yield batch.to_pandas(
                use_threads=True,
                split_blocks=True,
                self_destruct=True,
                types_mapper=ArrowDtype,
            )
        else:
            raise ValueError(f"return_type of {return_type} not valid")


def query(
    *selectors: list[Selector],
    resolution_name: str | None = None,
    combine_type: Literal["concat", "explode", "set_agg"] = "concat",
    return_type: Literal["pandas", "arrow"] = "pandas",
    threshold: int | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    return_batches: bool = False,
) -> DataFrame | pa.Table | Iterator[DataFrame] | Iterator[pa.Table]:
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

    Returns:
        If return_batches is False:
            Data in the requested return type (DataFrame or pa.Table)
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

    if return_type not in ("pandas", "arrow"):
        raise ValueError(f"return_type of {return_type} not valid")

    if not selectors:
        raise ValueError("At least one selector must be specified")

    selectors = list(itertools.chain(*selectors))

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
        # Return an iterator of batches
        return _query_batched(
            selectors=selectors,
            sub_limits=sub_limits,
            resolution_name=resolution_name,
            threshold=threshold,
            return_type=return_type,
            batch_size=batch_size,
        )
    else:
        # Process all data and return a single result
        tables = []
        for selector, sub_limit in zip(selectors, sub_limits, strict=True):
            # Get ids from matchbox
            mb_ids = _handler.query(
                source_address=selector.source.address,
                resolution_name=resolution_name,
                threshold=threshold,
                limit=sub_limit,
            )
            fields = None
            if selector.fields:
                fields = list(set(selector.fields))
            raw_data = selector.source.to_arrow(
                fields=fields,
                pks=mb_ids["source_pk"].to_pylist(),
                batch_size=batch_size,
            )

            # Join and select columns
            joined_table = raw_data.join(
                right_table=mb_ids,
                keys=selector.source.format_column(selector.source.db_pk),
                right_keys="source_pk",
                join_type="inner",
            )

            if selector.fields:
                keep_cols = ["id"] + [
                    selector.source.format_column(f) for f in selector.fields
                ]
                match_cols = [
                    col for col in joined_table.column_names if col in keep_cols
                ]
                tables.append(joined_table.select(match_cols))
            else:
                tables.append(joined_table)

        # Combine results based on combine_type
        if combine_type == "concat":
            result = pa.concat_tables(tables, promote_options="default")
        else:
            result = tables[0]
            for table in tables[1:]:
                result = result.join(table, keys=["id"], join_type="full outer")

            if combine_type == "set_agg":
                # Aggregate into lists
                aggregate_rule = [
                    (col, "distinct", pc.CountOptions(mode="only_valid"))
                    for col in result.column_names
                    if col != "id"
                ]
                result = result.group_by("id").aggregate(aggregate_rule)
                # Recover original column names
                rename_rule = {f"{col}_distinct": col for col, _, _ in aggregate_rule}
                result = result.rename_columns(rename_rule)

        # Return in requested format
        if return_type == "arrow":
            return result
        elif return_type == "pandas":
            return result.to_pandas(
                use_threads=True,
                split_blocks=True,
                self_destruct=True,
                types_mapper=ArrowDtype,
            )
        else:
            raise ValueError(f"return_type of {return_type} not valid")


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
    source = source[0].source.address

    targets = list(itertools.chain(*targets))
    targets = [t.source.address for t in targets]

    return _handler.match(
        targets=targets,
        source=source,
        source_pk=source_pk,
        resolution_name=resolution_name,
        threshold=threshold,
    )
