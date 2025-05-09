"""Functions to select and retrieve data from the Matchbox server."""

import itertools
from typing import Any, Generator, Iterator, Literal, Self, get_args

import polars as pl
from polars import DataFrame as PolarsDataFrame
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import create_engine

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.common.db import QueryReturnType, ReturnTypeStr
from matchbox.common.dtos import ResolutionName, SourceResolutionName
from matchbox.common.graph import DEFAULT_RESOLUTION
from matchbox.common.logging import logger
from matchbox.common.sources import Match, SourceConfig, SourceField


class Selector(BaseModel):
    """A selector to choose a source and optionally a subset of columns to select."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: SourceConfig
    fields: list[SourceField]

    @field_validator("source", mode="after")
    @classmethod
    def ensure_credentials(cls: type[Self], source: SourceConfig) -> SourceConfig:
        """Ensure that the source has credentials set."""
        if not source.location.credentials:
            raise ValueError("Source credentials are not set")
        return source

    @property
    def qualified_identifier(self) -> str:
        """Get the qualified identifier for the source."""
        return self.source.name + "_" + self.source.identifier

    @property
    def qualified_fields(self) -> list[str]:
        """Get the qualified fields for the source."""
        return [self.source.name + "_" + field.name for field in self.fields]

    @classmethod
    def from_name_and_credentials(
        cls: type[Self],
        name: SourceResolutionName,
        credentials: Any,
        fields: list[str] | None = None,
    ) -> "Selector":
        """Create a Selector from a source name and location credentials.

        Args:
            name: The name of the source to select from
            credentials: The credentials to use for the source
            fields: A list of fields to select from the source
        """
        source = _handler.get_source(name=name)
        selected_fields: list[SourceField] = []
        if fields:
            for field in source.fields:
                if field.name in fields:
                    selected_fields.append(field)
        else:
            selected_fields = source.fields
        source.set_credentials(credentials=credentials)
        return cls(source=source, fields=selected_fields)


def select(
    *selection: SourceResolutionName | dict[SourceResolutionName, list[str]],
    credentials: Any | None = None,
) -> list[Selector]:
    """From one engine, builds and verifies a list of selectors.

    Can we used on any number of sources as long as they share the same credentials.

    Args:
        selection: The source relations to retrieve data from
        credentials: The credentials to use for the source. Datatype will depend on
            the source's location type. For example, a RelationalDBLocation will require
            a SQLAlchemy engine. If not provided, will populate with a SQLAlchemy engine
            from the default warehouse set in the environment variable
            `MB__CLIENT__DEFAULT_WAREHOUSE`

    Returns:
        A list of Selector objects

    Examples:
        ```python
        select("companies_house", credentials=engine)
        ```

        ```python
        select(
            {"companies_house": ["crn"], "hmrc_exporters": ["name"]}, credentials=engine
        )
        ```
    """
    if not credentials:
        if default_credentials := settings.default_warehouse:
            credentials = create_engine(default_credentials)
            logger.warning("Using default engine")
        else:
            raise ValueError(
                "Credentials need to be provided if "
                "`MB__CLIENT__DEFAULT_WAREHOUSE` is unset"
            )

    selectors = []
    for s in selection:
        if isinstance(s, str):
            selectors.append(
                Selector.from_name_and_credentials(name=s, credentials=credentials)
            )
        elif isinstance(s, dict):
            for name, fields in s.items():
                selectors.append(
                    Selector.from_name_and_credentials(
                        name=name,
                        credentials=credentials,
                        fields=fields,
                    )
                )
        else:
            raise ValueError("Selection specified in incorrect format")

    return selectors


def _process_query_result(
    data: PolarsDataFrame,
    selector: Selector,
    mb_ids: PolarsDataFrame,
) -> PolarsDataFrame:
    """Process query results by joining with matchbox IDs and filtering fields.

    Args:
        data: The raw data from the source
        selector: The selector with source and fields information
        mb_ids: The matchbox IDs

    Returns:
        The processed table with joined matchbox IDs and filtered fields
    """
    # Join data with matchbox IDs
    joined_table = data.join(
        other=mb_ids,
        left_on=selector.qualified_identifier,
        right_on="source_identifier",
        how="inner",
    )

    # Apply field filtering if needed
    if selector.fields:
        keep_cols = ["id"] + selector.qualified_fields
        match_cols = [col for col in joined_table.columns if col in keep_cols]
        return joined_table.select(match_cols)
    else:
        return joined_table


def _process_selectors(
    selectors: list[Selector],
    resolution: ResolutionName | None,
    threshold: int | None,
    batch_size: int | None,
) -> Iterator[PolarsDataFrame]:
    """Helper function to process selectors and return an iterator of results.

    For non-batched queries, turn this into a list.

    For batched queries, yield from it.
    """
    # Create iterators for each selector
    selector_iters: list[Generator[PolarsDataFrame, None, None]] = []

    def _process_batches(
        batches: Iterator[PolarsDataFrame],
        selector: Selector,
        mb_ids: PolarsDataFrame,
    ) -> Generator[PolarsDataFrame, None, None]:
        """Process and transform each batch of results."""
        for batch in batches:
            yield _process_query_result(batch, selector, mb_ids)

    for selector in selectors:
        mb_ids = pl.from_arrow(
            _handler.query(
                source=selector.source.name,
                resolution=resolution,
                threshold=threshold,
            )
        )

        raw_batches = selector.source.query(
            qualify_names=True,
            return_batches=True,
            batch_size=batch_size,
            return_type="polars",
        )

        if isinstance(raw_batches, PolarsDataFrame):
            raw_batches = [raw_batches]

        # Process and transform each batch
        selector_iters.append(
            _process_batches(
                batches=raw_batches,
                selector=selector,
                mb_ids=mb_ids,
            )
        )

    # Chain iterators if multiple selectors
    if len(selector_iters) == 1:
        batches_iter = selector_iters[0]
    else:
        # Interleave batches from different selectors
        batches_iter = itertools.chain.from_iterable(selector_iters)

    return batches_iter


def query(
    *selectors: list[Selector],
    resolution: ResolutionName | None = None,
    combine_type: Literal["concat", "explode", "set_agg"] = "concat",
    return_type: ReturnTypeStr = "pandas",
    threshold: int | None = None,
    batch_size: int | None = None,
    return_batches: bool = False,
) -> QueryReturnType | Iterator[QueryReturnType]:
    """Runs queries against the selected backend.

    Args:
        selectors: Each selector is the output of `select()`.
            This allows querying sources coming from different engines
        resolution (optional): The name of the resolution point to query
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
        batch_size (optional): The size of each batch when fetching data from the
            warehouse, which helps reduce memory usage and load on the database.
            Default is None.
        return_batches (optional): If True, returns an iterator of batches instead of a
            single combined result, which is useful for processing large datasets with
            limited memory. Default is False.

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
            resolution="last_linker",
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

    if return_batches and combine_type != "concat":
        raise ValueError("Batching is only supported for `combine_type='concat'`")

    selectors: list[Selector] = list(itertools.chain(*selectors))

    if not resolution and len(selectors) > 1:
        resolution = DEFAULT_RESOLUTION

    res = _process_selectors(
        selectors=selectors,
        resolution=resolution,
        threshold=threshold,
        batch_size=batch_size,
    )
    if return_batches:
        # Return an iterator of batches
        def generate_batches() -> Iterator[Generator[PolarsDataFrame, None, None]]:
            """Yield batches of data in the requested format."""
            for batch in res:
                match return_type:
                    case "pandas":
                        yield batch.to_pandas()
                    case "polars":
                        yield batch
                    case "arrow":
                        yield batch.to_arrow()

        return generate_batches()

    else:
        # Process all data and return a single result
        tables: list[PolarsDataFrame] = list(res)

        # Make sure we have some results
        if not tables:
            result = pl.DataFrame()
        else:
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
    *targets: list[SourceResolutionName],
    source: SourceResolutionName,
    identifier: str,
    resolution: ResolutionName = DEFAULT_RESOLUTION,
    threshold: int | None = None,
) -> list[Match]:
    """Matches IDs against the selected backend.

    Args:
        targets: A list of source resolutions to find identifiers in
        source: The source resolution the provided identifier belongs to
        identifier: The value to match from the source. Usually a primary key
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
            source_identifier="8534735",
            resolution="last_linker",
        )
        ```
    """
    for name in targets + [source]:
        _ = _handler.get_source(name=name)

    return _handler.match(
        targets=targets,
        source=source,
        identifier=identifier,
        resolution=resolution,
        threshold=threshold,
    )
