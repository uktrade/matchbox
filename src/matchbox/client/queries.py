"""Functions to select and retrieve data from the Matchbox server."""

import itertools
from typing import Any, Iterator, Literal, Self, get_args

import duckdb
import polars as pl
from polars import DataFrame as PolarsDataFrame
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sqlalchemy import create_engine
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.common.db import QueryReturnType, ReturnTypeStr
from matchbox.common.graph import (
    DEFAULT_RESOLUTION,
    ResolutionName,
    SourceResolutionName,
)
from matchbox.common.logging import logger
from matchbox.common.sources import (
    Match,
    SourceConfig,
    SourceField,
)


class Selector(BaseModel):
    """A selector to choose a source and optionally a subset of columns to select."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: SourceConfig
    fields: list[SourceField]

    @property
    def qualified_key(self) -> str:
        """Get the qualified key name for the selected source."""
        return self.source.qualified_key

    @property
    def qualified_fields(self: Self) -> list[str]:
        """Get the qualified field names for the selected fields."""
        return self.source.f([field.name for field in self.fields])

    @field_validator("source", mode="after")
    @classmethod
    def ensure_client(cls: type[Self], source: SourceConfig) -> SourceConfig:
        """Ensure that the source has client set."""
        if not source.location.client:
            raise ValueError("Source client not set")
        return source

    @model_validator(mode="after")
    def ensure_fields(self: Self) -> Self:
        """Ensure that the fields are valid."""
        allowed_fields = set((self.source.key_field,) + self.source.index_fields)
        if set(self.fields) > allowed_fields:
            raise ValueError(
                "Selected fields are not valid for the source. "
                f"Valid fields are: {allowed_fields}"
            )
        return self

    @classmethod
    def from_name_and_client(
        cls: type[Self],
        name: SourceResolutionName,
        client: Any,
        fields: list[str] | None = None,
    ) -> "Selector":
        """Create a Selector from a source name and location client.

        Args:
            name: The name of the source to select from
            client: The client to use for the source
            fields: A list of fields to select from the source
        """
        source = _handler.get_source_config(name=name)
        field_map = {f.name: f for f in set((source.key_field,) + source.index_fields)}

        # Handle field selection
        if fields:
            selected_fields = [field_map[f] for f in fields]
        else:
            selected_fields = list(source.index_fields)  # Must actively select key

        source.location.add_client(client=client)
        return cls(source=source, fields=selected_fields)


def select(
    *selection: SourceResolutionName | dict[SourceResolutionName, list[str]],
    client: Any | None = None,
) -> list[Selector]:
    """From one location client, builds and verifies a list of selectors.

    Can be used on any number of sources as long as they share the same client.

    Args:
        selection: The source resolutions to retrieve data from
        client: The client to use for the source. Datatype will depend on
            the source's location type. For example, a RelationalDBLocation will require
            a SQLAlchemy engine. If not provided, will populate with a SQLAlchemy engine
            from the default warehouse set in the environment variable
            `MB__CLIENT__DEFAULT_WAREHOUSE`

    Returns:
        A list of Selector objects

    Examples:
        ```python
        select("companies_house", client=engine)
        ```

        ```python
        select({"companies_house": ["crn"], "hmrc_exporters": ["name"]}, client=engine)
        ```
    """
    if not client:
        if default_warehouse := settings.default_warehouse:
            client = create_engine(default_warehouse)
            logger.warning("Using default engine")
        else:
            raise ValueError(
                "Client needs to be provided if "
                "`MB__CLIENT__DEFAULT_WAREHOUSE` is unset"
            )

    selectors = []
    for s in selection:
        if isinstance(s, str):
            selectors.append(Selector.from_name_and_client(name=s, client=client))
        elif isinstance(s, dict):
            for name, fields in s.items():
                selectors.append(
                    Selector.from_name_and_client(
                        name=name,
                        client=client,
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
    return_leaf_id: bool,
) -> PolarsDataFrame:
    """Process query results by joining with matchbox IDs and filtering fields.

    Args:
        data: The raw data from the source
        selector: The selector with source and fields information
        mb_ids: The top-level matchbox IDs for the resolution
        return_leaf_id: Whether to return MB IDs of source clusters

    Returns:
        The processed table with joined matchbox IDs and filtered fields
    """
    # Join data with matchbox IDs
    joined_table = data.join(
        other=mb_ids,
        left_on=selector.qualified_key,
        right_on="key",
        how="inner",
    )

    # Apply field filtering if needed
    if selector.fields:
        base_fields = ["id"]
        if return_leaf_id:
            base_fields.append("leaf_id")
        keep_cols = base_fields + selector.qualified_fields
        match_cols = [col for col in joined_table.columns if col in keep_cols]
        return joined_table.select(match_cols)
    else:
        return joined_table


def _process_selectors(
    selectors: list[Selector],
    resolution: ResolutionName | None,
    return_leaf_id: bool,
    threshold: int | None,
    batch_size: int | None,
) -> Iterator[PolarsDataFrame]:
    """Helper function to process selectors and return an iterator of results.

    For non-batched queries, turn this into a list.

    For batched queries, yield from it.
    """
    selector_results: list[PolarsDataFrame] = []
    for selector in selectors:
        mb_ids = pl.from_arrow(
            _handler.query(
                source=selector.source.name,
                resolution=resolution,
                threshold=threshold,
                return_leaf_id=return_leaf_id,
            )
        )

        raw_batches = selector.source.query(
            qualify_names=True,
            batch_size=batch_size,
            return_type="polars",
        )

        processed_batches = [
            _process_query_result(
                data=b,
                selector=selector,
                mb_ids=mb_ids,
                return_leaf_id=return_leaf_id,
            )
            for b in raw_batches
        ]
        selector_results.append(pl.concat(processed_batches, how="vertical"))

    return selector_results


def query(
    *selectors: list[Selector],
    resolution: ResolutionName | None = None,
    combine_type: Literal["concat", "explode", "set_agg"] = "concat",
    return_leaf_id: bool = True,
    return_type: ReturnTypeStr = "pandas",
    threshold: int | None = None,
    batch_size: int | None = None,
) -> QueryReturnType:
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
        return_leaf_id: Whether matchbox IDs for source clusters should also be returned
        return_type: The form to return data in, one of "pandas" or "arrow"
            Defaults to pandas for ease of use
        threshold (optional): The threshold to use for creating clusters
            If None, uses the resolutions' default threshold
            If an integer, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors
        batch_size (optional): The size of each batch when fetching data from the
            warehouse, which helps reduce memory usage and load on the database.
            Default is None.

    Returns: Data in the requested return type (DataFrame or ArrowTable).


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

    """
    # Validate arguments
    if combine_type not in ("concat", "explode", "set_agg"):
        raise ValueError(f"combine_type of {combine_type} not valid")

    if return_type not in get_args(ReturnTypeStr):
        raise ValueError(f"return_type of {return_type} not valid")

    if not selectors:
        raise ValueError("At least one selector must be specified")

    selectors: list[Selector] = list(itertools.chain(*selectors))

    if not resolution and len(selectors) > 1:
        resolution = DEFAULT_RESOLUTION

    res = _process_selectors(
        selectors=selectors,
        resolution=resolution,
        return_leaf_id=return_leaf_id,
        threshold=threshold,
        batch_size=batch_size,
    )

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
        _ = _handler.get_source_config(name=name)

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
