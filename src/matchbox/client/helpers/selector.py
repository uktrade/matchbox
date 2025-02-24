import itertools
from typing import Literal
from warnings import warn

import pyarrow as pa
from pandas import ArrowDtype, DataFrame
from pydantic import BaseModel
from sqlalchemy import Engine

from matchbox.client import _handler
from matchbox.common.sources import Match, Source, SourceAddress


class Selector(BaseModel):
    source: Source
    fields: list[str] | None = None


def select(
    *selection: str | dict[str, str],
    engine: Engine,
) -> list[Selector]:
    """From one engine, builds and verifies a list of selectors.

    Args:
        selection: full source names and optionally a subset of columns to select
        engine: the engine to connect to the data warehouse hosting the source
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
                if not selected_cols <= indexed_cols:
                    warn(
                        "You are selecting columns that are not indexed in Matchbox",
                        stacklevel=2,
                    )
                selectors.append(Selector(source=source, fields=fields))
        else:
            raise ValueError("Selection specified in incorrect format")

    return selectors


def query(
    *selectors: list[Selector],
    resolution_name: str | None = None,
    return_type: Literal["pandas", "arrow"] = "pandas",
    threshold: int | None = None,
    limit: int | None = None,
) -> DataFrame | pa.Table:
    """Runs queries against the selected backend.

    Args:
        selectors: each selector is the output of `select()`
            This allows querying sources coming from different engines
        resolution_name (optional): the name of the resolution point to query
            It can only be `None` when querying from a single source, in which case the
            dataset resolution for that source will be used
        return_type: the form to return data in, one of "pandas" or "arrow"
            Defaults to pandas for ease of use
        threshold (optional): the threshold to use for creating clusters
            If None, uses the resolutions' default threshold
            If an integer, uses that threshold for the specified resolution, and the
            resolution's cached thresholds for its ancestors
        limit (optional): the number to use in a limit clause. Useful for testing

    Returns:
        Data in the requested return type

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
    """
    if not selectors:
        raise ValueError("At least one selector must be specified")

    selectors = list(itertools.chain(*selectors))

    if not resolution_name and len(selectors) > 1:
        raise ValueError(
            "A resolution name must be specified if querying more than one source"
        )

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
            match_cols = [col for col in joined_table.column_names if col in keep_cols]
            tables.append(joined_table.select(match_cols))
        else:
            tables.append(joined_table)

    # Combine results
    result = pa.concat_tables(tables, promote_options="default")

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
    resolution_name: str,
    threshold: int | None = None,
) -> list[Match]:
    """Matches IDs against the selected backend.

    Args:
        targets: each target is the output of `select()`
            This allows matching against sources coming from different engines
        source: The output of using `select()` on a single source.
        source_pk: The primary key value to match from the source.
        resolution_name: the resolution name to use for filtering results
        threshold (optional): the threshold to use for creating clusters
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
