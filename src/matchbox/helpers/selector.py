from typing import Literal

from pandas import DataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy import Engine, inspect

from matchbox.common.db import Match, Source, get_schema_table_names
from matchbox.server import MatchboxDBAdapter, inject_backend


@inject_backend
def selector(
    backend: MatchboxDBAdapter, table: str, fields: list[str], engine: Engine
) -> dict[Source, list[str]]:
    """
    Takes the full name of a table and the fields you want to select from it,
    and arranges them in a dictionary parsable by query().

    Args:
        table: a table name in the form "schema.table".
        fields: a list of columns in the table
        engine: the engine to use to connect to
            your data warehouse

    Returns:
        A dictionary of the validated Source and fields
    """
    db_schema, db_table = get_schema_table_names(table, validate=True)
    dataset = backend.get_dataset(db_schema=db_schema, db_table=db_table, engine=engine)

    # Validate the fields
    inspector = inspect(engine)
    all_cols = set(
        column["name"]
        for column in inspector.get_columns(dataset.db_table, schema=dataset.db_schema)
    )
    selected_cols = set(fields)
    if not selected_cols <= all_cols:
        raise ValueError(f"{selected_cols.difference(all_cols)} not found in {dataset}")

    return {dataset: fields}


def selectors(*selector: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Builds individual selector dictionaries into one object appropriate for
    the query() function.

    Args:
        selector: any number of selectors

    Returns:
        A dictionary of selectors
    """
    return {k: v for d in (selector) for k, v in d.items()}


@inject_backend
def query(
    backend: MatchboxDBAdapter,
    selector: dict[str, list[str]],
    return_type: Literal["pandas", "arrow"] = None,
    model: str | None = None,
    threshold: float | dict[str, float] | None = None,
    limit: int | None = None,
) -> DataFrame | ArrowTable:
    """Runs queries against the selected backend.

    Args:
        backend: the backend to query
        selector: the tables and fields to query
        return_type: the form to return data in, one of "pandas" or "arrow"
            Defaults to pandas for ease of use
        model (optional): the model to use for filtering results
        threshold (optional): the threshold to use for creating clusters
            If None, uses the models' default threshold
            If a float, uses that threshold for the specified model, and the
            model's cached thresholds for its ancestors
            If a dictionary, expects a shape similar to model.ancestors, keyed
            by model name and valued by the threshold to use for that model. Will
            use these threshold values instead of the cached thresholds
        limit (optional): the number to use in a limit clause. Useful for testing

    Returns:
        Data in the requested return type
    """
    return backend.query(
        selector=selector,
        model=model,
        threshold=threshold,
        return_type="pandas" if not return_type else return_type,
        limit=limit,
    )


@inject_backend
def match(
    backend: MatchboxDBAdapter,
    source_id: str,
    source: str,
    target: str | list[str],
    model: str,
    threshold: float | dict[str, float] | None = None,
) -> Match | list[Match]:
    """Matches IDs against the selected backend.

    Args:
        backend: the backend to query
        source_id: The ID of the source to match.
        source: The name of the source dataset.
        target: The name of the target dataset(s).
        model: the model to use for filtering results
        threshold (optional): the threshold to use for creating clusters
            If None, uses the models' default threshold
            If a float, uses that threshold for the specified model, and the
            model's cached thresholds for its ancestors
            If a dictionary, expects a shape similar to model.ancestors, keyed
            by model name and valued by the threshold to use for that model. Will
            use these threshold values instead of the cached thresholds
    """
    return backend.match(
        source_id=source_id,
        source=source,
        target=target,
        model=model,
        threshold=threshold,
    )
