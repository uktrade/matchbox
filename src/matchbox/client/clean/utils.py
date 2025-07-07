"""Generic utilities for default cleaning functions."""

from typing import Any, Callable

import duckdb
import polars as pl

from matchbox.client.helpers.cleaner import cleaners

STOPWORDS = [
    "limited",
    "uk",
    "company",
    "international",
    "group",
    "of",
    "the",
    "inc",
    "and",
    "plc",
    "corporation",
    "llp",
    "pvt",
    "gmbh",
    "u k",
    "pte",
    "usa",
    "bank",
    "b v",
    "bv",
]

ABBREVIATIONS = {"co": "company", "ltd": "limited"}


def cleaning_function(*functions: Callable) -> Callable:
    """Takes a list of basic cleaning functions and composes them into a callable.

    Functions must be appropriate for a select statement.

    Only for use with cleaning methods that take a single column as their argument.
    Consider using functools.partial to coerce functions that need arguments into
    this shape.

    Args:
        functions: a list of functions appropriate for a select statement.
            See clean_basic for some examples
    """

    def cleaning_method(df: pl.DataFrame, column: str) -> pl.DataFrame:  # noqa: ARG001
        """Applies a series of cleaning functions to a specified column.

        Create a single SQL statement that applies all transformations
        This maintains proper DuckDB context for lambda expressions
        """
        nested_transform: str = column

        for f in functions:
            nested_transform: str = f(nested_transform)

        sql = f"""
            select
                *
                replace ({nested_transform} as {column})
            from
                df;
        """

        return duckdb.sql(sql).pl()

    return cleaning_method


def alias(function: Callable, alias: str) -> Callable:
    """Takes a cleaning function and aliases the output to a new column.

    Args:
        function: an outut from a cleaning_function function
        alias: the new column name to use
    """

    def cleaning_method(df: pl.DataFrame, column: str) -> pl.DataFrame:
        aliased_sql = f"""
            select
                *,
                {column} as {alias}
            from
                df;
        """
        return function(duckdb.sql(aliased_sql).pl(), alias)

    return cleaning_method


def unnest_renest(function: Callable) -> Callable:
    """Takes a cleaning function and adds unnesting and renesting either side of it.

    Useful for applying the same function to an array where there are sub-functions
    that also use arrays, blocking list_transform.

    Args:
        function: an outut from a cleaning_function function
    """

    def cleaning_method(df: pl.DataFrame, column: str) -> pl.DataFrame:
        unnest_sql = f"""
            select
                row_number() over () as nest_id,
                *
                replace (unnest({column}) as {column})
            from
                df;
        """
        processed = function(duckdb.sql(unnest_sql).pl(), column)

        any_value = [
            f"any_value({col}) as {col}"
            for col in processed.columns
            if col not in ["nest_id", column]
        ]
        if len(any_value) > 0:
            any_value_select = f"{', '.join(any_value)}, "
        else:
            any_value_select = ""

        renest_sql = f"""
            select
                {any_value_select}
                list({column}) as {column}
            from
                processed
            group by nest_id;
        """

        return duckdb.sql(renest_sql).pl()

    return cleaning_method


def select_cleaners(*sources: tuple[dict, list[str]]) -> dict[str, dict[str, Any]]:
    """Select cleaners from one or more (cleaner_dict, keys) pairs.

    Useful for storing a dictionary of cleaning funtions for a particular source,
    then building them dynamically based on the keys you want to use.

    Examples:
        ```python
        from matchbox.client import clean
        from matchbox.client.dags import LinkStep, StepInput
        from matchbox.client.helpers.cleaner import cleaner, cleaners

        foo_cleaners = {
            "company_name": cleaner(
                clean.company_name,
                {"column": foo.qualify_field("company_name")},
            ),
            "company_number": cleaner(
                clean.company_number,
                {"column": foo.qualify_field("company_number")},
            ),
        }

        bar_cleaners = {
            "company_name": cleaner(
                clean.company_name,
                {"column": bar.qualify_field("company_name")},
            ),
            "postcode": cleaner(
                clean.postcode,
                {"column": bar.qualify_field("postcode")},
            ),
        }

        my_linker = LinkStep(
            left=StepInput(
                select={
                    foo: ["company_name"],
                    bar: ["postcode"],
                },
                cleaners=select_cleaners(
                    (foo_cleaners, ["company_name"]),
                    (bar_cleaners, ["postcode"]),
                ),
                ...
            ),
            ...
        )
        ```

    Returns:
        Dictionary matching the return type of `cleaners()`
    """
    selected = {}
    for cleaner_dict, keys in sources:
        selected.update({k: v for k, v in cleaner_dict.items() if k in keys})
    return cleaners(*selected.values())
