import ast
from functools import partial
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import pyarrow as pa
import pytest

from matchbox.client import clean
from matchbox.client.clean import drop
from matchbox.client.clean.steps import (
    clean_punctuation,
    expand_abbreviations,
    list_join_to_string,
    remove_stopwords,
    tokenise,
)
from matchbox.client.clean.utils import (
    alias,
    cleaning_function,
    select_cleaners,
    unnest_renest,
)
from matchbox.client.helpers.cleaner import cleaner, cleaners

"""
----------------------------
-- Feature cleaning tests --
----------------------------

To avoid bug-prone unit tests for complex cleaning functions like
cmf.clean.company_name, we instead test the constituent
parts, and the methods that build those parts into something complex.

To test company_name, therefore, we test the leaf functions in the stack:

* clean_comp_names
    * clean_punctuation
    * expand_abbreviations
    * tokenise
    * array_except
    * list_join_to_string

And the methods that assemble them:

* cleaning_function
* unnest_renest

See cleaning/ directory for more information on specific tests.

"""


def load_test_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dirty = pd.read_csv(Path(path, "dirty.csv"), converters={"list": ast.literal_eval})
    clean = pd.read_csv(Path(path, "clean.csv"), converters={"list": ast.literal_eval})

    dirty.columns = ["col"]
    clean.columns = ["col"]

    dirty = dirty.convert_dtypes(dtype_backend="pyarrow")
    clean = clean.convert_dtypes(dtype_backend="pyarrow")

    if isinstance(clean["col"][0], list):
        clean["col"] = clean["col"].astype(pd.ArrowDtype(pa.list_(pa.string())))
    if isinstance(dirty["col"][0], list):
        dirty["col"] = dirty["col"].astype(pd.ArrowDtype(pa.list_(pa.string())))

    return dirty, clean


def passthrough(input_column: str) -> str:
    """
    A passthrough cleaning function that does nothing. Helps test more complex
    building functions.
    """
    return f"{input_column}"


remove_stopwords_partial = partial(remove_stopwords, stopwords=["ltd", "plc"])
expand_abbreviations_partial = partial(
    expand_abbreviations, replacements={"co": "company", "ltd": "limited"}
)

cleaning_tests = [
    ("clean_punctuation", clean_punctuation),
    ("remove_stopwords", remove_stopwords_partial),
    ("list_join_to_string", list_join_to_string),
    ("tokenise", tokenise),
    ("expand_abbreviations", expand_abbreviations_partial),
]


@pytest.mark.parametrize("test", cleaning_tests)
def test_basic_functions(test: tuple[str, Callable], test_root_dir: Path):
    """
    Tests whether the basic cleaning functions do what they're supposed
    to. More complex functions should follow from here.
    """
    test_name = test[0]
    test_cleaning_function = test[1]

    dirty, clean = load_test_data(Path(test_root_dir, "client", "cleaning", test_name))

    cleaned = (
        duckdb.sql(
            f"""
        select
            {test_cleaning_function("col")} as col
        from
            dirty
    """
        )
        .arrow()
        .to_pandas(types_mapper=pd.ArrowDtype)
    )

    assert cleaned.equals(clean)


function_tests = [
    ("tokenise", [tokenise]),
    ("pass", [passthrough]),
    (
        "clean_comp_names",
        [
            clean_punctuation,
            expand_abbreviations_partial,
            tokenise,
            remove_stopwords_partial,
            list_join_to_string,
        ],
    ),
]


@pytest.mark.parametrize("test", function_tests)
def test_function(test: tuple[str, Callable], test_root_dir: Path):
    """
    Tests whether the cleaning function is accurately combining basic
    functions.
    """
    test_name = test[0]
    test_cleaning_function = cleaning_function(*test[1])

    dirty, clean = load_test_data(
        Path(test_root_dir, "client", "cleaning", "cleaning_function", test_name)
    )

    cleaned = test_cleaning_function(dirty, column="col")

    assert cleaned.equals(clean)


nest_unnest_tests = [
    ("expand_abbreviations", expand_abbreviations_partial),
    ("pass", passthrough),
]


@pytest.mark.parametrize("test", nest_unnest_tests)
def test_nest_unnest(test: tuple[str, Callable], test_root_dir: Path):
    """
    Tests whether the nest_unnest function is working.
    """
    test_name = test[0]
    test_cleaning_function = cleaning_function(test[1])

    dirty, clean = load_test_data(
        Path(test_root_dir, "client", "cleaning", "unnest_renest", test_name)
    )

    test_cleaning_function_arrayed = unnest_renest(test_cleaning_function)

    cleaned = test_cleaning_function_arrayed(dirty, column="col")

    # Handle arrays being unsortable
    cleaned["col"] = cleaned["col"].astype("string[pyarrow]")
    clean["col"] = clean["col"].astype("string[pyarrow]")

    cleaned = cleaned.sort_values(by="col").reset_index(drop=True)
    clean = clean.sort_values(by="col").reset_index(drop=True)

    assert cleaned.equals(clean)


def test_alias(test_root_dir: Path):
    """
    Tests whether the alias function is working.
    """
    test_cleaning_function = cleaning_function(passthrough)

    dirty, clean = load_test_data(Path(test_root_dir, "client", "cleaning", "alias"))

    alias_function = alias(test_cleaning_function, "foo")

    cleaned = alias_function(dirty, column="col")

    assert "foo" in cleaned.columns


def test_drop(test_root_dir: Path):
    """
    Tests whether the drop function is working.
    """
    dirty, clean = load_test_data(Path(test_root_dir, "client", "cleaning", "alias"))

    cleaned = drop(dirty, column="col")

    assert len(cleaned.columns) == 0


def test_select_cleaners():
    """Tests whether the select_cleaners function is working."""

    foo_cleaners = {
        "company_name": cleaner(
            clean.company_name,
            {"column": "company_name"},
        ),
    }

    bar_cleaners = {
        "postcode": cleaner(
            clean.postcode,
            {"column": "postcode"},
        ),
    }

    built_cleaners = select_cleaners(
        (foo_cleaners, ["company_name"]),
        (bar_cleaners, ["postcode"]),
    )

    regular_cleaners = cleaners(
        cleaner(
            clean.company_name,
            {"column": "company_name"},
        ),
        cleaner(
            clean.postcode,
            {"column": "postcode"},
        ),
    )

    assert built_cleaners == regular_cleaners


def test_remove_prefix():
    """Tests whether the remove_prefix function is working."""
    df = pd.DataFrame(
        {
            "prefix_col1": [1, 2, 3],
            "prefix_col2": [4, 5, 6],
            "other_col": ["a", "b", "c"],
        }
    )
    prefix = "prefix_"
    cleaned_df = clean.remove_prefix(df, column="", prefix=prefix)
    expected_df = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "other_col": ["a", "b", "c"],
        }
    )
    assert cleaned_df.equals(expected_df)
