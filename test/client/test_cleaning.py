"""Tests for cleaning functions using a programmatic approach.

This module provides tests for the cleaning functions in matchbox.client.clean.

To test complex cleaning functions like company_name, we test:

1. The leaf functions in the stack (e.g., clean_punctuation, tokenise)
2. The methods that assemble them (e.g., cleaning_function, unnest_renest)
"""

from functools import partial
from typing import Callable

import polars as pl
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
from test.client.cleaning.utils import (
    create_test_case,
    run_cleaner_test,
    run_composed_test,
)


def passthrough(input_column: str) -> str:
    """
    A passthrough cleaning function that does nothing. Helps test more complex
    building functions.
    """
    return f"{input_column}"


# Setup fixtures for reusable components
@pytest.fixture
def stopwords_remover() -> Callable[[str], str]:
    """Create a stopwords remover with predefined stopwords."""
    return partial(remove_stopwords, stopwords=["ltd", "plc"])


@pytest.fixture
def abbreviation_expander() -> Callable[[str], str]:
    """Create an abbreviation expander with predefined replacements."""
    return partial(
        expand_abbreviations, replacements={"co": "company", "ltd": "limited"}
    )


# --------------------------
# -- Basic Function Tests --
# --------------------------


def test_clean_punctuation_basic():
    """Test clean_punctuation with basic inputs."""
    cleaned, success = run_cleaner_test(
        cleaner_func=clean_punctuation,
        input_data=["!@#$%^&*()_+=-{}[]\"|\\'\\§±<>,./?`~`£__foo"],
        expected_output=["foo"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_clean_punctuation_spaces():
    """Test clean_punctuation with spaces and periods."""
    cleaned, success = run_cleaner_test(
        cleaner_func=clean_punctuation,
        input_data=["        bar.       "],
        expected_output=["bar"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_clean_punctuation_case():
    """Test clean_punctuation with uppercase."""
    cleaned, success = run_cleaner_test(
        cleaner_func=clean_punctuation,
        input_data=["BAZ"],
        expected_output=["baz"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_tokenise():
    """Test tokenise with multiple words."""
    cleaned, success = run_cleaner_test(
        cleaner_func=tokenise,
        input_data=["one two three"],
        expected_output=[["one", "two", "three"]],
    )
    assert success, f"Failed with output: {cleaned}"


def test_remove_stopwords_basic(stopwords_remover: Callable[[str], str]):
    """Test remove_stopwords with basic inputs."""
    cleaned, success = run_cleaner_test(
        cleaner_func=stopwords_remover,
        input_data=[["company", "ltd"]],
        expected_output=[["company"]],
    )
    assert success, f"Failed with output: {cleaned}"


def test_remove_stopwords_middle(stopwords_remover: Callable[[str], str]):
    """Test remove_stopwords with stopword in the middle."""
    cleaned, success = run_cleaner_test(
        cleaner_func=stopwords_remover,
        input_data=[["hello", "plc", "world"]],
        expected_output=[["hello", "world"]],
    )
    assert success, f"Failed with output: {cleaned}"


def test_list_join_basic():
    """Test list_join_to_string with basic inputs."""
    cleaned, success = run_cleaner_test(
        cleaner_func=list_join_to_string,
        input_data=[["hello", "world"]],
        expected_output=["hello world"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_list_join_multi():
    """Test list_join_to_string with multiple words."""
    cleaned, success = run_cleaner_test(
        cleaner_func=list_join_to_string,
        input_data=[["one", "two", "three"]],
        expected_output=["one two three"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_expand_abbreviations_both(abbreviation_expander: Callable[[str], str]):
    """Test expand_abbreviations with both replacements."""
    cleaned, success = run_cleaner_test(
        cleaner_func=abbreviation_expander,
        input_data=["co ltd"],
        expected_output=["company limited"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_expand_abbreviations_single(abbreviation_expander: Callable[[str], str]):
    """Test expand_abbreviations with one replacement."""
    cleaned, success = run_cleaner_test(
        cleaner_func=abbreviation_expander,
        input_data=["co only"],
        expected_output=["company only"],
    )
    assert success, f"Failed with output: {cleaned}"


# -----------------------------
# -- Composed Function Tests --
# -----------------------------


def test_function_tokenise():
    """Test composed tokenise function."""
    func = cleaning_function(tokenise)
    cleaned, success = run_composed_test(
        composed_func=func,
        input_data=["hello world"],
        expected_output=[["hello", "world"]],
    )
    assert success, f"Failed with output: {cleaned}"


def test_function_passthrough():
    """Test composed passthrough function."""
    func = cleaning_function(passthrough)
    cleaned, success = run_composed_test(
        composed_func=func,
        input_data=["unchanged text"],
        expected_output=["unchanged text"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_function_clean_names(
    abbreviation_expander: Callable[[str], str], stopwords_remover: Callable[[str], str]
):
    """Test complex composed cleaning function."""
    func = cleaning_function(
        clean_punctuation,
        abbreviation_expander,
        tokenise,
        stopwords_remover,
        list_join_to_string,
    )
    cleaned, success = run_composed_test(
        composed_func=func,
        input_data=["co. ltd!@#"],
        expected_output=["company limited"],
    )
    assert success, f"Failed with output: {cleaned}"


# ---------------------------
# -- Nest/Unnest Function --
# ---------------------------


def test_nest_unnest_abbreviations(abbreviation_expander: Callable[[str], str]):
    """Test unnest_renest with abbreviation expansion."""
    test_func = cleaning_function(abbreviation_expander)
    unnested_func = unnest_renest(test_func)

    dirty, clean = create_test_case(
        input_data=[["co ltd", "another co"]],
        expected_output=[["company limited", "another company"]],
    )
    cleaned = unnested_func(dirty, column="col")

    assert all((cleaned == clean)["col"].to_list()), f"Failed with output: {cleaned}"


def test_nest_unnest_passthrough():
    """Test unnest_renest with passthrough function."""
    test_func = cleaning_function(passthrough)
    unnested_func = unnest_renest(test_func)

    dirty, clean = create_test_case(
        input_data=[["text1", "text2"]],
        expected_output=[["text1", "text2"]],
    )
    cleaned = unnested_func(dirty, column="col")

    assert all((cleaned == clean)["col"].to_list()), f"Failed with output: {cleaned}"


# ---------------------
# -- Utility Tests --
# ---------------------


def test_alias():
    """Test the alias function."""
    test_func = cleaning_function(passthrough)
    alias_func = alias(test_func, "foo")

    dirty, _ = create_test_case(
        input_data=["test text"],
        expected_output=["test text"],
    )
    cleaned = alias_func(dirty, column="col")

    assert "foo" in cleaned.columns, f"Alias column not found in {cleaned.columns}"


def test_drop():
    """Test the drop function."""
    dirty, _ = create_test_case(
        input_data=["text to drop"],
        expected_output=[""],
    )
    cleaned = drop(dirty, column="col")

    assert len(cleaned.columns) == 0, (
        f"Column was not dropped, found: {cleaned.columns}"
    )


def test_select_cleaners():
    """Tests whether the select_cleaners function is working."""

    foo_cleaners = {
        "company_name": cleaner(
            clean.company_name,
            {"column": "company_name"},
        ),
        "company_number": cleaner(
            clean.company_number,
            {"column": "company_number"},
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
    assert len(built_cleaners) == 2


def test_remove_prefix():
    """Tests whether the remove_prefix function is working."""
    df = pl.DataFrame(
        {
            "prefix_col1": [1, 2, 3],
            "prefix_col2": [4, 5, 6],
            "other_col": ["a", "b", "c"],
        }
    )
    prefix = "prefix_"
    cleaned_df = clean.remove_prefix(df, column="", prefix=prefix)
    expected_df = pl.DataFrame(
        {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "other_col": ["a", "b", "c"],
        }
    )
    assert cleaned_df.equals(expected_df)
