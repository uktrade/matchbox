"""Tests for cleaning functions using a programmatic approach.

This module provides tests for the cleaning functions in matchbox.client.clean.

To test complex cleaning functions like company_name, we test:

1. The leaf functions in the stack (e.g., clean_punctuation, tokenise)
2. The methods that assemble them (e.g., cleaning_function, unnest_renest)
"""

from functools import partial
from typing import Callable

import pytest

from matchbox.client.clean import drop
from matchbox.client.clean.steps import (
    clean_punctuation,
    expand_abbreviations,
    list_join_to_string,
    remove_stopwords,
    tokenise,
)
from matchbox.client.clean.utils import alias, cleaning_function, unnest_renest
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
        clean_punctuation,
        ["!@#$%^&*()_+=-{}[]\"|\\'\\§±<>,./?`~`£__foo"],
        ["foo"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_clean_punctuation_spaces():
    """Test clean_punctuation with spaces and periods."""
    cleaned, success = run_cleaner_test(
        clean_punctuation,
        ["        bar.       "],
        ["bar"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_clean_punctuation_case():
    """Test clean_punctuation with uppercase."""
    cleaned, success = run_cleaner_test(
        clean_punctuation,
        ["BAZ"],
        ["baz"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_tokenise_basic():
    """Test tokenise with basic inputs."""
    cleaned, _ = run_cleaner_test(
        tokenise,
        ["hello world"],
        [["hello", "world"]],
    )

    # Convert numpy array to list for comparison
    result = cleaned["col"].iloc[0]
    expected = ["hello", "world"]

    assert list(result) == expected, f"Got {list(result)} instead of {expected}"


def test_tokenise_multi_words():
    """Test tokenise with multiple words."""
    cleaned, _ = run_cleaner_test(
        tokenise,
        ["one two three"],
        [["one", "two", "three"]],
    )

    result = cleaned["col"].iloc[0]
    expected = ["one", "two", "three"]

    assert list(result) == expected, f"Got {list(result)} instead of {expected}"


def test_remove_stopwords_basic(stopwords_remover: Callable[[str], str]):
    """Test remove_stopwords with basic inputs."""
    cleaned, _ = run_cleaner_test(
        stopwords_remover,
        [["company", "ltd"]],
        [["company"]],
    )

    result = cleaned["col"].iloc[0]
    expected = ["company"]

    assert list(result) == expected, f"Got {list(result)} instead of {expected}"


def test_remove_stopwords_middle(stopwords_remover: Callable[[str], str]):
    """Test remove_stopwords with stopword in the middle."""
    cleaned, _ = run_cleaner_test(
        stopwords_remover,
        [["hello", "plc", "world"]],
        [["hello", "world"]],
    )

    result = cleaned["col"].iloc[0]
    expected = ["hello", "world"]

    assert list(result) == expected, f"Got {list(result)} instead of {expected}"


def test_list_join_basic():
    """Test list_join_to_string with basic inputs."""
    cleaned, success = run_cleaner_test(
        list_join_to_string,
        [["hello", "world"]],
        ["hello world"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_list_join_multi():
    """Test list_join_to_string with multiple words."""
    cleaned, success = run_cleaner_test(
        list_join_to_string,
        [["one", "two", "three"]],
        ["one two three"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_expand_abbreviations_both(abbreviation_expander: Callable[[str], str]):
    """Test expand_abbreviations with both replacements."""
    cleaned, success = run_cleaner_test(
        abbreviation_expander,
        ["co ltd"],
        ["company limited"],
    )
    assert success, f"Failed with output: {cleaned}"


def test_expand_abbreviations_single(abbreviation_expander: Callable[[str], str]):
    """Test expand_abbreviations with one replacement."""
    cleaned, success = run_cleaner_test(
        abbreviation_expander,
        ["co only"],
        ["company only"],
    )
    assert success, f"Failed with output: {cleaned}"


# -----------------------------
# -- Composed Function Tests --
# -----------------------------


def test_function_tokenise():
    """Test composed tokenise function."""
    func = cleaning_function(tokenise)
    cleaned, success = run_composed_test(
        func,
        ["hello world"],
        [["hello", "world"]],
    )
    assert success, f"Failed with output: {cleaned}"


def test_function_passthrough():
    """Test composed passthrough function."""
    func = cleaning_function(passthrough)
    expected = "unchanged text"
    cleaned, _ = run_composed_test(
        func,
        ["unchanged text"],
        [expected],
    )

    result = cleaned["col"].iloc[0]

    assert result == expected, f"Got {result} instead of {expected}"


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
    expected = "company limited"
    cleaned, _ = run_composed_test(
        func,
        ["co. ltd!@#"],
        [expected],
    )

    result = cleaned["col"].iloc[0]

    assert result == expected, f"Got {result} instead of {expected}"


# ---------------------------
# -- Nest/Unnest Function --
# ---------------------------


def test_nest_unnest_abbreviations(abbreviation_expander: Callable[[str], str]):
    """Test unnest_renest with abbreviation expansion."""
    test_func = cleaning_function(abbreviation_expander)
    unnested_func = unnest_renest(test_func)

    dirty, clean = create_test_case(
        [["co ltd", "another co"]],
        [["company limited", "another company"]],
    )
    cleaned = unnested_func(dirty, column="col")

    # Convert to Python lists for deep comparison
    cleaned_list = cleaned["col"].tolist()
    clean_list = clean["col"].tolist()

    # Compare the lists directly - they should be identical nested lists
    assert cleaned_list == clean_list, (
        f"Failed with output: {cleaned_list}, expected: {clean_list}"
    )


def test_nest_unnest_passthrough():
    """Test unnest_renest with passthrough function."""
    test_func = cleaning_function(passthrough)
    unnested_func = unnest_renest(test_func)

    dirty, clean = create_test_case(
        [["text1", "text2"]],
        [["text1", "text2"]],
    )
    cleaned = unnested_func(dirty, column="col")

    # Convert to Python lists for deep comparison
    cleaned_list = cleaned["col"].tolist()
    clean_list = clean["col"].tolist()

    # Compare the lists directly - they should be identical nested lists
    assert cleaned_list == clean_list, (
        f"Failed with output: {cleaned_list}, expected: {clean_list}"
    )


# ---------------------
# -- Utility Tests --
# ---------------------


def test_alias():
    """Test the alias function."""
    test_func = cleaning_function(passthrough)
    alias_func = alias(test_func, "foo")

    dirty, _ = create_test_case(["test text"], ["test text"])
    cleaned = alias_func(dirty, column="col")

    assert "foo" in cleaned.columns, f"Alias column not found in {cleaned.columns}"


def test_drop():
    """Test the drop function."""
    dirty, _ = create_test_case(["text to drop"], [""])
    cleaned = drop(dirty, column="col")

    assert len(cleaned.columns) == 0, (
        f"Column was not dropped, found: {cleaned.columns}"
    )
