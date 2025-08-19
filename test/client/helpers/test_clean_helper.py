import polars as pl
import pytest
from sqlglot.errors import ParseError

from matchbox.client.queries import clean


@pytest.mark.parametrize(
    ("cleaning_dict", "expected_columns", "expected_values"),
    [
        pytest.param(
            {"name": "lower(foo_name)"},
            ["id", "name", "foo_status"],
            {"name": ["a", "b", "c"], "foo_status": ["active", "inactive", "active"]},
            id="basic_cleaning_with_passthrough",
        ),
        pytest.param(
            {"new_status": "foo_status", "lower_name": "lower(foo_name)"},
            ["id", "new_status", "lower_name"],
            {
                "new_status": ["active", "inactive", "active"],
                "lower_name": ["a", "b", "c"],
            },
            id="column_dropping_and_renaming",
        ),
    ],
)
def test_clean_basic_functionality(
    cleaning_dict: dict[str, str],
    expected_columns: list[str],
    expected_values: dict[str, list],
):
    """Test that clean() basic functionality works."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "foo_name": ["A", "B", "C"],
            "foo_status": ["active", "inactive", "active"],
        }
    )

    result = clean(test_data, cleaning_dict)
    assert len(result) == 3
    assert set(result.columns) == set(expected_columns)

    for column, values in expected_values.items():
        assert result[column].to_list() == values


def test_clean_none_returns_original():
    """Test that None cleaning_dict returns original data."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "foo_name": ["A", "B", "C"],
            "foo_status": ["active", "inactive", "active"],
        }
    )

    result = clean(test_data, None)
    assert set(result.columns) == {"id", "foo_name", "foo_status"}

    result_sorted = result.select(sorted(result.columns))
    test_data_sorted = test_data.select(sorted(test_data.columns))
    assert result_sorted.equals(test_data_sorted)


@pytest.mark.parametrize(
    ("extra_columns", "expected_columns"),
    [
        pytest.param(
            {
                "leaf_id": ["a", "b", "c"],
                "key": ["x", "y", "z"],
                "status": ["active", "inactive", "pending"],
            },
            ["id", "leaf_id", "key", "processed_value", "status"],
            id="both_special_columns",
        ),
        pytest.param(
            {"leaf_id": ["a", "b", "c"]},
            ["id", "leaf_id", "processed_value"],
            id="only_leaf_id",
        ),
        pytest.param(
            {"key": ["x", "y", "z"]}, ["id", "key", "processed_value"], id="only_key"
        ),
    ],
)
def test_clean_special_columns_handling(
    extra_columns: dict[str, list], expected_columns: list[str]
):
    """Test that leaf_id and key columns are automatically passed through."""
    base_data = {
        "id": [1, 2, 3],
        "value": [10, 20, 30],
    }

    test_data = pl.DataFrame({**base_data, **extra_columns})
    cleaning_dict = {"processed_value": "value * 2"}
    result = clean(test_data, cleaning_dict)

    assert set(result.columns) == set(expected_columns)
    assert result["processed_value"].to_list() == [20, 40, 60]

    # Check passthrough columns if they exist
    if "status" in extra_columns:
        assert result["status"].to_list() == ["active", "inactive", "pending"]


def test_clean_multiple_column_references():
    """Test expressions that reference multiple columns."""
    test_data = pl.DataFrame(
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

    result = clean(test_data, cleaning_dict)

    # first, last, and salary are dropped (used in expressions)
    assert set(result.columns) == {"id", "name", "high_earner"}
    assert result["name"].to_list() == ["John Doe", "Jane Smith", "Bob Johnson"]
    assert result["high_earner"].to_list() == [False, True, False]


def test_clean_complex_sql_expressions():
    """Test more complex SQL expressions."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "price": [10.5, 20.0, 15.75],
            "quantity": [2, 1, 3],
            "category": ["A", "B", "A"],
        }
    )

    cleaning_dict = {
        "total": "price * quantity",
        "expensive": "price > 15.0",
        "category_upper": "upper(category)",
    }

    result = clean(test_data, cleaning_dict)

    # Use set comparison for columns
    assert set(result.columns) == {"id", "total", "expensive", "category_upper"}
    assert result["total"].to_list() == [21.0, 20.0, 47.25]
    assert result["expensive"].to_list() == [False, True, True]
    assert result["category_upper"].to_list() == ["A", "B", "A"]


def test_clean_empty_cleaning_dict():
    """Test with empty cleaning dict."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
        }
    )

    result = clean(test_data, {})

    # Only id is selected, plus all unused columns (name, value)
    assert set(result.columns) == {"id", "value", "name"}

    result_sorted = result.select(sorted(result.columns))
    test_data_sorted = test_data.select(sorted(test_data.columns))
    assert result_sorted.equals(test_data_sorted)


def test_clean_invalid_sql():
    """Test that invalid SQL raises ParseError."""
    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        }
    )

    cleaning_dict = {
        "invalid": "foo bar baz",  # Invalid SQL
    }

    with pytest.raises(ParseError):
        clean(test_data, cleaning_dict)


def test_clean_column_passthrough():
    """Test that unused columns are passed through unchanged."""
    test_data = pl.DataFrame(
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

    result = clean(test_data, cleaning_dict)

    # name is dropped because it was used in cleaning_dict
    # age and city are passed through unchanged
    assert set(result.columns) == {"id", "full_name", "city", "age"}
    assert result["full_name"].to_list() == ["John", "Jane", "Bob"]
    assert result["age"].to_list() == [25, 30, 35]
    assert result["city"].to_list() == ["London", "Hull", "Stratford-upon-Avon"]
