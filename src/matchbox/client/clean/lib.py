"""Implementation of default cleaning functions."""

from functools import partial

from pandas import DataFrame

from matchbox.client.clean import steps
from matchbox.client.clean import utils as cu


def company_name(
    df: DataFrame,
    column: str,
    column_secondary: str = None,
    stopwords: str = cu.STOPWORDS,
) -> DataFrame:
    """Standard cleaning function for company names.

    * Lower case, remove punctuation & tokenise the company name into an array
    * Extract tokens into: 'unusual' and 'stopwords'. Dedupe. Sort alphabetically
    * Untokenise the unusual words back to a string

    Args:
        df: a dataframe
        column: a column containing the company's main name
        column_secondary: a column containing an array of the company's
            secondary names
        stopwords: a list of stopwords to use for this clean

    Returns:
        dataframe: the same as went in, but cleaned
    """
    remove_stopwords = partial(steps.remove_stopwords, stopwords=stopwords)

    clean_primary = cu.cleaning_function(
        steps.clean_punctuation,
        steps.expand_abbreviations,
        steps.tokenise,  # returns array
        remove_stopwords,
        steps.list_join_to_string,  # returns col
    )

    clean_secondary = cu.unnest_renest(clean_primary)

    df = clean_primary(df, column)

    if column_secondary is not None:
        df = clean_secondary(df, column_secondary)

    return df


def company_number(df: DataFrame, column: str) -> DataFrame:
    """Remove non-numbers, and then leading zeroes.

    Args:
        df: a dataframe
        column: a column containing a company number

    Returns:
        dataframe: the same as went in, but cleaned
    """
    clean_number = cu.cleaning_function(steps.remove_notnumbers_leadingzeroes)

    df = clean_number(df, column)

    return df


def postcode(df: DataFrame, column: str) -> DataFrame:
    """Removes all punctuation, converts to upper, removes all spaces.

    Args:
        df: a dataframe
        column: a column containing a postcode

    Returns:
        dataframe: the same as went in, but cleaned

    """
    clean_postcode = cu.cleaning_function(
        steps.punctuation_to_spaces, steps.to_upper, steps.remove_whitespace
    )

    df = clean_postcode(df, column)

    return df


def postcode_to_area(df: DataFrame, column: str) -> DataFrame:
    """Extracts postcode area from a postcode.

    Args:
        df: a dataframe
        column: a column containing a postcode

    Returns:
        dataframe: the same as went in, but cleaned
    """
    extract_area = cu.cleaning_function(steps.get_postcode_area)

    df = extract_area(df, column)

    return df


def extract_company_number_to_new(
    df: DataFrame, column: str, new_column: str
) -> DataFrame:
    """Detects the Companies House CRN in a column and moves it to a new column.

    Args:
        df: a dataframe
        column: a column containing some company numbers
        new_column: the name of the column to add

    Returns:
        dataframe: the same as went in with a new column for CRNs
    """
    clean_crn = cu.cleaning_function(
        steps.clean_punctuation_except_hyphens,
        steps.to_upper,
        steps.filter_company_number,
    )

    clean_crn_aliased = cu.alias(clean_crn, alias=new_column)

    df = clean_crn_aliased(df, column)

    return df


def extract_duns_number_to_new(
    df: DataFrame, column: str, new_column: str
) -> DataFrame:
    """Detects the Dun & Bradstreet DUNS nuber in a column and moves it to a new column.

    Args:
        df: a dataframe
        column: a column containing some DUNS numbers
        new_column: the name of the column to add

    Returns:
        dataframe: the same as went in with a new column for DUNs numbers
    """
    clean_duns = cu.cleaning_function(
        steps.clean_punctuation_except_hyphens, steps.to_upper, steps.filter_duns_number
    )

    clean_duns_aliased = cu.alias(clean_duns, alias=new_column)

    df = clean_duns_aliased(df, column)

    return df


def extract_cdms_number_to_new(
    df: DataFrame, column: str, new_column: str
) -> DataFrame:
    """Detects the CDMS nuber in a column and moves it to a new column.

    Args:
        df: a dataframe
        column: a column containing some CDMS numbers
        new_column: the name of the column to add

    Returns:
        dataframe: the same as went in with a new column for CDMS numbers
    """
    clean_cdms = cu.cleaning_function(
        steps.clean_punctuation_except_hyphens, steps.to_upper, steps.filter_cdms_number
    )

    clean_cdms_aliased = cu.alias(clean_cdms, alias=new_column)

    df = clean_cdms_aliased(df, column)

    return df


def drop(df: DataFrame, column: str) -> DataFrame:
    """Drops the column from the dataframe.

    Args:
        df: a dataframe
        column: a column

    Returns:
        dataframe: the same as went in without the column
    """
    return df.drop(columns=[column])
