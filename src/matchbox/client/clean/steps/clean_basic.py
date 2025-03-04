"""Low-level primitives supporting default cleaning functions."""

from matchbox.client.clean.utils import ABBREVIATIONS, STOPWORDS


def remove_whitespace(column: str) -> str:
    """Removes all whitespaces.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return rf"""
        regexp_replace(
            {column},
            '\s',
            '',
            'g'
        )
    """


def punctuation_to_spaces(column: str) -> str:
    """Removes all punctuation and replaces with spaces.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
        regexp_replace(
            {column},
            '[^a-zA-Z0-9 ]+',
            ' ',
            'g'
        )
    """


def periods_to_nothing(column: str) -> str:
    """Removes periods and replaces with nothing (U.K. -> UK).

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
        regexp_replace(
            {column},
            '[.]+',
            '',
            'g'
        )
    """


def clean_punctuation(column: str) -> str:
    """Removes all punctuation and spaces, trim, lowercase.

    * Set to lower case
    * Remove punctuation
    * Multiple to single space replace.
    * Trim leading and trailing spaces.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return rf"""
    trim(
        regexp_replace(
            lower({punctuation_to_spaces(periods_to_nothing(column))}),
            '\s+',
            ' ',
            'g'
        )
    )
    """


def clean_punctuation_except_hyphens(column: str) -> str:
    """Revove all punctuation and spaces except hyphens, trim.

    Useful for cleaning reference numbers.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
        trim(
            regexp_replace(
                {column},
                '[^a-zA-Z0-9-]+',
                '',
                'g'
            )
        )
    """


def expand_abbreviations(
    column: str, replacements: dict[str, str] = ABBREVIATIONS
) -> str:
    """Expand abbreviations found in the column.

    Takes a dictionary where the keys are matches and the values are what to
    replace them with.

    Matches only when term is surrounded by regex word boundaries.

    Args:
        column: the name of the column to clean
        replacements: a dictionary where keys are matches and values are
            what the replace them with

    Returns:
        String to insert into SQL query
    """
    replace_stack = ""
    for i, (match, replacement) in enumerate(replacements.items()):
        if i == 0:
            replace_stack = rf"""
                regexp_replace(
                    lower({column}),
                    '\b({match})\b',
                    '{replacement}',
                    'g'
                )
            """
        else:
            replace_stack = rf"""
                regexp_replace(
                    {replace_stack},
                    '\b({match})\b',
                    '{replacement}',
                    'g'
                )
            """

    return replace_stack


def tokenise(column: str) -> str:
    """Split the text in column into an array.

    Uses any char that is _not_ alphanumeric as delimeter.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
    regexp_split_to_array(
        trim({column}),
        '[^a-zA-Z0-9]+'
    )
    """


def dedupe_and_sort(column: str) -> str:
    """De-duplicate an array of tokens and sort alphabetically.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
    array(
        select distinct unnest(
            {column}
        ) tokens
        order by tokens
    )
    """


def remove_notnumbers_leadingzeroes(column: str) -> str:
    """Remove any char that is not a number, then remove all leading zeroes.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
    regexp_replace(
        regexp_replace(
            {column},
            '[^0-9]',
            '',
            'g'
        ),
        '^0+',
        ''
    )
    """


def array_except(column: str, terms_to_remove: list[str]) -> str:
    """Remove terms from an array.

    Args:
        column: The name of the column to treat
        terms_to_remove: A list of terms to remove

    Returns:
        String to insert into SQL query
    """
    return f"""
    array_filter(
        {column},
        x -> not array_contains({terms_to_remove}, x)
    )
    """


def array_intersect(column: str, terms_to_keep: list[str]) -> str:
    """Filter an array to only keep terms in a list.

    Args:
        column: The name of the column to treat
        terms_to_keep: A list of terms to keep

    Returns:
        String to insert into SQL query
    """
    return f"""
    array_filter(
        {column},
        x -> array_contains({terms_to_keep}, x)
    )
    """


def remove_stopwords(column: str, stopwords: list[str] = STOPWORDS) -> str:
    """A thin optinionated wrapper for array_except to clean the global stopwords list.

    Args:
        column: The name of the column to treat
        stopwords: A list of terms to remove

    Returns:
        String to insert into SQL query
    """
    return f"""
        {array_except(column, terms_to_remove=stopwords)}
    """


def regex_remove_list_of_strings(column: str, list_of_strings: list[str]) -> str:
    """Remove a list of strings from a column using regex.

    Args:
        column: The name of the column to treat
        list_of_strings: A list of strings to remove

    Returns:
        String to insert into SQL query
    """
    to_remove = "|".join(list_of_strings)
    return rf"""
    trim(
        regexp_replace(
            regexp_replace(
                lower({column}),
                '{to_remove}',
                '',
                'g'
            ),
        '\s{(2,)}',
        ' ',
        'g'
        )
    )
    """


def regex_extract_list_of_strings(column: str, list_of_strings: list[str]) -> str:
    """Extract a list of strings from a column using regex.

    Args:
        column: The name of the column to treat
        list_of_strings: A list of strings to extract

    Returns:
        String to insert into SQL query
    """
    to_extract = "|".join(list_of_strings)
    return f"""
    regexp_extract_all({column}, '{to_extract}', 0)
    """


def list_join_to_string(column: str, seperator: str = " ") -> str:
    """Join a list of strings into a single string.

    Args:
        column: The name of the column to treat
        seperator: The string to use to join the list

    Returns:
        String to insert into SQL query
    """
    return f"""list_aggr({column},
        'string_agg',
        '{seperator}'
    )
    """


def get_postcode_area(column: str) -> str:
    """Extract the postcode area from a column.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
        regexp_extract(
            {column},
            '^[a-zA-Z][a-zA-Z]?'
        )
    """


def get_low_freq_char_sig(column: str) -> str:
    """Removes letters with a frequency of 5% or higher, and spaces.

    See https://en.wikipedia.org/wiki/Letter_frequency for details.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
        regexp_replace(
            lower({column}),
            '[rhsnioate ]+',
            '',
            'g'
        )
    """


def filter_cdms_number(column: str) -> str:
    """Filter out non-CDMS numbers.

    Returns a CASE WHEN filter on the specified column that will match only CDMS
    numbers. Must be either:

    * 6 or 12 digits long
    * Start with '000'
    * Start with 'ORG-'

    Will return false positives on some CRN numbers when they are 8 digits long and
    begin with '000'.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"""
        case
            when (
                length({column}) = 12
                or length({column}) = 6
                or left({column}, 3) = '000'
                or left({column}, 4) = 'ORG-'
            )
            then {column}
            else null
        end
    """


def filter_company_number(column: str) -> str:
    """Filter out non-Companies House numbers.

    Returns a CASE WHEN filter on the specified column that will match only Companies
    House numbers, CRNs.

    Uses regex derived from:
    https://gist.github.com/rob-murray/01d43581114a6b319034732bcbda29e1

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    crn_regex = (
        r"^(((AC|CE|CS|FC|FE|GE|GS|IC|LP|NC|NF|NI|NL|NO|NP|OC|OE|PC|R0|RC|"
        r"SA|SC|SE|SF|SG|SI|SL|SO|SR|SZ|ZC|\d{2})\d{6})|((IP|SP|RS)[A-Z\d]"
        r"{6})|(SL\d{5}[\dA]))$"
    )
    return f"""
        case
            when regexp_full_match({column}, '{crn_regex}')
            then {column}
            else null
        end
    """


def filter_duns_number(column: str) -> str:
    """Filter out non-DUNS numbers.

    Returns a CASE WHEN filter on the specified column that will match only a
    Dun & Bradstreet DUNS number. Must be both:

    * 9 characters
    * Numeric

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return rf"""
        case
            when (
                length({column}) = 9
                and regexp_full_match({column}, '\d+')
            )
            then {column}
            else null
        end
    """


def to_upper(column: str) -> str:
    """All characters to uppercase.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"upper({column})"


def to_lower(column: str) -> str:
    """All characters to lowercase.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"lower({column})"


def trim(column: str) -> str:
    """Remove leading and trailing whitespace.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return f"trim({column})"


def get_digits_only(column: str) -> str:
    """Extract digits only, including nonconsecutive.

    Args:
        column: The name of the column to treat

    Returns:
        String to insert into SQL query
    """
    return rf"""
        nullif(
            list_aggregate(
                regexp_extract_all({column}, '\d+'),
                'string_agg',
                ''
            ),
            ''
        )
    """
