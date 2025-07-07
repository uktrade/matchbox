"""Library of default cleaning functions."""

from matchbox.client.clean.lib import (
    company_name,
    company_number,
    drop,
    extract_cdms_number_to_new,
    extract_company_number_to_new,
    extract_duns_number_to_new,
    postcode,
    postcode_to_area,
    remove_prefix,
)
from matchbox.client.clean.utils import (
    alias,
    cleaning_function,
    unnest_renest,
)

__all__ = (
    # Cleaning functions
    "company_name",
    "company_number",
    "drop",
    "extract_cdms_number_to_new",
    "extract_company_number_to_new",
    "extract_duns_number_to_new",
    "postcode",
    "postcode_to_area",
    # Utility functions
    "alias",
    "cleaning_function",
    "unnest_renest",
    "remove_prefix",
)
