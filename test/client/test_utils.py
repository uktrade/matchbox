from pandas import Series, concat

from matchbox.common.factories.sources import source_factory
from matchbox.common.hash import fields_to_value_ordered_hash


def test_hash_conversion():
    """Tests SHA1 conversion works as expected."""
    source_testkit = source_factory(
        features=[
            {"name": "company_name", "base_generator": "company"},
            {"name": "address", "base_generator": "address"},
            {
                "name": "crn",
                "base_generator": "bothify",
                "parameters": (("text", "???-###-???-###"),),
            },
            {
                "name": "duns",
                "base_generator": "bothify",
                "parameters": (("text", "??######"),),
            },
        ],
    )
    all_companies = source_testkit.data.to_pandas()
    sha1_series_1 = fields_to_value_ordered_hash(
        data=all_companies,
        fields=["id", "key", "company_name", "address", "crn", "duns"],
    )

    assert isinstance(sha1_series_1, Series)
    assert len(sha1_series_1) == all_companies.shape[0]

    all_companies_reordered_top = (
        all_companies.head(len(all_companies) // 2)
        .rename(
            columns={
                "company_name": "address",
                "address": "company_name",
                "duns": "crn",
                "crn": "duns",
            }
        )
        .filter(["id", "key", "company_name", "address", "crn", "duns"])
    )

    all_companies_reodered = concat(
        [all_companies_reordered_top, all_companies.tail(len(all_companies) // 2)]
    )

    sha1_series_2 = fields_to_value_ordered_hash(
        data=all_companies_reodered,
        fields=["id", "key", "company_name", "address", "crn", "duns"],
    )

    assert sha1_series_1.equals(sha1_series_2)
