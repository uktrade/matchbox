import os

import pytest
from pandas import DataFrame, Series, concat
from sqlalchemy.orm import Session

from cmf import make_deduper, process, query
from cmf.clean import company_name
from cmf.data import Models
from cmf.data import utils as du
from cmf.dedupers import Naive
from cmf.helpers import cleaner, cleaners, selector


@pytest.fixture(scope="function")
def query_clean_crn(db_engine):
    # Select
    select_crn = selector(
        table=f"{os.getenv('SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine[1],
    )

    crn = query(
        selector=select_crn, model=None, return_type="pandas", engine=db_engine[1]
    )

    # Clean
    col_prefix = f"{os.getenv('SCHEMA')}_crn_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_crn = cleaners(cleaner_name)

    crn_cleaned = process(data=crn, pipeline=cleaner_crn)

    return crn_cleaned


@pytest.fixture(scope="function")
def query_clean_duns(db_engine):
    # Select
    select_duns = selector(
        table=f"{os.getenv('SCHEMA')}.duns",
        fields=["duns", "company_name"],
        engine=db_engine[1],
    )

    duns = query(
        selector=select_duns, model=None, return_type="pandas", engine=db_engine[1]
    )

    # Clean
    col_prefix = f"{os.getenv('SCHEMA')}_duns_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_duns = cleaners(cleaner_name)

    duns_cleaned = process(data=duns, pipeline=cleaner_duns)

    return duns_cleaned


@pytest.fixture(scope="function")
def query_clean_cdms(db_engine):
    # Select
    select_cdms = selector(
        table=f"{os.getenv('SCHEMA')}.cdms",
        fields=["crn", "cdms"],
        engine=db_engine[1],
    )

    cdms = query(
        selector=select_cdms, model=None, return_type="pandas", engine=db_engine[1]
    )

    # No cleaning needed, see original data
    return cdms


def test_sha1_conversion(all_companies):
    """Tests SHA1 conversion works as expected."""
    sha1_series_1 = du.columns_to_value_ordered_sha1(
        data=all_companies,
        columns=["id", "company_name", "address", "crn", "duns", "cdms"],
    )

    assert isinstance(sha1_series_1, Series)
    assert len(sha1_series_1) == all_companies.shape[0]

    all_companies_reordered_top = (
        all_companies.head(500)
        .rename(
            columns={
                "company_name": "address",
                "address": "company_name",
                "duns": "crn",
                "crn": "duns",
            }
        )
        .filter(["id", "company_name", "address", "crn", "duns", "cdms"])
    )

    all_companies_reodered = concat(
        [all_companies_reordered_top, all_companies.tail(500)]
    )

    sha1_series_2 = du.columns_to_value_ordered_sha1(
        data=all_companies_reodered,
        columns=["id", "company_name", "address", "crn", "duns", "cdms"],
    )

    assert sha1_series_1.equals(sha1_series_2)


def test_naive_crn(db_engine, query_clean_crn, crn_companies):
    """Dedupes a table made entirely of 3000 duplicates."""
    col_prefix = f"{os.getenv('SCHEMA')}_crn_"

    # Confirm this is 3000 duplicates
    assert isinstance(query_clean_crn, DataFrame)
    assert query_clean_crn.shape[0] == 3000
    assert (
        query_clean_crn[[f"{col_prefix}company_name", f"{col_prefix}crn"]]
        .drop_duplicates()
        .shape[0]
        == 1000
    )

    # Confirm query table and original table are more or less the same
    # df1 = query_clean_crn[[
    #     '_team_cmf_crn_company_name',
    #     '_team_cmf_crn_crn'
    # ]].rename(columns={
    #     "_team_cmf_crn_company_name":"company_name",
    #     "_team_cmf_crn_crn":"crn",
    # })
    # df1 = df1.reset_index(names="id")
    # df2 = crn_companies[[
    #     'company_name',
    #     'crn'
    # ]]
    # print(df1)
    # print(df2)

    # query_clean_crn = query_clean_crn.reset_index(names="id_test")

    # print(query_clean_crn["data_sha1"])

    crn_naive_deduper = make_deduper(
        dedupe_run_name="basic_crn",
        description="Clean company name, CRN",
        deduper=Naive,
        deduper_settings={
            "id": "data_sha1",
            "unique_fields": [f"{col_prefix}company_name", f"{col_prefix}crn"],
        },
        data_source=f"{os.getenv('SCHEMA')}.crn",
        data=query_clean_crn,
    )
    # crn_naive_deduper = make_deduper(
    #     dedupe_run_name="basic_crn",
    #     description="Clean company name, CRN",
    #     deduper=Naive,
    #     deduper_settings={
    #         "id": "id",
    #         "unique_fields": [
    #             'company_name',
    #             'crn'
    #         ],
    #     },
    #     data_source=f"{os.getenv('SCHEMA')}.crn",
    #     data=crn_companies,
    # )

    crn_deduped = crn_naive_deduper()

    crn_deduped_df = crn_deduped.to_df()
    # We're at 9000. I think this is because each of the 3000 has two duplicates
    # But then why not 6000?
    print(crn_deduped_df)

    # print(query_clean_crn[[f"{col_prefix}company_name", f"{col_prefix}crn"]].head(3))

    assert isinstance(crn_deduped_df, DataFrame)
    # every row is a duplicate, order doesn't count, so 3000
    assert crn_deduped_df.shape[0] == 3000

    # df = crn_deduped._prep_to_cmf(crn_deduped.dataframe)

    # print(crn_deduped_df.head(3))
    # assert df == ""

    crn_deduped.to_cmf(engine=db_engine[1])

    with Session(db_engine[1]) as session:
        model = session.query(Models).filter_by(name="basic_crn").first()
        proposed_dedupes = model.proposes_dedupes()

    assert len(proposed_dedupes) == 3000  # successfully inserted 3000


# def test_naive_duns(db_engine, query_clean_duns):
#     """Dedupes a table made entirely of 500 unique items."""

#     col_prefix = f"{os.getenv('SCHEMA')}_duns_"
#     duns_naive_deduper = make_deduper(
#         dedupe_run_name="basic_duns",
#         description="Clean company name, DUNS",
#         deduper=Naive,
#         deduper_settings={
#             "id": f"{col_prefix}id",
#             "unique_fields": [f"{col_prefix}company_name", f"{col_prefix}duns"],
#         },
#         data_source=f"{os.getenv('SCHEMA')}.duns",
#         data=query_clean_duns,
#     )

#     duns_deduped = duns_naive_deduper()

#     duns_deduped_df = duns_deduped.to_df()

#     assert isinstance(duns_deduped_df, DataFrame)
#     assert duns_deduped_df.shape[0] == 0  # no duplicated rows

#     duns_deduped.to_cmf(engine=db_engine[1])

#     with Session(db_engine[1]) as session:
#         model = (
#             session.query(Models).filter_by(name="basic_duns").first()
#         )
#         proposed_dedupes = model.proposes_dedupes()

#     assert len(proposed_dedupes) == 0 # successfully inserted 0
