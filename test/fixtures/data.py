import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv
from matchbox import process, query
from matchbox.clean import company_name
from matchbox.helpers import cleaner, cleaners, selector
from pandas import DataFrame
from sqlalchemy.engine import Engine

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)
TEST_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def test_root_dir() -> Path:
    return TEST_ROOT


@pytest.fixture(scope="session")
def all_companies(test_root_dir: Path) -> DataFrame:
    """
    Raw, correct company data. Uses UUID as ID to replicate Data Workspace.
    1,000 entries.
    """
    df = pd.read_csv(
        Path(test_root_dir, "data", "all_companies.csv"), encoding="utf-8"
    ).reset_index(names="id")
    df["id"] = df["id"].apply(lambda x: uuid.UUID(int=x))
    return df


@pytest.fixture(scope="session")
def crn_companies(all_companies: DataFrame) -> DataFrame:
    """
    Company data split into CRN version.

    Company name has Limited, UK and Company added -- our first three stopwords.

    Tests a link/dedupe situation with dirty duplicates.

    3,000 entries, 1,000 unique.
    """
    df_raw = all_companies.filter(["company_name", "crn"])
    df_crn = pd.concat(
        [
            df_raw.assign(company_name=lambda df: df["company_name"] + " Limited"),
            df_raw.assign(company_name=lambda df: df["company_name"] + " UK"),
            df_raw.assign(company_name=lambda df: df["company_name"] + " Company"),
        ]
    )

    df_crn["id"] = range(df_crn.shape[0])
    df_crn = df_crn.filter(["id", "company_name", "crn"])
    df_crn["id"] = df_crn["id"].apply(lambda x: uuid.UUID(int=x))
    df_crn = df_crn.convert_dtypes(dtype_backend="pyarrow")

    return df_crn


@pytest.fixture(scope="session")
def duns_companies(all_companies: DataFrame) -> DataFrame:
    """
    Company data split into DUNS version.

    Data is clean.

    Tests a link/dedupe situation with no duplicates.

    500 entries.
    """
    df_duns = (
        all_companies.filter(["company_name", "duns"])
        .sample(n=500)
        .reset_index(drop=True)
        .reset_index(names="id")
        .convert_dtypes(dtype_backend="pyarrow")
    )
    df_duns["id"] = df_duns["id"].apply(lambda x: uuid.UUID(int=x))

    return df_duns


@pytest.fixture(scope="session")
def cdms_companies(all_companies: DataFrame) -> DataFrame:
    """
    Company data split into CDMS version.

    All rows are repeated twice.

    Tests a link/dedupe situation with clean duplicates: edge case in prod,
    but exists in some of the HMRC tables.

    2,000 entries, 1,000 unique.
    """
    df_cdms = pd.DataFrame(
        np.repeat(all_companies.filter(["crn", "cdms"]).values, 2, axis=0)
    )
    df_cdms.columns = ["crn", "cdms"]

    df_cdms.reset_index(names="id", inplace=True)
    df_cdms["id"] = df_cdms["id"].apply(lambda x: uuid.UUID(int=x))
    df_cdms = df_cdms.convert_dtypes(dtype_backend="pyarrow")

    return df_cdms


@pytest.fixture(scope="function")
def query_clean_crn(db_engine: Engine) -> DataFrame:
    # Select
    select_crn = selector(
        table=f"{os.getenv('MB_SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine,
    )

    crn = query(selector=select_crn, model=None, return_type="pandas", engine=db_engine)

    # Clean
    col_prefix = f"{os.getenv('MB_SCHEMA')}_crn_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_crn = cleaners(cleaner_name)

    crn_cleaned = process(data=crn, pipeline=cleaner_crn)

    return crn_cleaned


@pytest.fixture(scope="function")
def query_clean_duns(db_engine: Engine) -> DataFrame:
    # Select
    select_duns = selector(
        table=f"{os.getenv('MB_SCHEMA')}.duns",
        fields=["duns", "company_name"],
        engine=db_engine,
    )

    duns = query(
        selector=select_duns, model=None, return_type="pandas", engine=db_engine
    )

    # Clean
    col_prefix = f"{os.getenv('MB_SCHEMA')}_duns_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_duns = cleaners(cleaner_name)

    duns_cleaned = process(data=duns, pipeline=cleaner_duns)

    return duns_cleaned


@pytest.fixture(scope="function")
def query_clean_cdms(db_engine: Engine) -> DataFrame:
    # Select
    select_cdms = selector(
        table=f"{os.getenv('MB_SCHEMA')}.cdms",
        fields=["crn", "cdms"],
        engine=db_engine,
    )

    cdms = query(
        selector=select_cdms, model=None, return_type="pandas", engine=db_engine
    )

    # No cleaning needed, see original data
    return cdms


@pytest.fixture(scope="function")
def query_clean_crn_deduped(db_engine: Engine) -> DataFrame:
    # Select
    select_crn = selector(
        table=f"{os.getenv('MB_SCHEMA')}.crn",
        fields=["crn", "company_name"],
        engine=db_engine,
    )

    crn = query(
        selector=select_crn,
        model=f"naive_{os.getenv('MB_SCHEMA')}.crn",
        return_type="pandas",
        engine=db_engine,
    )

    # Clean
    col_prefix = f"{os.getenv('MB_SCHEMA')}_crn_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_crn = cleaners(cleaner_name)

    crn_cleaned = process(data=crn, pipeline=cleaner_crn)

    return crn_cleaned


@pytest.fixture(scope="function")
def query_clean_duns_deduped(db_engine: Engine) -> DataFrame:
    # Select
    select_duns = selector(
        table=f"{os.getenv('MB_SCHEMA')}.duns",
        fields=["duns", "company_name"],
        engine=db_engine,
    )

    duns = query(
        selector=select_duns,
        model=f"naive_{os.getenv('MB_SCHEMA')}.duns",
        return_type="pandas",
        engine=db_engine,
    )

    # Clean
    col_prefix = f"{os.getenv('MB_SCHEMA')}_duns_"
    cleaner_name = cleaner(
        function=company_name, arguments={"column": f"{col_prefix}company_name"}
    )
    cleaner_duns = cleaners(cleaner_name)

    duns_cleaned = process(data=duns, pipeline=cleaner_duns)

    return duns_cleaned


@pytest.fixture(scope="function")
def query_clean_cdms_deduped(db_engine: Engine) -> DataFrame:
    # Select
    select_cdms = selector(
        table=f"{os.getenv('MB_SCHEMA')}.cdms",
        fields=["crn", "cdms"],
        engine=db_engine,
    )

    cdms = query(
        selector=select_cdms,
        model=f"naive_{os.getenv('MB_SCHEMA')}.cdms",
        return_type="pandas",
        engine=db_engine,
    )

    # No cleaning needed, see original data
    return cdms
