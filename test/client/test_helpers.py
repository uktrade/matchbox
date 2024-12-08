import logging

from dotenv import find_dotenv, load_dotenv
from matchbox import process, query
from matchbox.clean import company_name, company_number
from matchbox.common.db import Source
from matchbox.helpers import (
    cleaner,
    cleaners,
    comparison,
    selector,
    selectors,
)
from matchbox.server.postgresql import MatchboxPostgres
from pandas import DataFrame

from ..fixtures.db import AddIndexedDataCallable

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

LOGGER = logging.getLogger(__name__)


def test_selectors(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    crn_wh = warehouse_data[0]
    select_crn = selector(
        table=str(crn_wh),
        fields=["id", "crn"],
        engine=crn_wh.database.engine,
    )

    duns_wh = warehouse_data[1]
    select_duns = selector(
        table=str(duns_wh),
        fields=["id", "duns"],
        engine=duns_wh.database.engine,
    )

    select_crn_duns = selectors(select_crn, select_duns)

    assert select_crn_duns is not None


def test_cleaners():
    cleaner_name = cleaner(function=company_name, arguments={"column": "company_name"})
    cleaner_number = cleaner(
        function=company_number, arguments={"column": "company_number"}
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    assert cleaner_name_number is not None


def test_process(
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse_data: list[Source],
):
    # Setup
    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    crn_wh = warehouse_data[0]
    select_crn = selector(
        table=str(crn_wh),
        fields=["crn", "company_name"],
        engine=crn_wh.database.engine,
    )
    crn = query(
        selector=select_crn,
        backend=matchbox_postgres,
        model=None,
        return_type="pandas",
    )

    cleaner_name = cleaner(
        function=company_name,
        arguments={"column": "test_crn_company_name"},
    )
    cleaner_number = cleaner(
        function=company_number,
        arguments={"column": "test_crn_crn"},
    )
    cleaner_name_number = cleaners(cleaner_name, cleaner_number)

    df_name_cleaned = process(data=crn, pipeline=cleaner_name_number)

    assert isinstance(df_name_cleaned, DataFrame)
    assert df_name_cleaned.shape[0] == 3000


def test_comparisons():
    comparison_name_id = comparison(
        sql_condition=(
            "l.company_name = r.company_name" " and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None
