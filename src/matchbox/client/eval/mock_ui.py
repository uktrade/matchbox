"""Launcher for evaluation UI which writes some mock data."""

import atexit
import logging
import pathlib
import subprocess
import sys
from os import environ

import polars as pl
from polars.datatypes import String
from sqlalchemy import create_engine

from matchbox import index, make_model, query, select
from matchbox.client._handler import create_client
from matchbox.client._settings import settings as client_settings
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import source_from_tuple
from matchbox.common.graph import DEFAULT_RESOLUTION

MOCK_WH_FILE = "sqlite:///eval_mock.db"

logger = logging.getLogger("mock_ui_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def setup_mock_database():
    """Add some mock data to test the evaluation UI."""
    logger.info("Mocking DB")

    warehouse = create_engine(MOCK_WH_FILE)

    testkit_foo = source_from_tuple(
        data_tuple=(
            {"name": "Moore PLC", "postcode": "EH1"},
            {"name": "Moore PLC", "postcode": "EH8"},
            {"name": "Kuuhir Inc.", "postcode": "CM14"},
            {"name": "Kuvalis Group International", "postcode": "GU24"},
        ),
        data_keys=["1", "2", "3", "4"],
        name="foo",
        engine=warehouse,
    ).write_to_location()
    foo = testkit_foo.source

    index(source=foo)

    testkit_bar = source_from_tuple(
        data_tuple=(
            {"name": "Moore", "region": "Scotland"},
            {"name": "Kuuhir Incoroporated.", "region": "Essex"},
            {"name": "Kuvalis Group", "postcode": None},
        ),
        data_keys=["a", "b", "c"],
        name="bar",
        engine=warehouse,
    ).write_to_location()

    bar = testkit_bar.source_config
    bar.location.add_client(warehouse)
    index(source_config=bar)

    foo_df = query(select("foo", client=warehouse), return_type="polars")
    bar_df = query(select("bar", client=warehouse), return_type="polars")

    foo_df = foo_df.with_columns(
        pl.col("foo_name")
        .map_elements(lambda x: x.split(" ")[0], return_dtype=String)
        .alias("foo_name")
    )

    bar_df = bar_df.with_columns(
        pl.col("bar_name")
        .map_elements(lambda x: x.split(" ")[0], return_dtype=String)
        .alias("bar_name")
    )

    linker = make_model(
        name=DEFAULT_RESOLUTION,
        description="Linking model",
        model_class=DeterministicLinker,
        model_settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": ("l.foo_name = r.bar_name",),
        },
        left_data=pl.from_arrow(foo_df),
        left_resolution="foo",
        right_data=pl.from_arrow(bar_df),
        right_resolution="bar",
    )

    results = linker.run()
    results.to_matchbox()

    return warehouse.url


def cleanup_database():
    """Clean up mock data for the evaluation UI."""
    logger.info("Cleaning up DB")
    pathlib.Path(MOCK_WH_FILE).unlink(missing_ok=True)

    matchbox_client = create_client(settings=client_settings)
    matchbox_client.delete("/database", params={"certain": "true"})


if __name__ == "__main__":
    atexit.register(cleanup_database)
    warehouse_url = setup_mock_database()

    environ["MB__CLIENT__DEFAULT_WAREHOUSE"] = str(warehouse_url)
    environ["MB__CLIENT__USER"] = "scott.mcgregor"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/matchbox/client/eval/ui.py"]
    )
