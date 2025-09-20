"""Launcher for evaluation UI which writes some mock data."""

import atexit
import logging
import pathlib
import subprocess
import sys
from os import environ

from sqlalchemy import create_engine

from matchbox.client._handler import create_client
from matchbox.client._settings import settings as client_settings
from matchbox.client.dags import DAG
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import source_from_tuple

MOCK_WH_FILE = "sqlite:///eval_mock.db"

logger = logging.getLogger("mock_ui_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def setup_mock_database():
    """Add some mock data to test the evaluation UI."""
    logger.info("Mocking DB")

    warehouse = create_engine(MOCK_WH_FILE)

    # Write dummy sources
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

    dag = DAG("companies", new=True)
    foo = dag.source(**testkit_foo.into_dag())
    bar = dag.source(**testkit_bar.into_dag())

    foo.query(cleaning={"comp": f"split_part({foo.f('name')}, ' ', 1)"}).linker(
        bar.query(cleaning={"comp": f"split_part({bar.f('name')}, ' ', 1)"}),
        name="final",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.comp = r.comp"},
    )

    dag.run_and_sync()

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
