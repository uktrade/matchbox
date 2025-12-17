"""Integration tests for all methodology classes."""

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from sqlalchemy import Engine

# Import configurator functions from methodology tests
from test.client.models.methodologies.test_dedupers_deterministic import DEDUPERS
from test.client.models.methodologies.test_linkers_deterministic import (
    configure_deterministic_linker,
    configure_splink_linker,
    configure_weighted_deterministic_linker,
)

from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.models.linkers.deterministic import DeterministicLinker
from matchbox.client.models.linkers.splinklinker import SplinkLinker
from matchbox.client.models.linkers.weighteddeterministic import (
    WeightedDeterministicLinker,
)
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.common.factories.sources import SourceTestkit
from matchbox.server.base import MatchboxDBAdapter

# Type aliases for configurator functions
DeduperConfigurator = Callable[[SourceTestkit], dict[str, Any]]
LinkerConfigurator = Callable[[SourceTestkit, SourceTestkit], dict[str, Any]]


@pytest.mark.docker
class TestE2EMethodologyIntegration:
    """Integration tests for all methodology classes with real pipeline execution."""

    @pytest.fixture(autouse=True)
    def setup(
        self,
        matchbox_postgres: MatchboxDBAdapter,
        sqla_sqlite_warehouse: Engine,
    ) -> None:
        """Set up scenario system for tests."""
        self.backend = matchbox_postgres
        self.warehouse = sqla_sqlite_warehouse

    def _clean_field(self, column: str) -> str:
        """Generate basic cleaning SQL."""
        return f"trim({column})"

    @pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
    def test_deduper_integration(
        self, Deduper: type[Deduper], configure_deduper: DeduperConfigurator
    ) -> None:
        """Test that dedupers work end-to-end."""
        with setup_scenario(self.backend, "index", self.warehouse) as dag_testkit:
            # Get the DAG from the testkit
            dag = dag_testkit.dag

            # Get a source that has duplicates (CRN has repetition in the scenario)
            source_testkit = dag_testkit.sources.get("crn")
            source = dag.get_source("crn")

            # Create settings and deduper
            settings = configure_deduper(source_testkit)

            dedupe_result = source.query(
                cleaning={
                    "company_name": self._clean_field(source.f("company_name")),
                    "crn": source.f("crn"),
                },
            ).deduper(
                name=f"test_{Deduper.__name__}",
                description=f"Integration test for {Deduper.__name__}",
                model_class=Deduper,
                model_settings=settings,
            )

            # Get results
            results = dedupe_result.run()

            # Basic integration checks
            assert results is not None, f"{Deduper.__name__} returned None"
            assert results.probabilities is not None
            assert isinstance(results.probabilities, pl.DataFrame)
            assert len(results.probabilities) >= 0

    @pytest.mark.parametrize(
        ("Linker", "configure_linker"),
        [
            pytest.param(
                DeterministicLinker,
                configure_deterministic_linker,
                id="DeterministicLinker",
            ),
            pytest.param(
                WeightedDeterministicLinker,
                configure_weighted_deterministic_linker,
                id="WeightedDeterministicLinker",
            ),
            pytest.param(
                SplinkLinker,
                configure_splink_linker,
                id="SplinkLinker",
            ),
        ],
    )
    def test_linker_integration(
        self, Linker: type[Linker], configure_linker: LinkerConfigurator
    ) -> None:
        """Test that linkers work end-to-end."""
        with setup_scenario(self.backend, "index", self.warehouse) as dag_testkit:
            # Get the DAG from the testkit
            dag = dag_testkit.dag

            # Get two sources that can be linked
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            crn_source = dag.get_source("crn")
            dh_source = dag.get_source("dh")

            # Create settings and linker
            settings = configure_linker(crn_testkit, dh_testkit)

            link_result = crn_source.query(
                cleaning={
                    "company_name": self._clean_field(crn_source.f("company_name")),
                },
            ).linker(
                dh_source.query(
                    cleaning={
                        "company_name": self._clean_field(dh_source.f("company_name")),
                    },
                ),
                name=f"test_{Linker.__name__}",
                description=f"Integration test for {Linker.__name__}",
                model_class=Linker,
                model_settings=settings,
            )

            # Get results
            results = link_result.run()

            # Basic integration checks
            assert results is not None, f"{Linker.__name__} returned None"
            assert results.probabilities is not None
            assert isinstance(results.probabilities, pl.DataFrame)
            assert len(results.probabilities) >= 0
