"""Tests for MatchboxLocalDuckDBDataMixin: replace-on-rerun.

Local stores replace-on-rerun, the deliberate inversion of the server's
write-once rule: iteration is the point locally.
"""

from functools import partial

import pytest
from sqlalchemy import Engine

from matchbox.client.base import MatchboxLocalDBAdapter
from matchbox.common.factories.scenarios import setup_scenario
from test.fixtures.db import LOCAL_BACKENDS


@pytest.mark.parametrize("backend", LOCAL_BACKENDS)
class TestReplaceOnRerun:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxLocalDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxLocalDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

    def test_reinsert_source_replaces_and_cascades(self) -> None:
        """Re-running a source's insert clears downstream model/resolver data."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn = dag_testkit.sources["crn"]
            model = dag_testkit.models["naive_test_crn"]
            resolver = dag_testkit.resolvers["resolver_naive_test_crn"]

            assert self.backend.get_model_data(model.path).num_rows > 0
            assert self.backend.get_resolver_data(resolver.resolver.path).num_rows > 0

            self.backend.insert_source_data(path=crn.path, data_hashes=crn.data_hashes)

            assert self.backend.get_model_data(model.path).num_rows == 0
            assert self.backend.get_resolver_data(resolver.resolver.path).num_rows == 0

            # But the source itself still has data, correctly replaced
            assert self.backend.query(source=crn.path).num_rows == crn.data.num_rows

    def test_reinsert_model_replaces_own_data_and_cascades(self) -> None:
        """Re-running a model's insert replaces its own edges and cascades."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            model = dag_testkit.models["naive_test_crn"]
            resolver = dag_testkit.resolvers["resolver_naive_test_crn"]

            before = self.backend.get_model_data(model.path).num_rows
            assert before > 0

            self.backend.insert_model_data(
                path=model.path, results=model.scores.to_arrow()
            )

            assert self.backend.get_model_data(model.path).num_rows == before
            # Cascaded: the downstream resolver's data is gone
            assert self.backend.get_resolver_data(resolver.resolver.path).num_rows == 0

    def test_reinsert_resolver_replaces_own_data(self) -> None:
        """Re-running a resolver's insert replaces its own assignments."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            resolver = dag_testkit.resolvers["resolver_naive_test_crn"]

            before = self.backend.get_resolver_data(resolver.resolver.path).num_rows
            assert before > 0

            self.backend.insert_resolver_data(
                path=resolver.resolver.path, data=resolver.resolver.results.to_arrow()
            )

            assert (
                self.backend.get_resolver_data(resolver.resolver.path).num_rows
                == before
            )
