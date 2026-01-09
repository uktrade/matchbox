"""Test the backend adapter's basic functions."""

from functools import partial

import pytest
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.docker
class TestMatchboxMainBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

    def test_properties(self) -> None:
        """Test that properties obey their protocol restrictions."""
        with self.scenario(self.backend, "index"):
            assert isinstance(self.backend.sources.list_all(), list)
            assert isinstance(self.backend.sources.count(), int)
            assert isinstance(self.backend.models.count(), int)
            assert isinstance(self.backend.source_clusters.count(), int)
            assert isinstance(self.backend.model_clusters.count(), int)
            assert isinstance(self.backend.all_clusters.count(), int)
            assert isinstance(self.backend.creates.count(), int)
            assert isinstance(self.backend.merges.count(), int)
            assert isinstance(self.backend.proposes.count(), int)
