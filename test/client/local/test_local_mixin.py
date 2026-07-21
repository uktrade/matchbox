"""Tests for MatchboxLocalDuckDBLocalMixin: raw data, query cache, cascade."""

from functools import partial

import pyarrow as pa
import pytest
from sqlalchemy import Engine

from matchbox.client.base import MatchboxLocalDBAdapter
from matchbox.common.dtos import CollectionName, RunID, StepPath
from matchbox.common.exceptions import MatchboxDataNotFound, MatchboxStepNotFoundError
from matchbox.common.factories.scenarios import setup_scenario
from test.fixtures.db import LOCAL_BACKENDS

# Placeholder collection/run: local ignores both, only the step name matters.
_NONEXISTENT_STEP = StepPath(
    collection=CollectionName("local_test"), run=RunID(1), name="nonexistent"
)


@pytest.mark.parametrize("backend", LOCAL_BACKENDS)
class TestRawData:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxLocalDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxLocalDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

    def test_round_trip(self) -> None:
        with self.scenario(self.backend, "index") as dag_testkit:
            crn = dag_testkit.sources["crn"].path

            table = pa.table({"key": ["k1", "k2"], "name": ["Alice", "Bob"]})
            self.backend.insert_raw_data(crn, table)

            fetched = self.backend.get_raw_data(crn)
            assert sorted(fetched.to_pylist(), key=lambda r: r["key"]) == sorted(
                table.to_pylist(), key=lambda r: r["key"]
            )

    def test_filters_by_keys(self) -> None:
        with self.scenario(self.backend, "index") as dag_testkit:
            crn = dag_testkit.sources["crn"].path

            table = pa.table(
                {"key": ["k1", "k2", "k3"], "name": ["Alice", "Bob", "Carl"]}
            )
            self.backend.insert_raw_data(crn, table)

            fetched = self.backend.get_raw_data(crn, keys=["k2"])
            assert fetched.to_pylist() == [{"key": "k2", "name": "Bob"}]

    def test_filters_by_empty_keys(self) -> None:
        with self.scenario(self.backend, "index") as dag_testkit:
            crn = dag_testkit.sources["crn"].path
            self.backend.insert_raw_data(
                crn, pa.table({"key": ["k1"], "name": ["Alice"]})
            )

            fetched = self.backend.get_raw_data(crn, keys=[])
            assert fetched.num_rows == 0

    def test_missing_raises(self) -> None:
        with self.scenario(self.backend, "index") as dag_testkit:
            dh = dag_testkit.sources["dh"].path

            with pytest.raises(MatchboxDataNotFound):
                self.backend.get_raw_data(dh)

    def test_unknown_step_raises(self) -> None:
        with pytest.raises(MatchboxStepNotFoundError):
            self.backend.get_raw_data(_NONEXISTENT_STEP)

    def test_replaces(self) -> None:
        with self.scenario(self.backend, "index") as dag_testkit:
            crn = dag_testkit.sources["crn"].path

            self.backend.insert_raw_data(
                crn, pa.table({"key": ["k1"], "name": ["Alice"]})
            )
            self.backend.insert_raw_data(
                crn, pa.table({"key": ["k2"], "name": ["Bob"]})
            )

            fetched = self.backend.get_raw_data(crn)
            assert fetched.to_pylist() == [{"key": "k2", "name": "Bob"}]


@pytest.mark.parametrize("backend", LOCAL_BACKENDS)
class TestQueryCache:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxLocalDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxLocalDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

    def test_round_trip(self) -> None:
        with self.scenario(self.backend, "index") as dag_testkit:
            crn = dag_testkit.sources["crn"].path

            table = pa.table({"id": [1, 2], "value": ["a", "b"]})
            self.backend.cache_query("key1", table, depends_on=[crn])

            fetched = self.backend.get_cached_query("key1")
            assert sorted(fetched.to_pylist(), key=lambda r: r["id"]) == sorted(
                table.to_pylist(), key=lambda r: r["id"]
            )

    def test_missing_returns_none(self) -> None:
        assert self.backend.get_cached_query("missing") is None

    def test_replaces(self) -> None:
        with self.scenario(self.backend, "index") as dag_testkit:
            crn = dag_testkit.sources["crn"].path

            self.backend.cache_query("key1", pa.table({"id": [1]}), depends_on=[crn])
            self.backend.cache_query("key1", pa.table({"id": [2]}), depends_on=[crn])

            fetched = self.backend.get_cached_query("key1")
            assert fetched.to_pylist() == [{"id": 2}]

    def test_unknown_dependency_raises(self) -> None:
        with pytest.raises(MatchboxStepNotFoundError):
            self.backend.cache_query(
                "key1", pa.table({"id": [1]}), depends_on=[_NONEXISTENT_STEP]
            )


@pytest.mark.parametrize("backend", LOCAL_BACKENDS)
class TestDropStepData:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxLocalDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxLocalDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

    def test_cascades(self) -> None:
        """Dropping a step's data cascades to its descendants."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            model = dag_testkit.models["naive_test_crn"]
            resolver = dag_testkit.resolvers["resolver_naive_test_crn"]

            assert self.backend.get_model_data(model.path).num_rows > 0
            assert self.backend.get_resolver_data(resolver.resolver.path).num_rows > 0

            self.backend.drop_step_data(model.path)

            assert self.backend.get_model_data(model.path).num_rows == 0
            assert self.backend.get_resolver_data(resolver.resolver.path).num_rows == 0

    def test_clears_dependent_cache_only(self) -> None:
        """Only cache entries depending on the dropped step are cleared."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn = dag_testkit.sources["crn"].path
            dh = dag_testkit.sources["dh"].path

            self.backend.cache_query(
                "dependent", pa.table({"a": [1]}), depends_on=[crn]
            )
            self.backend.cache_query(
                "independent", pa.table({"a": [2]}), depends_on=[dh]
            )

            self.backend.drop_step_data(crn)

            assert self.backend.get_cached_query("dependent") is None
            assert self.backend.get_cached_query("independent") is not None

    def test_raw_data_survives(self) -> None:
        """drop_step_data doesn't touch RawData - it's canonical, not a cache."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn = dag_testkit.sources["crn"].path
            self.backend.insert_raw_data(
                crn, pa.table({"key": ["k1"], "name": ["Alice"]})
            )

            self.backend.drop_step_data(crn)

            assert self.backend.get_raw_data(crn).num_rows == 1
