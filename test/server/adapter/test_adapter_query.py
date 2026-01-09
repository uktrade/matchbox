"""Test the backend adapter's query functions."""

from functools import partial

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.arrow import SCHEMA_QUERY, SCHEMA_QUERY_WITH_LEAVES
from matchbox.common.dtos import Match
from matchbox.common.factories.entities import SourceEntity
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.docker
class TestMatchboxQueryBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

    def test_query_only_source(self) -> None:
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")

            df_crn_sample = self.backend.query(
                source=crn_testkit.resolution_path,
                limit=10,
            )

            assert isinstance(df_crn_sample, pa.Table)
            assert df_crn_sample.num_rows == 10

            df_crn_full = self.backend.query(source=crn_testkit.resolution_path)

            assert df_crn_full.num_rows == crn_testkit.data.num_rows
            assert df_crn_full.schema.equals(SCHEMA_QUERY)

    def test_query_return_leaf_ids(self) -> None:
        """Test querying data and additionally requesting leaf IDs."""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")

            df_crn_full = self.backend.query(
                source=crn_testkit.resolution_path, return_leaf_id=True
            )

            assert df_crn_full.num_rows == crn_testkit.data.num_rows
            assert df_crn_full.schema.equals(SCHEMA_QUERY_WITH_LEAVES)

    def test_query_with_dedupe_model(self) -> None:
        """Test querying data from a deduplication point of truth."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            df_crn = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.data.num_rows
            assert df_crn.schema.equals(SCHEMA_QUERY)

            linked = dag_testkit.source_to_linked["crn"]

            assert pc.count_distinct(df_crn["id"]).as_py() == len(
                linked.true_entity_subset("crn")
            )

    def test_query_with_link_model(self) -> None:
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "link") as dag_testkit:
            linker_name = "deterministic_naive_test_crn_naive_test_dh"
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            linker_testkit = dag_testkit.models.get(linker_name)

            df_crn = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=linker_testkit.resolution_path,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.data.num_rows
            assert df_crn.schema.equals(SCHEMA_QUERY)

            df_dh = self.backend.query(
                source=dh_testkit.resolution_path,
                point_of_truth=linker_testkit.resolution_path,
            )

            assert isinstance(df_dh, pa.Table)
            assert df_dh.num_rows == dh_testkit.data.num_rows
            assert df_dh.schema.equals(SCHEMA_QUERY)

            # Assumes CRN and DH come from same LinkedSourcesTestkit
            linked = dag_testkit.source_to_linked["crn"]

            all_ids = pa.concat_arrays(
                [df_crn["id"].combine_chunks(), df_dh["id"].combine_chunks()]
            )

            assert pc.count_distinct(all_ids).as_py() == len(
                linked.true_entity_subset("crn", "dh")
            )

    def test_threshold_query_with_link_model(self) -> None:
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "link") as dag_testkit:
            linker_name = "probabilistic_naive_test_crn_naive_test_cdms"
            crn_testkit = dag_testkit.sources.get("crn")
            cdms_testkit = dag_testkit.sources.get("cdms")
            linker_testkit = dag_testkit.models.get(linker_name)

            df_crn = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=linker_testkit.resolution_path,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.data.num_rows
            assert df_crn.schema.equals(SCHEMA_QUERY)

            df_cdms = self.backend.query(
                source=cdms_testkit.resolution_path,
                point_of_truth=linker_testkit.resolution_path,
            )

            assert isinstance(df_cdms, pa.Table)
            assert df_cdms.num_rows == cdms_testkit.data.num_rows
            assert df_cdms.schema.equals(SCHEMA_QUERY)

            # Assumes CRN and DH come from same LinkedSourcesTestkit
            linked = dag_testkit.source_to_linked["crn"]

            # Test query with threshold
            df_crn_threshold = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=linker_testkit.resolution_path,
                threshold=100,
            )
            df_cdms_threshold = self.backend.query(
                source=cdms_testkit.resolution_path,
                point_of_truth=linker_testkit.resolution_path,
                threshold=100,
            )
            threshold_ids = pa.concat_arrays(
                [
                    df_crn_threshold["id"].combine_chunks(),
                    df_cdms_threshold["id"].combine_chunks(),
                ]
            )

            # Query returns more clusters when threshold exceeds
            # true entity match probabilities
            assert pc.count_distinct(threshold_ids).as_py() > len(
                linked.true_entity_subset("crn", "cdms")
            )

    def test_match_one_to_many(self) -> None:
        """Test that matching data works when the target has many IDs."""
        with self.scenario(self.backend, "link") as dag_testkit:
            linker_name = "deterministic_naive_test_crn_naive_test_dh"
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            linker_testkit = dag_testkit.models.get(linker_name)

            # Assumes CRN and DH come from same LinkedSourcesTestkit
            linked = dag_testkit.source_to_linked["crn"]

            # A random one:many entity
            source_entity: SourceEntity = linked.find_entities(
                min_appearances={"crn": 2, "dh": 1},
                max_appearances={"dh": 1},
            )[0]

            res = self.backend.match(
                key=next(iter(source_entity.keys["dh"])),
                source=dh_testkit.resolution_path,
                targets=[crn_testkit.resolution_path],
                point_of_truth=linker_testkit.resolution_path,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == dh_testkit.source.resolution_path
            assert res[0].target == crn_testkit.source.resolution_path
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["dh"]
            assert res[0].target_id == source_entity.keys["crn"]

    def test_match_many_to_one(self) -> None:
        """Test that matching data works when the source has more possible IDs."""
        with self.scenario(self.backend, "link") as dag_testkit:
            linker_name = "deterministic_naive_test_crn_naive_test_dh"
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            linker_testkit = dag_testkit.models.get(linker_name)

            # Assumes CRN and DH come from same LinkedSourcesTestkit
            linked = dag_testkit.source_to_linked["crn"]

            # A random many:one entity
            source_entity: SourceEntity = linked.find_entities(
                min_appearances={"crn": 2, "dh": 1},
                max_appearances={"dh": 1},
            )[0]

            res = self.backend.match(
                key=next(iter(source_entity.keys["crn"])),
                source=crn_testkit.resolution_path,
                targets=[dh_testkit.resolution_path],
                point_of_truth=linker_testkit.resolution_path,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.resolution_path
            assert res[0].target == dh_testkit.source.resolution_path
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["crn"]
            assert res[0].target_id == source_entity.keys["dh"]

    def test_match_one_to_none(self) -> None:
        """Test that matching data works when the target has no IDs."""
        with self.scenario(self.backend, "link") as dag_testkit:
            linker_name = "deterministic_naive_test_crn_naive_test_dh"
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            linker_testkit = dag_testkit.models.get(linker_name)

            # Assumes CRN and DH come from same LinkedSourcesTestkit
            linked = dag_testkit.source_to_linked["crn"]

            # A random one:none entity
            source_entity: SourceEntity = linked.find_entities(
                min_appearances={"crn": 1},
                max_appearances={"dh": 0},
            )[0]

            res = self.backend.match(
                key=next(iter(source_entity.keys["crn"])),
                source=crn_testkit.resolution_path,
                targets=[dh_testkit.resolution_path],
                point_of_truth=linker_testkit.resolution_path,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.resolution_path
            assert res[0].target == dh_testkit.source.resolution_path
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["crn"]
            assert res[0].target_id == source_entity.keys.get("dh", set())

    def test_match_none_to_none(self) -> None:
        """Test that matching data works when the supplied key doesn't exist."""
        with self.scenario(self.backend, "link") as dag_testkit:
            linker_name = "deterministic_naive_test_crn_naive_test_dh"
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            linker_testkit = dag_testkit.models.get(linker_name)

            # Use a non-existent source key
            non_existent_key = "foo"

            res = self.backend.match(
                key=non_existent_key,
                source=crn_testkit.resolution_path,
                targets=[dh_testkit.resolution_path],
                point_of_truth=linker_testkit.resolution_path,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.resolution_path
            assert res[0].target == dh_testkit.source.resolution_path
            assert res[0].cluster is None
            assert res[0].source_id == set()
            assert res[0].target_id == set()

    def test_threshold_match_many_to_one(self) -> None:
        """Test that matching data works when the target has many IDs."""
        with self.scenario(self.backend, "link") as dag_testkit:
            linker_name = "probabilistic_naive_test_crn_naive_test_cdms"
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            linker_testkit = dag_testkit.models.get(linker_name)

            # Assumes CRN and DH come from same LinkedSourcesTestkit
            linked = dag_testkit.source_to_linked["crn"]

            # A random one:many entity
            source_entity: SourceEntity = linked.find_entities(
                min_appearances={"crn": 2, "dh": 1},
                max_appearances={"dh": 1},
            )[0]

            res = self.backend.match(
                key=next(iter(source_entity.keys["crn"])),
                source=crn_testkit.resolution_path,
                targets=[dh_testkit.resolution_path],
                point_of_truth=linker_testkit.resolution_path,
                threshold=100,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.resolution_path
            assert res[0].target == dh_testkit.source.resolution_path
            assert res[0].source_id == source_entity.keys["crn"]
            # Match does not return true target ids when threshold
            # exceeds match probability
            assert len(res[0].target_id) < len(source_entity.keys["dh"])
