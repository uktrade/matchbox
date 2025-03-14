from functools import partial

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from sqlalchemy import Engine

from matchbox.common.dtos import ModelAncestor, ModelMetadata, ModelType
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.entities import SourceEntity
from matchbox.common.factories.sources import SourceTestkit
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import HASH_FUNC
from matchbox.common.sources import Match, SourceAddress
from matchbox.server.base import MatchboxDBAdapter

from ..fixtures.db import setup_scenario

backends = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(scope="function")
def backend_instance(request: pytest.FixtureRequest, backend: str):
    """Create a fresh backend instance for each test."""
    backend_obj = request.getfixturevalue(backend)
    backend_obj.clear(certain=True)
    return backend_obj


@pytest.mark.parametrize("backend", backends)
@pytest.mark.docker
class TestMatchboxBackend:
    @pytest.fixture(autouse=True)
    def setup(self, backend_instance: str, sqlite_warehouse: Engine):
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

    def test_properties(self):
        """Test that properties obey their protocol restrictions."""
        with self.scenario(self.backend, "index"):
            assert isinstance(self.backend.datasets.list(), list)
            assert isinstance(self.backend.datasets.count(), int)
            assert isinstance(self.backend.models.count(), int)
            assert isinstance(self.backend.data.count(), int)
            assert isinstance(self.backend.clusters.count(), int)
            assert isinstance(self.backend.creates.count(), int)
            assert isinstance(self.backend.merges.count(), int)
            assert isinstance(self.backend.proposes.count(), int)

    def test_validate_ids(self):
        """Test validating data IDs."""
        with self.scenario(self.backend, "dedupe") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn = self.backend.query(
                source_address=crn_testkit.source.address,
                resolution_name="naive_test.crn",
            )

            ids = df_crn["id"].to_pylist()
            assert len(ids) > 0
            self.backend.validate_ids(ids=ids)

            with pytest.raises(MatchboxDataNotFound):
                self.backend.validate_ids(ids=[-6])

    def test_validate_hashes(self):
        """Test validating data hashes."""
        with self.scenario(self.backend, "dedupe") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn = self.backend.query(
                source_address=crn_testkit.source.address,
                resolution_name="naive_test.crn",
            )

            ids = df_crn["id"].to_pylist()
            hashes = list(self.backend.cluster_id_to_hash(ids=ids).values())
            assert len(hashes) > 0
            self.backend.validate_hashes(hashes=hashes)

            with pytest.raises(MatchboxDataNotFound):
                self.backend.validate_hashes(
                    hashes=[HASH_FUNC(b"nonexistent").digest()]
                )

    def test_cluster_id_to_hash(self):
        """Test getting ID to Cluster hash lookup from the database."""
        with self.scenario(self.backend, "dedupe") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn = self.backend.query(
                source_address=crn_testkit.source.address,
                resolution_name="naive_test.crn",
            )

            ids = df_crn["id"].to_pylist()
            assert len(ids) > 0

            hashes = self.backend.cluster_id_to_hash(ids=ids)
            assert len(hashes) == len(set(ids))
            assert set(ids) == set(hashes.keys())
            assert all(isinstance(h, bytes) for h in hashes.values())

            assert self.backend.cluster_id_to_hash(ids=[-6]) == {-6: None}

    def test_get_source(self):
        """Test querying data from the database."""
        with self.scenario(self.backend, "index") as dag:
            crn_testkit = dag.sources.get("crn")

            crn_retrieved = self.backend.get_source(crn_testkit.source.address)
            # Equality between the two is False because one lacks the Engine
            assert crn_testkit.source.model_dump() == crn_retrieved.model_dump()

            with pytest.raises(MatchboxSourceNotFoundError):
                self.backend.get_source(
                    SourceAddress(
                        full_name="foo", warehouse_hash=bytes("bar".encode("ascii"))
                    )
                )

    def test_get_resolution_graph(self):
        """Test getting the resolution graph."""
        graph = self.backend.get_resolution_graph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert isinstance(graph, ResolutionGraph)

        with self.scenario(self.backend, "link"):
            graph = self.backend.get_resolution_graph()
            # Nodes: 3 datasets, 3 dedupers, and 3 linkers
            # Edges: 1 per deduper, 2 per linker
            assert len(graph.nodes) == 9
            assert len(graph.edges) == 9

    def test_get_model(self):
        """Test getting a model from the database."""
        with self.scenario(self.backend, "dedupe"):
            model = self.backend.get_model(model="naive_test.crn")
            assert isinstance(model, ModelMetadata)

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_model(model="nonexistant")

    def test_delete_model(self):
        """
        Tests the deletion of:

        * The model from the model table
        * The creates edges the model made
        * Any models that depended on this model, and their creates edges
        * Any probability values associated with the model
        * All of the above for all parent models. As every model is defined by
            its parents, deleting a model means cascading deletion to all descendants
        """
        with self.scenario(self.backend, "link") as dag:
            # Expect it to delete itself, its probabilities,
            # its parents, and their probabilities
            deduper_to_delete = "naive_test.crn"
            total_models = len(dag.models)

            models_pre_delete = self.backend.models.count()
            cluster_count_pre_delete = self.backend.clusters.count()
            cluster_assoc_count_pre_delete = self.backend.creates.count()
            proposed_merge_probs_pre_delete = self.backend.proposes.count()
            actual_merges_pre_delete = self.backend.merges.count()

            assert models_pre_delete == total_models
            assert cluster_count_pre_delete > 0
            assert cluster_assoc_count_pre_delete > 0
            assert proposed_merge_probs_pre_delete > 0
            assert actual_merges_pre_delete > 0

            # Perform deletion
            self.backend.delete_model(deduper_to_delete, certain=True)

            models_post_delete = self.backend.models.count()
            cluster_count_post_delete = self.backend.clusters.count()
            cluster_assoc_count_post_delete = self.backend.creates.count()
            proposed_merge_probs_post_delete = self.backend.proposes.count()
            actual_merges_post_delete = self.backend.merges.count()

            # Deletes deduper and parent linkers: 4 models gone
            assert models_post_delete == models_pre_delete - 4

            # Cluster, dedupe and link count unaffected
            assert cluster_count_post_delete == cluster_count_pre_delete
            assert actual_merges_post_delete == actual_merges_pre_delete

            # But count of propose and create edges has dropped
            assert cluster_assoc_count_post_delete < cluster_assoc_count_pre_delete
            assert proposed_merge_probs_post_delete < proposed_merge_probs_pre_delete

    def test_insert_model(self):
        """Test that models can be inserted."""
        with self.scenario(self.backend, "index") as dag:
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            # Test deduper insertion
            models_count = self.backend.models.count()

            self.backend.insert_model(
                model=ModelMetadata(
                    name="dedupe_1",
                    description="Test deduper 1",
                    type=ModelType.DEDUPER,
                    left_resolution=crn_testkit.source.alias,
                )
            )
            self.backend.insert_model(
                model=ModelMetadata(
                    name="dedupe_2",
                    description="Test deduper 2",
                    type=ModelType.DEDUPER,
                    left_resolution=duns_testkit.source.alias,
                )
            )

            assert self.backend.models.count() == models_count + 2

            # Test linker insertion
            self.backend.insert_model(
                model=ModelMetadata(
                    name="link_1",
                    description="Test linker 1",
                    type=ModelType.LINKER,
                    left_resolution="dedupe_1",
                    right_resolution="dedupe_2",
                )
            )

            assert self.backend.models.count() == models_count + 3

            # Test model upsert
            self.backend.insert_model(
                model=ModelMetadata(
                    name="link_1",
                    description="Test upsert",
                    type=ModelType.LINKER,
                    left_resolution="dedupe_1",
                    right_resolution="dedupe_2",
                )
            )

            assert self.backend.models.count() == models_count + 3

    def test_model_results(self):
        """Test that a model's results data can be set and retrieved."""
        with self.scenario(self.backend, "dedupe"):
            # Retrieve
            pre_results = self.backend.get_model_results(model="naive_test.crn")

            assert isinstance(pre_results, pa.Table)
            assert len(pre_results) > 0

            self.backend.validate_ids(ids=pre_results["id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["left_id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["right_id"].to_pylist())

            # Set
            target_row = pre_results.to_pylist()[0]
            target_id = target_row["id"]
            target_left_id = target_row["left_id"]
            target_right_id = target_row["right_id"]

            matches_id_mask = pc.not_equal(pre_results["id"], target_id)
            matches_left_mask = pc.not_equal(pre_results["left_id"], target_left_id)
            matches_right_mask = pc.not_equal(pre_results["right_id"], target_right_id)

            combined_mask = pc.and_(
                pc.and_(matches_id_mask, matches_left_mask), matches_right_mask
            )
            df_probabilities_truncated = pre_results.filter(combined_mask)

            results = df_probabilities_truncated.select(
                ["left_id", "right_id", "probability"]
            )

            self.backend.set_model_results(model="naive_test.crn", results=results)

            # Retrieve again
            post_results = self.backend.get_model_results(model="naive_test.crn")

            # Check difference
            assert len(pre_results) != len(post_results)

    def test_model_truth(self):
        """Test that a model's truth can be set and retrieved."""
        with self.scenario(self.backend, "dedupe"):
            # Retrieve
            pre_truth = self.backend.get_model_truth(model="naive_test.crn")

            # Set
            self.backend.set_model_truth(model="naive_test.crn", truth=0.5)

            # Retrieve again
            post_truth = self.backend.get_model_truth(model="naive_test.crn")

            # Check difference
            assert pre_truth != post_truth

    def test_model_ancestors(self):
        """Test that a model's ancestors can be retrieved."""
        with self.scenario(self.backend, "link"):
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            linker_ancestors = self.backend.get_model_ancestors(model=linker_name)

            assert [
                isinstance(ancestor, ModelAncestor) for ancestor in linker_ancestors
            ]

            # Not all ancestors have truth values, but one must
            truth_found = False
            for ancestor in linker_ancestors:
                if isinstance(ancestor.truth, float):
                    truth_found = True

            assert truth_found

    def test_model_ancestors_cache(self):
        """Test that a model's ancestors cache can be set and retrieved."""
        with self.scenario(self.backend, "link"):
            linker_name = "deterministic_naive_test.crn_naive_test.duns"

            # Retrieve
            pre_ancestors_cache = self.backend.get_model_ancestors_cache(
                model=linker_name
            )

            # Set
            updated_ancestors_cache = [
                ModelAncestor(name=ancestor.name, truth=0.5)
                for ancestor in pre_ancestors_cache
            ]
            self.backend.set_model_ancestors_cache(
                model=linker_name, ancestors_cache=updated_ancestors_cache
            )

            # Retrieve again
            post_ancestors_cache = self.backend.get_model_ancestors_cache(
                model=linker_name
            )

            # Check difference
            assert pre_ancestors_cache != post_ancestors_cache
            assert post_ancestors_cache == updated_ancestors_cache

    def test_index(self):
        """Test that indexing data works."""
        assert self.backend.data.count() == 0

        with self.scenario(self.backend, "index") as dag:
            assert self.backend.data.count() == (
                len(dag.sources["crn"].entities)
                + len(dag.sources["cdms"].entities)
                + len(dag.sources["duns"].entities)
            )

    def test_index_new_source(self):
        """Test that indexing identical works."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")

            assert self.backend.clusters.count() == 0

            self.backend.index(crn_testkit.source, crn_testkit.data_hashes)

            crn_retrieved = self.backend.get_source(crn_testkit.source.address)

            # Equality between the two is False because one lacks the Engine
            assert crn_testkit.source.model_dump() == crn_retrieved.model_dump()
            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            # I can add it again with no consequences
            self.backend.index(crn_testkit.source, crn_testkit.data_hashes)
            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 1

    def test_index_duplicate_clusters(self):
        """Test that indexing new data with duplicate hashes works."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")

            data_hashes_halved = crn_testkit.data_hashes.slice(
                0, crn_testkit.data_hashes.num_rows // 2
            )

            assert self.backend.data.count() == 0
            self.backend.index(crn_testkit.source, data_hashes_halved)
            assert self.backend.data.count() == data_hashes_halved.num_rows
            self.backend.index(crn_testkit.source, crn_testkit.data_hashes)
            assert self.backend.data.count() == crn_testkit.data_hashes.num_rows
            assert self.backend.source_resolutions.count() == 1

    def test_index_same_resolution(self):
        """Test that indexing same-name sources in different warehouses works."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")

            crn_source_1 = crn_testkit.source
            crn_source_1.address.warehouse_hash = b"bar1"
            crn_source_2 = crn_testkit.source.model_copy(deep=True)
            crn_source_2.address.warehouse_hash = b"bar2"

            self.backend.index(crn_source_1, crn_testkit.data_hashes)
            self.backend.index(crn_source_2, crn_testkit.data_hashes)

            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 1

    def test_index_different_resolution_same_hashes(self):
        """Test that indexing data with the same hashes but different sources works."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")
            duns_testkit: SourceTestkit = dag.sources.get("duns")

            self.backend.index(crn_testkit.source, crn_testkit.data_hashes)
            # Different source, same data
            # TODO: this will now error, and it shouldn't
            with pytest.raises(NotImplementedError):
                self.backend.index(duns_testkit.source, crn_testkit.data_hashes)

    def test_query_only_source(self):
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "index") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn_sample = self.backend.query(
                source_address=crn_testkit.source.address,
                limit=10,
            )

            assert isinstance(df_crn_sample, pa.Table)
            assert df_crn_sample.num_rows == 10

            df_crn_full = self.backend.query(source_address=crn_testkit.source.address)

            assert df_crn_full.num_rows == crn_testkit.query.num_rows
            assert set(df_crn_full.column_names) == {"id", "source_pk"}

    def test_query_with_dedupe_model(self):
        """Test querying data from a deduplication point of truth."""
        with self.scenario(self.backend, "dedupe") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn = self.backend.query(
                source_address=crn_testkit.source.address,
                resolution_name="naive_test.crn",
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.query.num_rows
            assert set(df_crn.column_names) == {"id", "source_pk"}

            sources_dict = dag.get_sources_for_model("naive_test.crn")
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            assert pc.count_distinct(df_crn["id"]).as_py() == len(
                linked.true_entity_subset("crn")
            )

    def test_query_with_link_model(self):
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            df_crn = self.backend.query(
                source_address=crn_testkit.source.address,
                resolution_name=linker_name,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.query.num_rows
            assert set(df_crn.column_names) == {"id", "source_pk"}

            df_duns = self.backend.query(
                source_address=duns_testkit.source.address,
                resolution_name=linker_name,
            )

            assert isinstance(df_duns, pa.Table)
            assert df_duns.num_rows == duns_testkit.query.num_rows
            assert set(df_duns.column_names) == {"id", "source_pk"}

            sources_dict = dag.get_sources_for_model(linker_name)
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            all_ids = pa.concat_arrays(
                [df_crn["id"].combine_chunks(), df_duns["id"].combine_chunks()]
            )

            assert pc.count_distinct(all_ids).as_py() == len(
                linked.true_entity_subset("crn", "duns")
            )

    def test_match_one_to_many(self):
        """Test that matching data works when the target has many IDs."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            sources_dict = dag.get_sources_for_model(linker_name)
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            # A random one:many entity
            source_entity: SourceEntity = linked.find_entities(
                min_appearances={"crn": 2, "duns": 1},
                max_appearances={"duns": 1},
            )[0]

            res = self.backend.match(
                source_pk=next(iter(source_entity.source_pks["duns"])),
                source=duns_testkit.source.address,
                targets=[crn_testkit.source.address],
                resolution_name=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == duns_testkit.source.address
            assert res[0].target == crn_testkit.source.address
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.source_pks["duns"]
            assert res[0].target_id == source_entity.source_pks["crn"]

    def test_match_many_to_one(self):
        """Test that matching data works when the source has more possible IDs."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            sources_dict = dag.get_sources_for_model(linker_name)
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            # A random many:one entity
            source_entity: SourceEntity = linked.find_entities(
                min_appearances={"crn": 2, "duns": 1},
                max_appearances={"duns": 1},
            )[0]

            res = self.backend.match(
                source_pk=next(iter(source_entity.source_pks["crn"])),
                source=crn_testkit.source.address,
                targets=[duns_testkit.source.address],
                resolution_name=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.address
            assert res[0].target == duns_testkit.source.address
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.source_pks["crn"]
            assert res[0].target_id == source_entity.source_pks["duns"]

    def test_match_one_to_none(self):
        """Test that matching data works when the target has no IDs."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            sources_dict = dag.get_sources_for_model(linker_name)
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            # A random one:none entity
            source_entity: SourceEntity = linked.find_entities(
                min_appearances={"crn": 1},
                max_appearances={"duns": 0},
            )[0]

            res = self.backend.match(
                source_pk=next(iter(source_entity.source_pks["crn"])),
                source=crn_testkit.source.address,
                targets=[duns_testkit.source.address],
                resolution_name=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.address
            assert res[0].target == duns_testkit.source.address
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.source_pks["crn"]
            assert res[0].target_id == source_entity.source_pks.get("duns", set())

    def test_match_none_to_none(self):
        """Test that matching data works when the supplied key doesn't exist."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            # Use a non-existent source primary key
            non_existent_pk = "foo"

            res = self.backend.match(
                source_pk=non_existent_pk,
                source=crn_testkit.source.address,
                targets=[duns_testkit.source.address],
                resolution_name=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.address
            assert res[0].target == duns_testkit.source.address
            assert res[0].cluster is None
            assert res[0].source_id == set()
            assert res[0].target_id == set()

    def test_clear(self):
        """Test clearing the database."""
        with self.scenario(self.backend, "dedupe"):
            assert self.backend.datasets.count() > 0
            assert self.backend.data.count() > 0
            assert self.backend.models.count() > 0
            assert self.backend.clusters.count() > 0
            assert self.backend.creates.count() > 0
            assert self.backend.merges.count() > 0
            assert self.backend.proposes.count() > 0

            self.backend.clear(certain=True)

            assert self.backend.datasets.count() == 0
            assert self.backend.data.count() == 0
            assert self.backend.models.count() == 0
            assert self.backend.clusters.count() == 0
            assert self.backend.creates.count() == 0
            assert self.backend.merges.count() == 0
            assert self.backend.proposes.count() == 0

    def test_dump_and_restore(self):
        """Test that dumping and restoring the database works."""
        with self.scenario(self.backend, "link") as dag:
            crn_testkit = dag.sources.get("crn")

            # Verify we have data
            pre_dump_datasets_count = self.backend.datasets.count()
            pre_dump_models_count = self.backend.models.count()
            pre_dump_data_count = self.backend.data.count()
            pre_dump_clusters_count = self.backend.clusters.count()
            pre_dump_merges_count = self.backend.merges.count()
            pre_dump_creates_count = self.backend.creates.count()
            pre_dump_proposes_count = self.backend.proposes.count()

            # All these should be greater than zero after setup
            assert pre_dump_datasets_count > 0
            assert pre_dump_models_count > 0
            assert pre_dump_data_count > 0
            assert pre_dump_clusters_count > 0
            assert pre_dump_merges_count > 0
            assert pre_dump_creates_count > 0
            assert pre_dump_proposes_count > 0

            # Get some specific IDs to verify they're restored properly
            df_crn_before = self.backend.query(
                source_address=crn_testkit.source.address,
                resolution_name="naive_test.crn",
            )
            sample_ids_before = df_crn_before["id"].to_pylist()[:5]  # Take first 5 IDs

            # Dump the database
            snapshot = self.backend.dump()

            # Clear the database
            self.backend.clear(certain=True)

            # Verify database is empty
            assert self.backend.datasets.count() == 0
            assert self.backend.models.count() == 0
            assert self.backend.data.count() == 0
            assert self.backend.clusters.count() == 0
            assert self.backend.merges.count() == 0
            assert self.backend.creates.count() == 0
            assert self.backend.proposes.count() == 0

            # Restore from snapshot
            self.backend.restore(snapshot)

            # Verify counts match pre-dump state
            assert self.backend.datasets.count() == pre_dump_datasets_count
            assert self.backend.models.count() == pre_dump_models_count
            assert self.backend.data.count() == pre_dump_data_count
            assert self.backend.clusters.count() == pre_dump_clusters_count
            assert self.backend.merges.count() == pre_dump_merges_count
            assert self.backend.creates.count() == pre_dump_creates_count
            assert self.backend.proposes.count() == pre_dump_proposes_count

            # Verify specific data was restored correctly
            df_crn_after = self.backend.query(
                source_address=crn_testkit.source.address,
                resolution_name="naive_test.crn",
            )
            sample_ids_after = df_crn_after["id"].to_pylist()[:5]  # Take first 5 IDs

            # The same IDs should be present after restoration
            assert set(sample_ids_before) == set(sample_ids_after)

            # Test the clear parameter of restore
            self.backend.restore(snapshot, clear=True)

            # Verify counts still match
            assert self.backend.datasets.count() == pre_dump_datasets_count
            assert self.backend.models.count() == pre_dump_models_count
            assert self.backend.data.count() == pre_dump_data_count
