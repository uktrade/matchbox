from functools import partial

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from sqlalchemy import Engine

from matchbox.common.arrow import SCHEMA_MB_IDS
from matchbox.common.dtos import ModelAncestor, ModelConfig, ModelType
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.entities import (
    SourceEntity,
    diff_results,
    query_to_cluster_entities,
)
from matchbox.common.factories.sources import SourceTestkit
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import HASH_FUNC
from matchbox.common.sources import Match, RelationalDBLocation
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
            assert isinstance(self.backend.sources.list_all(), list)
            assert isinstance(self.backend.sources.count(), int)
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
                source=crn_testkit.source_config.name,
                resolution="naive_test.crn",
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
                source=crn_testkit.source_config.name,
                resolution="naive_test.crn",
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
                source=crn_testkit.source_config.name,
                resolution="naive_test.crn",
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

            crn_retrieved = self.backend.get_source_config(
                crn_testkit.source_config.name
            )
            # Equality between the two is False because one lacks the Engine
            assert crn_testkit.source_config.model_dump() == crn_retrieved.model_dump()

            with pytest.raises(MatchboxSourceNotFoundError):
                self.backend.get_source_config(name="foo")

    def test_get_resolution_sources(self):
        """Test retrieving sources available to a resolution."""
        with self.scenario(self.backend, "link") as dag:
            crn, duns = dag.sources["crn"], dag.sources["duns"]
            dedupe_sources = self.backend.get_resolution_source_configs(
                name="naive_test.crn"
            )
            link_sources = self.backend.get_resolution_source_configs(
                name="deterministic_naive_test.crn_naive_test.duns"
            )

            assert {s.name for s in dedupe_sources} == {crn.source_config.name}

            assert {s.name for s in link_sources} == {
                crn.source_config.name,
                duns.source_config.name,
            }

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_resolution_source_configs(name="nonexistent")

    def test_get_resolution_graph(self):
        """Test getting the resolution graph."""
        graph = self.backend.get_resolution_graph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert isinstance(graph, ResolutionGraph)

        with self.scenario(self.backend, "link"):
            graph = self.backend.get_resolution_graph()
            # Nodes: 3 sources, 3 dedupers, and 3 linkers
            # Edges: 1 per deduper, 2 per linker
            assert len(graph.nodes) == 9
            assert len(graph.edges) == 9

    def test_get_model(self):
        """Test getting a model from the database."""
        with self.scenario(self.backend, "dedupe"):
            model = self.backend.get_model(name="naive_test.crn")
            assert isinstance(model, ModelConfig)

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_model(name="nonexistent")

    def test_delete_resolution(self):
        """
        Tests the deletion of:

        * The resolution from the resolutions table
        * The source configuration attached to the resolution
        * Any models that depended on this model (descendants)
        * All probabilities for all descendants
        """
        with self.scenario(self.backend, "link") as dag:
            resolution_to_delete = dag.sources["crn"].source_config.name
            total_sources = len(dag.sources)
            total_models = len(dag.models)

            source_configs_pre_delete = self.backend.sources.count()
            sources_pre_delete = self.backend.source_resolutions.count()
            models_pre_delete = self.backend.models.count()
            cluster_count_pre_delete = self.backend.clusters.count()
            cluster_assoc_count_pre_delete = self.backend.creates.count()
            proposed_merge_probs_pre_delete = self.backend.proposes.count()

            assert sources_pre_delete == total_sources
            assert models_pre_delete == total_models
            assert cluster_count_pre_delete > 0
            assert cluster_assoc_count_pre_delete > 0
            assert proposed_merge_probs_pre_delete > 0

            # Perform deletion
            self.backend.delete_resolution(resolution_to_delete, certain=True)

            source_configs_post_delete = self.backend.sources.count()
            sources_post_delete = self.backend.source_resolutions.count()
            models_post_delete = self.backend.models.count()
            cluster_count_post_delete = self.backend.clusters.count()
            cluster_assoc_count_post_delete = self.backend.creates.count()
            proposed_merge_probs_post_delete = self.backend.proposes.count()

            # 1 source, 1 index, (1 deduper + 3 linkers) = 4 models are gone
            assert source_configs_post_delete == source_configs_pre_delete - 1
            assert sources_post_delete == sources_pre_delete - 1
            assert models_post_delete == models_pre_delete - 4

            # We've lost some composite clusters
            assert cluster_count_post_delete < cluster_count_pre_delete

            # Count of propose and create edges has dropped
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
                model_config=ModelConfig(
                    name="dedupe_1",
                    description="Test deduper 1",
                    type=ModelType.DEDUPER,
                    left_resolution=crn_testkit.source_config.name,
                )
            )
            self.backend.insert_model(
                model_config=ModelConfig(
                    name="dedupe_2",
                    description="Test deduper 2",
                    type=ModelType.DEDUPER,
                    left_resolution=duns_testkit.source_config.name,
                )
            )

            assert self.backend.models.count() == models_count + 2

            # Test linker insertion
            self.backend.insert_model(
                model_config=ModelConfig(
                    name="link_1",
                    description="Test linker 1",
                    type=ModelType.LINKER,
                    left_resolution="dedupe_1",
                    right_resolution="dedupe_2",
                )
            )

            assert self.backend.models.count() == models_count + 3

            # Test can't insert duplicate
            with pytest.raises(MatchboxResolutionAlreadyExists):
                self.backend.insert_model(
                    model_config=ModelConfig(
                        name="link_1",
                        description="Test upsert",
                        type=ModelType.LINKER,
                        left_resolution="dedupe_1",
                        right_resolution="dedupe_2",
                    )
                )

            assert self.backend.models.count() == models_count + 3

    def test_model_results_basic(self):
        """Test that a model's results data can be set and retrieved."""
        with self.scenario(self.backend, "dedupe") as dag:
            # Query returns the same results as the testkit, showing
            # that processing was performed accurately
            res = self.backend.query(
                source=dag.sources["crn"].source_config.name,
                resolution="naive_test.crn",
            )
            res_clusters = query_to_cluster_entities(
                query=res,
                keys={dag.sources["crn"].name: "key"},
            )

            identical, report = diff_results(
                expected=dag.models["naive_test.crn"].entities,
                actual=res_clusters,
            )

            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_results(name="naive_test.crn")

            assert isinstance(pre_results, pa.Table)
            assert len(pre_results) > 0

            # Validate IDs
            self.backend.validate_ids(ids=pre_results["left_id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["right_id"].to_pylist())

            # Wrangle in polars
            pre_results_pl = pl.from_arrow(pre_results)

            # Remove a single row from the results
            target_row = pre_results_pl.row(0, named=True)

            results_truncated = pre_results_pl.filter(
                ~(
                    (pl.col("left_id") == target_row["left_id"])
                    & (pl.col("right_id") == target_row["right_id"])
                )
            )

            # Set new results
            self.backend.set_model_results(
                name="naive_test.crn", results=results_truncated.to_arrow()
            )

            # Retrieve again
            post_results = self.backend.get_model_results(name="naive_test.crn")

            # Check difference
            assert len(pre_results) != len(post_results)
            assert len(post_results) == len(pre_results) - 1

    def test_model_results_probabilistic(self):
        """Test that a probabilistic model's results data can be set and retrieved."""
        with self.scenario(self.backend, "probabilistic_dedupe") as dag:
            # Query returns the same results as the testkit, showing
            # that processing was performed accurately
            res = self.backend.query(
                source=dag.sources["crn"].source_config.name,
                resolution="probabilistic_test.crn",
            )
            res_clusters = query_to_cluster_entities(
                query=res,
                keys={dag.sources["crn"].name: "key"},
            )

            identical, report = diff_results(
                expected=dag.models["probabilistic_test.crn"].entities,
                actual=res_clusters,
            )
            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_results(name="probabilistic_test.crn")

            assert isinstance(pre_results, pa.Table)
            assert len(pre_results) > 0

            # Validate IDs
            self.backend.validate_ids(ids=pre_results["left_id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["right_id"].to_pylist())

            # Wrangle in polars
            pre_results_pl = pl.from_arrow(pre_results)

            # Remove a single row from the results
            target_row = pre_results_pl.row(0, named=True)

            results_truncated = pre_results_pl.filter(
                ~(
                    (pl.col("left_id") == target_row["left_id"])
                    & (pl.col("right_id") == target_row["right_id"])
                )
            )

            # Set new results
            self.backend.set_model_results(
                name="probabilistic_test.crn", results=results_truncated.to_arrow()
            )

            # Retrieve again
            post_results = self.backend.get_model_results(name="probabilistic_test.crn")

            # Check difference
            assert len(pre_results) != len(post_results)
            assert len(post_results) == len(pre_results) - 1

    def test_model_results_shared_clusters(self):
        """Test that model results data can be inserted when clusters are shared."""
        with self.scenario(self.backend, "convergent") as dag:
            for model_testkit in dag.models.values():
                self.backend.insert_model(model_config=model_testkit.model.model_config)
                self.backend.set_model_results(
                    name=model_testkit.name, results=model_testkit.probabilities
                )

    def test_model_truth(self):
        """Test that a model's truth can be set and retrieved."""
        with self.scenario(self.backend, "dedupe"):
            # Retrieve
            pre_truth = self.backend.get_model_truth(name="naive_test.crn")

            # Set
            self.backend.set_model_truth(name="naive_test.crn", truth=75)

            # Retrieve again
            post_truth = self.backend.get_model_truth(name="naive_test.crn")

            # Check difference
            assert pre_truth != post_truth

    def test_model_ancestors(self):
        """Test that a model's ancestors can be retrieved."""
        with self.scenario(self.backend, "link"):
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            linker_ancestors = self.backend.get_model_ancestors(name=linker_name)

            assert [
                isinstance(ancestor, ModelAncestor) for ancestor in linker_ancestors
            ]

            # Not all ancestors have truth values, but one must
            truth_found = False
            for ancestor in linker_ancestors:
                if isinstance(ancestor.truth, int):
                    truth_found = True

            assert truth_found

    def test_model_ancestors_cache(self):
        """Test that a model's ancestors cache can be set and retrieved."""
        with self.scenario(self.backend, "link"):
            linker_name = "deterministic_naive_test.crn_naive_test.duns"

            # Retrieve
            pre_ancestors_cache = self.backend.get_model_ancestors_cache(
                name=linker_name
            )

            # Set
            updated_ancestors_cache = [
                ModelAncestor(name=ancestor.name, truth=90)
                for ancestor in pre_ancestors_cache
            ]
            self.backend.set_model_ancestors_cache(
                name=linker_name, ancestors_cache=updated_ancestors_cache
            )

            # Retrieve again
            post_ancestors_cache = self.backend.get_model_ancestors_cache(
                name=linker_name
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

            self.backend.index(crn_testkit.source_config, crn_testkit.data_hashes)

            crn_retrieved = self.backend.get_source_config(
                crn_testkit.source_config.name
            )

            # Equality between the two is False because one lacks the Engine
            assert crn_testkit.source_config.model_dump() == crn_retrieved.model_dump()
            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            # I can add it again with no consequences
            self.backend.index(crn_testkit.source_config, crn_testkit.data_hashes)
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
            self.backend.index(crn_testkit.source_config, data_hashes_halved)
            assert self.backend.data.count() == data_hashes_halved.num_rows
            self.backend.index(crn_testkit.source_config, crn_testkit.data_hashes)
            assert self.backend.data.count() == crn_testkit.data_hashes.num_rows
            assert self.backend.source_resolutions.count() == 1

    def test_index_same_resolution(self):
        """Test that indexing same-name sources in different warehouses works."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")

            crn_source_1 = crn_testkit.source_config.model_copy(
                update={"location": RelationalDBLocation(uri="postgres://")}
            )
            crn_source_2 = crn_testkit.source_config.model_copy(
                deep=True, update={"location": RelationalDBLocation(uri="mongodb://")}
            )

            self.backend.index(crn_source_1, crn_testkit.data_hashes)
            self.backend.index(crn_source_2, crn_testkit.data_hashes)

            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 1

    def test_index_different_resolution_same_hashes(self):
        """Test that indexing data with the same hashes but different sources works."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")
            duns_testkit: SourceTestkit = dag.sources.get("duns")

            self.backend.index(crn_testkit.source_config, crn_testkit.data_hashes)
            # Different source, same data
            self.backend.index(duns_testkit.source_config, crn_testkit.data_hashes)
            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 2

    def test_query_only_source(self):
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "index") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn_sample = self.backend.query(
                source=crn_testkit.source_config.name,
                limit=10,
            )

            assert isinstance(df_crn_sample, pa.Table)
            assert df_crn_sample.num_rows == 10

            df_crn_full = self.backend.query(source=crn_testkit.source_config.name)

            assert df_crn_full.num_rows == crn_testkit.query.num_rows
            assert set(df_crn_full.column_names) == set(SCHEMA_MB_IDS.names)

    def test_query_with_dedupe_model(self):
        """Test querying data from a deduplication point of truth."""
        with self.scenario(self.backend, "dedupe") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn = self.backend.query(
                source=crn_testkit.source_config.name,
                resolution="naive_test.crn",
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.query.num_rows
            assert set(df_crn.column_names) == set(SCHEMA_MB_IDS.names)

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
                source=crn_testkit.source_config.name,
                resolution=linker_name,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.query.num_rows
            assert set(df_crn.column_names) == set(SCHEMA_MB_IDS.names)

            df_duns = self.backend.query(
                source=duns_testkit.source_config.name,
                resolution=linker_name,
            )

            assert isinstance(df_duns, pa.Table)
            assert df_duns.num_rows == duns_testkit.query.num_rows
            assert set(df_duns.column_names) == set(SCHEMA_MB_IDS.names)

            sources_dict = dag.get_sources_for_model(linker_name)
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            all_ids = pa.concat_arrays(
                [df_crn["id"].combine_chunks(), df_duns["id"].combine_chunks()]
            )

            assert pc.count_distinct(all_ids).as_py() == len(
                linked.true_entity_subset("crn", "duns")
            )

    def test_threshold_query_with_link_model(self):
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "probabilistic_naive_test.crn_naive_test.cdms"
            crn_testkit = dag.sources.get("crn")
            cdms_testkit = dag.sources.get("cdms")

            df_crn = self.backend.query(
                source=crn_testkit.source_config.name,
                resolution=linker_name,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.query.num_rows
            assert set(df_crn.column_names) == set(SCHEMA_MB_IDS.names)

            df_cdms = self.backend.query(
                source=cdms_testkit.source_config.name,
                resolution=linker_name,
            )

            assert isinstance(df_cdms, pa.Table)
            assert df_cdms.num_rows == cdms_testkit.query.num_rows
            assert set(df_cdms.column_names) == set(SCHEMA_MB_IDS.names)

            sources_dict = dag.get_sources_for_model(linker_name)
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            # Test query with threshold
            df_crn_threshold = self.backend.query(
                source=crn_testkit.source_config.name,
                resolution=linker_name,
                threshold=100,
            )
            df_cdms_threshold = self.backend.query(
                source=cdms_testkit.source_config.name,
                resolution=linker_name,
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
                key=next(iter(source_entity.keys["duns"])),
                source=duns_testkit.source_config.name,
                targets=[crn_testkit.source_config.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == duns_testkit.source_config.name
            assert res[0].target == crn_testkit.source_config.name
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["duns"]
            assert res[0].target_id == source_entity.keys["crn"]

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
                key=next(iter(source_entity.keys["crn"])),
                source=crn_testkit.source_config.name,
                targets=[duns_testkit.source_config.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source_config.name
            assert res[0].target == duns_testkit.source_config.name
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["crn"]
            assert res[0].target_id == source_entity.keys["duns"]

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
                key=next(iter(source_entity.keys["crn"])),
                source=crn_testkit.source_config.name,
                targets=[duns_testkit.source_config.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source_config.name
            assert res[0].target == duns_testkit.source_config.name
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["crn"]
            assert res[0].target_id == source_entity.keys.get("duns", set())

    def test_match_none_to_none(self):
        """Test that matching data works when the supplied key doesn't exist."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test.crn_naive_test.duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            # Use a non-existent source key
            non_existent_key = "foo"

            res = self.backend.match(
                key=non_existent_key,
                source=crn_testkit.source_config.name,
                targets=[duns_testkit.source_config.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source_config.name
            assert res[0].target == duns_testkit.source_config.name
            assert res[0].cluster is None
            assert res[0].source_id == set()
            assert res[0].target_id == set()

    def test_threshold_match_many_to_one(self):
        """Test that matching data works when the target has many IDs."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "probabilistic_naive_test.crn_naive_test.cdms"
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
                key=next(iter(source_entity.keys["crn"])),
                source=crn_testkit.source_config.name,
                targets=[duns_testkit.source_config.name],
                resolution=linker_name,
                threshold=100,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source_config.name
            assert res[0].target == duns_testkit.source_config.name
            assert res[0].source_id == source_entity.keys["crn"]
            # Match does not return true target ids when threshold
            # exceeds match probability
            assert len(res[0].target_id) < len(source_entity.keys["duns"])

    def test_clear(self):
        """Test deleting all rows in the database."""
        with self.scenario(self.backend, "dedupe"):
            assert self.backend.sources.count() > 0
            assert self.backend.data.count() > 0
            assert self.backend.models.count() > 0
            assert self.backend.clusters.count() > 0
            assert self.backend.creates.count() > 0
            assert self.backend.merges.count() > 0
            assert self.backend.proposes.count() > 0

            self.backend.clear(certain=True)

            assert self.backend.sources.count() == 0
            assert self.backend.data.count() == 0
            assert self.backend.models.count() == 0
            assert self.backend.clusters.count() == 0
            assert self.backend.creates.count() == 0
            assert self.backend.merges.count() == 0
            assert self.backend.proposes.count() == 0

    def test_clear_and_restore(self):
        """Test that clearing and restoring the database works."""
        with self.scenario(self.backend, "link") as dag:
            crn_testkit = dag.sources.get("crn")

            count_funcs = [
                self.backend.sources.count,
                self.backend.models.count,
                self.backend.data.count,
                self.backend.clusters.count,
                self.backend.merges.count,
                self.backend.creates.count,
                self.backend.proposes.count,
            ]

            def get_counts():
                return [f() for f in count_funcs]

            # Verify we have data
            pre_dump_counts = get_counts()
            assert all(count > 0 for count in pre_dump_counts)

            # Get some specific IDs to verify they're restored properly
            df_crn_before = self.backend.query(
                source=crn_testkit.source_config.name,
                resolution="naive_test.crn",
            )
            sample_ids_before = df_crn_before["id"].to_pylist()[:5]  # Take first 5 IDs

            # Dump the database
            snapshot = self.backend.dump()

        with self.scenario(self.backend, "bare"):
            # Verify counts match pre-dump state
            assert all(c == 0 for c in get_counts())

            # Restore from snapshot
            self.backend.restore(snapshot)

            # Verify counts match pre-dump state
            assert get_counts() == pre_dump_counts

            # Verify specific data was restored correctly
            df_crn_after = self.backend.query(
                source=crn_testkit.source_config.name,
                resolution="naive_test.crn",
            )
            sample_ids_after = df_crn_after["id"].to_pylist()[:5]  # Take first 5 IDs

            # The same IDs should be present after restoration
            assert set(sample_ids_before) == set(sample_ids_after)

            # Test that restoring also clears the database
            self.backend.restore(snapshot)

            # Verify counts still match
            assert get_counts() == pre_dump_counts
