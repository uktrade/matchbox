from functools import partial

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from polars.testing import assert_frame_equal
from sqlalchemy import Engine

from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES,
    SCHEMA_JUDGEMENTS,
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
)
from matchbox.common.dtos import (
    LocationConfig,
    LocationType,
    Match,
    ModelConfig,
    ModelType,
    QueryConfig,
    Resolution,
)
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxNoJudgements,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.entities import (
    SourceEntity,
    diff_results,
    query_to_cluster_entities,
)
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.common.factories.sources import SourceTestkit
from matchbox.common.graph import ResolutionGraph, ResolutionType
from matchbox.common.hash import HASH_FUNC
from matchbox.server.base import MatchboxDBAdapter

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
                source=crn_testkit.source.name,
                resolution="naive_test_crn",
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
                source=crn_testkit.source.name,
                resolution="naive_test_crn",
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
                source=crn_testkit.source.name,
                resolution="naive_test_crn",
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

            crn_retrieved = self.backend.get_resolution(
                crn_testkit.name, validate=ResolutionType.SOURCE
            )
            assert isinstance(crn_retrieved, Resolution)
            assert crn_testkit.source_config == crn_retrieved.config

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_resolution(name="foo", validate=ResolutionType.SOURCE)

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_resolution(
                    name=crn_testkit.name, validate=ResolutionType.MODEL
                )

    def test_get_resolution_sources(self):
        """Test retrieving sources available to a resolution."""
        with self.scenario(self.backend, "link") as dag:
            crn, duns = dag.sources["crn"], dag.sources["duns"]
            dedupe_sources = self.backend.get_leaf_source_resolutions(
                name="naive_test_crn"
            )
            link_sources = self.backend.get_leaf_source_resolutions(
                name="deterministic_naive_test_crn_naive_test_duns"
            )

            assert {s.name for s in dedupe_sources} == {crn.source.name}

            assert {s.name for s in link_sources} == {
                crn.source.name,
                duns.source.name,
            }

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_leaf_source_resolutions(name="nonexistent")

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
        with self.scenario(self.backend, "dedupe") as dag:
            model_testkit = dag.models.get("naive_test_crn")

            model_retrieved = self.backend.get_resolution(
                name=model_testkit.name, validate=ResolutionType.MODEL
            )

            assert isinstance(model_retrieved, Resolution)

            assert model_testkit.model.config == model_retrieved.config

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_resolution(
                    name="nonexistent", validate=ResolutionType.MODEL
                )

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_resolution(
                    name=model_testkit.name, validate=ResolutionType.SOURCE
                )

    def test_delete_resolution(self):
        """
        Tests the deletion of:

        * The resolution from the resolutions table
        * The source configuration attached to the resolution
        * Any models that depended on this model (descendants)
        * All probabilities for all descendants
        """
        with self.scenario(self.backend, "link") as dag:
            resolution_to_delete = dag.sources["crn"].source.name
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

            self.backend.insert_resolution(
                resolution=Resolution(
                    name="dedupe_1",
                    description="Test deduper 1",
                    resolution_type=ResolutionType.MODEL,
                    truth=100,
                    config=ModelConfig(
                        type=ModelType.DEDUPER,
                        model_class="NaiveDeduper",
                        model_settings="{}",
                        left_query=QueryConfig(
                            source_resolutions=[crn_testkit.source.name]
                        ),
                    ),
                )
            )
            self.backend.insert_resolution(
                resolution=Resolution(
                    name="dedupe_2",
                    description="Test deduper 2",
                    resolution_type=ResolutionType.MODEL,
                    truth=100,
                    config=ModelConfig(
                        type=ModelType.DEDUPER,
                        model_class="NaiveDeduper",
                        model_settings="{}",
                        left_query=QueryConfig(
                            source_resolutions=[duns_testkit.source.name]
                        ),
                    ),
                )
            )

            assert self.backend.models.count() == models_count + 2

            # Test linker insertion
            linker_resolution = Resolution(
                name="link_1",
                description="Test linker 1",
                resolution_type=ResolutionType.MODEL,
                truth=100,
                config=ModelConfig(
                    type=ModelType.LINKER,
                    model_class="NaiveDeduper",
                    model_settings="{}",
                    left_query=QueryConfig(
                        source_resolutions=["crn"], model_resolution="dedupe_1"
                    ),
                    right_query=QueryConfig(
                        source_resolutions=["duns"], model_resolution="dedupe_2"
                    ),
                ),
            )
            self.backend.insert_resolution(resolution=linker_resolution)

            assert self.backend.models.count() == models_count + 3

            # Test can't insert duplicate
            with pytest.raises(MatchboxResolutionAlreadyExists):
                self.backend.insert_resolution(linker_resolution)

            assert self.backend.models.count() == models_count + 3

    def test_model_results_basic(self):
        """Test that a model's results data can be set and retrieved."""
        with self.scenario(self.backend, "dedupe") as dag:
            # Query returns the same results as the testkit, showing
            # that processing was performed accurately
            res = self.backend.query(
                source=dag.sources["crn"].source.name,
                resolution="naive_test_crn",
            )
            res_clusters = query_to_cluster_entities(
                data=res,
                keys={dag.sources["crn"].name: "key"},
            )

            identical, report = diff_results(
                expected=dag.models["naive_test_crn"].entities,
                actual=res_clusters,
            )

            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_data(name="naive_test_crn")

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
            self.backend.insert_model_data(
                name="naive_test_crn", results=results_truncated.to_arrow()
            )

            # Retrieve again
            post_results = self.backend.get_model_data(name="naive_test_crn")

            # Check difference
            assert len(pre_results) != len(post_results)
            assert len(post_results) == len(pre_results) - 1

    def test_model_results_probabilistic(self):
        """Test that a probabilistic model's results data can be set and retrieved."""
        with self.scenario(self.backend, "probabilistic_dedupe") as dag:
            # Query returns the same results as the testkit, showing
            # that processing was performed accurately
            res = self.backend.query(
                source=dag.sources["crn"].source.name,
                resolution="probabilistic_test_crn",
            )
            res_clusters = query_to_cluster_entities(
                data=res,
                keys={dag.sources["crn"].name: "key"},
            )

            identical, report = diff_results(
                expected=dag.models["probabilistic_test_crn"].entities,
                actual=res_clusters,
            )
            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_data(name="probabilistic_test_crn")

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
            self.backend.insert_model_data(
                name="probabilistic_test_crn", results=results_truncated.to_arrow()
            )

            # Retrieve again
            post_results = self.backend.get_model_data(name="probabilistic_test_crn")

            # Check difference
            assert len(pre_results) != len(post_results)
            assert len(post_results) == len(pre_results) - 1

    def test_model_results_shared_clusters(self):
        """Test that model results data can be inserted when clusters are shared."""
        with self.scenario(self.backend, "convergent") as dag:
            for model_testkit in dag.models.values():
                self.backend.insert_resolution(
                    resolution=model_testkit.model.to_resolution()
                )
                self.backend.insert_model_data(
                    name=model_testkit.name, results=model_testkit.probabilities
                )

    def test_model_truth(self):
        """Test that a model's truth can be set and retrieved."""
        with self.scenario(self.backend, "dedupe"):
            # Retrieve
            pre_truth = self.backend.get_model_truth(name="naive_test_crn")

            # Set
            self.backend.set_model_truth(name="naive_test_crn", truth=75)

            # Retrieve again
            post_truth = self.backend.get_model_truth(name="naive_test_crn")

            # Check difference
            assert pre_truth != post_truth

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

            self.backend.insert_resolution(crn_testkit.source.to_resolution())
            self.backend.insert_source_data(
                crn_testkit.source.name, crn_testkit.data_hashes
            )

            crn_retrieved = self.backend.get_resolution(
                crn_testkit.source.name, validate=ResolutionType.SOURCE
            )

            # Equality between the two is False because one lacks the Engine
            assert (
                crn_testkit.source_config.model_dump()
                == crn_retrieved.config.model_dump()
            )
            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            # I can add it again with no consequences
            self.backend.insert_source_data(
                crn_testkit.source.name, crn_testkit.data_hashes
            )
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
            self.backend.insert_resolution(crn_testkit.source.to_resolution())
            self.backend.insert_source_data(crn_testkit.source.name, data_hashes_halved)
            assert self.backend.data.count() == data_hashes_halved.num_rows
            self.backend.insert_source_data(
                crn_testkit.source.name, crn_testkit.data_hashes
            )
            assert self.backend.data.count() == crn_testkit.data_hashes.num_rows
            assert self.backend.source_resolutions.count() == 1

    def test_index_same_resolution(self):
        """Test that indexing same-name sources errors."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")

            crn_source_config_1 = crn_testkit.source_config.model_copy(
                update={
                    "location": LocationConfig(type=LocationType.RDBMS, name="postgres")
                }
            )
            crn_source_config_2 = crn_testkit.source_config.model_copy(
                deep=True,
                update={
                    "location": LocationConfig(type=LocationType.RDBMS, name="mongodb")
                },
            )

            crn_resolution_1 = crn_testkit.source.to_resolution().model_copy(
                update={"config": crn_source_config_1}
            )
            crn_resolution_2 = crn_testkit.source.to_resolution().model_copy(
                update={"config": crn_source_config_2}
            )

            self.backend.insert_resolution(crn_resolution_1)
            self.backend.insert_source_data(
                crn_testkit.source.name, crn_testkit.data_hashes
            )

            with pytest.raises(MatchboxResolutionAlreadyExists):
                self.backend.insert_resolution(crn_resolution_2)

            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 1

    def test_index_different_resolution_same_hashes(self):
        """Test that indexing data with the same hashes but different sources works."""
        with self.scenario(self.backend, "bare") as dag:
            crn_testkit: SourceTestkit = dag.sources.get("crn")
            duns_testkit: SourceTestkit = dag.sources.get("duns")

            self.backend.insert_resolution(crn_testkit.source.to_resolution())
            self.backend.insert_source_data(
                crn_testkit.source.name, crn_testkit.data_hashes
            )
            # Different source, same data
            self.backend.insert_resolution(duns_testkit.source.to_resolution())
            self.backend.insert_source_data(
                duns_testkit.source.name, crn_testkit.data_hashes
            )
            assert self.backend.data.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 2

    def test_query_only_source(self):
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "index") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn_sample = self.backend.query(
                source=crn_testkit.source.name,
                limit=10,
            )

            assert isinstance(df_crn_sample, pa.Table)
            assert df_crn_sample.num_rows == 10

            df_crn_full = self.backend.query(source=crn_testkit.source.name)

            assert df_crn_full.num_rows == crn_testkit.data.num_rows
            assert df_crn_full.schema.equals(SCHEMA_QUERY)

    def test_query_return_leaf_ids(self):
        """Test querying data and additionally requesting leaf IDs."""
        with self.scenario(self.backend, "index") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn_full = self.backend.query(
                source=crn_testkit.source.name, return_leaf_id=True
            )

            assert df_crn_full.num_rows == crn_testkit.data.num_rows
            assert df_crn_full.schema.equals(SCHEMA_QUERY_WITH_LEAVES)

    def test_query_with_dedupe_model(self):
        """Test querying data from a deduplication point of truth."""
        with self.scenario(self.backend, "dedupe") as dag:
            crn_testkit = dag.sources.get("crn")

            df_crn = self.backend.query(
                source=crn_testkit.source.name,
                resolution="naive_test_crn",
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.data.num_rows
            assert df_crn.schema.equals(SCHEMA_QUERY)

            sources_dict = dag.get_sources_for_model("naive_test_crn")
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            assert pc.count_distinct(df_crn["id"]).as_py() == len(
                linked.true_entity_subset("crn")
            )

    def test_query_with_link_model(self):
        """Test querying data from a link point of truth."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test_crn_naive_test_duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            df_crn = self.backend.query(
                source=crn_testkit.source.name,
                resolution=linker_name,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.data.num_rows
            assert df_crn.schema.equals(SCHEMA_QUERY)

            df_duns = self.backend.query(
                source=duns_testkit.source.name,
                resolution=linker_name,
            )

            assert isinstance(df_duns, pa.Table)
            assert df_duns.num_rows == duns_testkit.data.num_rows
            assert df_duns.schema.equals(SCHEMA_QUERY)

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
            linker_name = "probabilistic_naive_test_crn_naive_test_cdms"
            crn_testkit = dag.sources.get("crn")
            cdms_testkit = dag.sources.get("cdms")

            df_crn = self.backend.query(
                source=crn_testkit.source.name,
                resolution=linker_name,
            )

            assert isinstance(df_crn, pa.Table)
            assert df_crn.num_rows == crn_testkit.data.num_rows
            assert df_crn.schema.equals(SCHEMA_QUERY)

            df_cdms = self.backend.query(
                source=cdms_testkit.source.name,
                resolution=linker_name,
            )

            assert isinstance(df_cdms, pa.Table)
            assert df_cdms.num_rows == cdms_testkit.data.num_rows
            assert df_cdms.schema.equals(SCHEMA_QUERY)

            sources_dict = dag.get_sources_for_model(linker_name)
            assert len(sources_dict) == 1
            linked = dag.linked[next(iter(sources_dict))]

            # Test query with threshold
            df_crn_threshold = self.backend.query(
                source=crn_testkit.source.name,
                resolution=linker_name,
                threshold=100,
            )
            df_cdms_threshold = self.backend.query(
                source=cdms_testkit.source.name,
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
            linker_name = "deterministic_naive_test_crn_naive_test_duns"
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
                source=duns_testkit.source.name,
                targets=[crn_testkit.source.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == duns_testkit.source.name
            assert res[0].target == crn_testkit.source.name
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["duns"]
            assert res[0].target_id == source_entity.keys["crn"]

    def test_match_many_to_one(self):
        """Test that matching data works when the source has more possible IDs."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test_crn_naive_test_duns"
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
                source=crn_testkit.source.name,
                targets=[duns_testkit.source.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.name
            assert res[0].target == duns_testkit.source.name
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["crn"]
            assert res[0].target_id == source_entity.keys["duns"]

    def test_match_one_to_none(self):
        """Test that matching data works when the target has no IDs."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test_crn_naive_test_duns"
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
                source=crn_testkit.source.name,
                targets=[duns_testkit.source.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.name
            assert res[0].target == duns_testkit.source.name
            assert res[0].cluster is not None
            assert res[0].source_id == source_entity.keys["crn"]
            assert res[0].target_id == source_entity.keys.get("duns", set())

    def test_match_none_to_none(self):
        """Test that matching data works when the supplied key doesn't exist."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "deterministic_naive_test_crn_naive_test_duns"
            crn_testkit = dag.sources.get("crn")
            duns_testkit = dag.sources.get("duns")

            # Use a non-existent source key
            non_existent_key = "foo"

            res = self.backend.match(
                key=non_existent_key,
                source=crn_testkit.source.name,
                targets=[duns_testkit.source.name],
                resolution=linker_name,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.name
            assert res[0].target == duns_testkit.source.name
            assert res[0].cluster is None
            assert res[0].source_id == set()
            assert res[0].target_id == set()

    def test_threshold_match_many_to_one(self):
        """Test that matching data works when the target has many IDs."""
        with self.scenario(self.backend, "link") as dag:
            linker_name = "probabilistic_naive_test_crn_naive_test_cdms"
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
                source=crn_testkit.source.name,
                targets=[duns_testkit.source.name],
                resolution=linker_name,
                threshold=100,
            )

            assert len(res) == 1
            assert isinstance(res[0], Match)
            assert res[0].source == crn_testkit.source.name
            assert res[0].target == duns_testkit.source.name
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
                source=crn_testkit.source.name,
                resolution="naive_test_crn",
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
                source=crn_testkit.source.name,
                resolution="naive_test_crn",
            )
            sample_ids_after = df_crn_after["id"].to_pylist()[:5]  # Take first 5 IDs

            # The same IDs should be present after restoration
            assert set(sample_ids_before) == set(sample_ids_after)

            # Test that restoring also clears the database
            self.backend.restore(snapshot)

            # Verify counts still match
            assert get_counts() == pre_dump_counts

    def test_login(self):
        """Can swap user name with user ID."""
        with self.scenario(self.backend, "bare"):
            alice_id = self.backend.login("alice")
            assert alice_id == self.backend.login("alice")
            assert alice_id != self.backend.login("bob")

    def test_insert_and_get_judgement(self):
        """Can insert and retrieve judgements."""
        with self.scenario(self.backend, "dedupe"):
            # To begin with, no judgements to retrieve
            judgements, expansion = self.backend.get_judgements()
            assert len(judgements) == len(expansion) == 0

            # Do some queries to find real source cluster IDs
            deduped_query = pl.from_arrow(
                self.backend.query(source="crn", resolution="naive_test_crn")
            )
            unique_ids = deduped_query["id"].unique()
            all_leaves = pl.from_arrow(self.backend.query(source="crn"))

            def get_leaf_ids(cluster_id: int) -> list[int]:
                return (
                    deduped_query.filter(pl.col("id") == cluster_id)
                    .join(all_leaves, on="key", suffix="_leaf")["id_leaf"]
                    .to_list()
                )

            alice_id = self.backend.login("alice")

            original_cluster_num = self.backend.clusters.count()

            # Can endorse the same cluster that is shown
            clust1_leaves = get_leaf_ids(unique_ids[0])
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_id=alice_id,
                    shown=unique_ids[0],
                    endorsed=[clust1_leaves],
                ),
            )
            # Can send redundant data
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_id=alice_id,
                    shown=unique_ids[0],
                    endorsed=[clust1_leaves],
                ),
            )
            assert self.backend.clusters.count() == original_cluster_num

            # Now split a cluster
            clust2_leaves = get_leaf_ids(unique_ids[1])
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_id=alice_id,
                    shown=unique_ids[1],
                    endorsed=[clust2_leaves[:1], clust2_leaves[1:]],
                ),
            )
            # Now, two new clusters should have been created
            assert self.backend.clusters.count() == original_cluster_num + 2

            # Let's check failures
            # First, confirm that the following leaves don't exist
            fake_leaves = [10000, 10001]
            with pytest.raises(MatchboxDataNotFound):
                self.backend.validate_ids(fake_leaves)
            # Now, let's test an exception is raised
            with pytest.raises(MatchboxDataNotFound):
                self.backend.insert_judgement(
                    judgement=Judgement(
                        user_id=alice_id, shown=unique_ids[0], endorsed=[fake_leaves]
                    ),
                )

            # Now, let's try to get the judgements back
            # Data gets back in the right shape
            judgements, expansion = self.backend.get_judgements()
            judgements.schema.equals(SCHEMA_JUDGEMENTS)
            expansion.schema.equals(SCHEMA_CLUSTER_EXPANSION)

            # Only one user ID was used
            assert judgements["user_id"].unique().to_pylist() == [alice_id]
            # The first shown cluster is repeated because we judged it twice
            # The second shown cluster is repeated because we split it (see above)
            assert sorted(judgements["shown"].to_pylist()) == sorted(
                [unique_ids[0], unique_ids[0], unique_ids[1], unique_ids[1]]
            )
            # On the other hand, the root-leaf mapping table has no duplicates
            assert len(expansion) == 4  # 2 shown clusters + 2 new endorsed clusters

            # Let's massage tables into a root-leaf dict for all endorsed clusters
            endorsed_dict = dict(
                pl.from_arrow(judgements)
                .join(pl.from_arrow(expansion), left_on="endorsed", right_on="root")
                .select(["endorsed", "leaves"])
                .rows()
            )

            # The root we know about has the leaves we expect
            assert endorsed_dict[unique_ids[0]] == clust1_leaves
            # Other than the root we know about, there are two new ones
            assert len(set(endorsed_dict.keys())) == 3
            # The other two sets of leaves are there too
            assert sorted(endorsed_dict.values()) == sorted(
                [clust1_leaves, clust2_leaves[:1], clust2_leaves[1:]]
            )

    def test_compare_models_fails(self):
        """Model comparison errors with no judgement data."""
        with (
            self.scenario(self.backend, "bare"),
            pytest.raises(MatchboxNoJudgements),
        ):
            self.backend.compare_models([])

    def test_compare_models(self):
        """Can compute precision and recall for list of models."""
        with self.scenario(self.backend, "alt_dedupe") as dag:
            user_id = self.backend.login("alice")

            model_names = list(dag.models.keys())

            root_leaves = (
                pl.from_arrow(
                    self.backend.sample_for_eval(
                        n=10, resolution=model_names[0], user_id=user_id
                    )
                )
                .select(["root", "leaf"])
                .unique()
                .group_by("root")
                .agg("leaf")
            )
            for row in root_leaves.rows(named=True):
                self.backend.insert_judgement(
                    judgement=Judgement(
                        user_id=user_id, shown=row["root"], endorsed=[row["leaf"]]
                    )
                )

            pr = self.backend.compare_models(list(dag.models.keys()))
            # Precision must be 1 for both as the second model is like the first
            # but more conservative
            assert pr[model_names[0]][0] == pr[model_names[1]][0] == 1
            # Recall must be 1 for the first model and lower for the second
            assert pr[model_names[0]][1] == 1
            assert pr[model_names[1]][1] < 1

    def test_sample_for_eval(self):
        """Can extract samples for a user and a resolution."""

        # Missing resolution raises error
        with (
            self.scenario(self.backend, "bare"),
            pytest.raises(MatchboxResolutionNotFoundError, match="naive_test_crn"),
        ):
            user_id = self.backend.login("alice")
            self.backend.sample_for_eval(
                n=10, resolution="naive_test_crn", user_id=user_id
            )

        with self.scenario(self.backend, "dedupe"):
            user_id = self.backend.login("alice")

            # Source clusters should not be returned
            # So if we sample from a source resolution, we get nothing
            user_id = self.backend.login("alice")
            samples_source = self.backend.sample_for_eval(
                n=10, resolution="crn", user_id=user_id
            )
            assert len(samples_source) == 0

            # We now look at more interesting cases
            # Query backend to form expectations
            resolution_clusters = pl.from_arrow(
                self.backend.query(source="crn", resolution="naive_test_crn")
            )
            source_clusters = pl.from_arrow(self.backend.query(source="crn"))
            # We can request more than available
            assert len(resolution_clusters["id"].unique()) < 99

            samples_99 = self.backend.sample_for_eval(
                n=99, resolution="naive_test_crn", user_id=user_id
            )

            assert samples_99.schema.equals(SCHEMA_EVAL_SAMPLES)

            # We can reconstruct the expected sample from resolution and source queries
            expected_sample = (
                resolution_clusters.join(source_clusters, on="key", suffix="_source")
                .rename({"id": "root", "id_source": "leaf"})
                .with_columns(pl.lit("crn").alias("source"))
            )

            assert_frame_equal(
                pl.from_arrow(samples_99),
                expected_sample,
                check_row_order=False,
                check_column_order=False,
                check_dtypes=False,
            )

            # We can request less than available
            assert len(resolution_clusters["id"].unique()) > 5
            samples_5 = self.backend.sample_for_eval(
                n=5, resolution="naive_test_crn", user_id=user_id
            )
            assert len(samples_5["root"].unique()) == 5

            # If user has recent judgements, exclude clusters
            first_cluster_id = resolution_clusters["id"][0]
            first_cluster = resolution_clusters.filter(pl.col("id") == first_cluster_id)
            first_cluster_leaves = first_cluster.join(
                source_clusters, on="key", suffix="_source"
            )["id_source"].to_list()

            self.backend.insert_judgement(
                judgement=Judgement(
                    user_id=user_id,
                    shown=first_cluster_id,
                    endorsed=[first_cluster_leaves],
                ),
            )

            samples_without_cluster = self.backend.sample_for_eval(
                n=99, resolution="naive_test_crn", user_id=user_id
            )
            # Compared to the first query, we should have one fewer cluster
            assert len(samples_99["root"].unique()) - 1 == len(
                samples_without_cluster["root"].unique()
            )
            # And that cluster is the one on which the judgement is based
            assert first_cluster_id in samples_99["root"].to_pylist()
            assert first_cluster_id not in samples_without_cluster["root"].to_pylist()

            # If a user has judged all available clusters, nothing is returned
            for cluster_id in resolution_clusters["id"].unique():
                cluster = resolution_clusters.filter(pl.col("id") == cluster_id)
                cluster_leaves = cluster.join(
                    source_clusters, on="key", suffix="_source"
                )["id_source"].to_list()

                self.backend.insert_judgement(
                    judgement=Judgement(
                        user_id=user_id,
                        shown=cluster_id,
                        endorsed=[cluster_leaves],
                    ),
                )

            samples_all_done = self.backend.sample_for_eval(
                n=99, resolution="naive_test_crn", user_id=user_id
            )
            assert len(samples_all_done) == 0
