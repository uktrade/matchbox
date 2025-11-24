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
    DataTypes,
    Match,
    ModelConfig,
    Resolution,
    ResolutionPath,
    SourceConfig,
    SourceField,
    UploadStage,
)
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxDataNotFound,
    MatchboxNoJudgements,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionExistingData,
    MatchboxResolutionNotFoundError,
    MatchboxResolutionUpdateError,
    MatchboxRunNotFoundError,
    MatchboxRunNotWriteable,
)
from matchbox.common.factories.entities import (
    SourceEntity,
    diff_results,
    query_to_cluster_entities,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.common.factories.sources import SourceTestkit
from matchbox.server.base import MatchboxDBAdapter

backends = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(scope="function")
def backend_instance(request: pytest.FixtureRequest, backend: str) -> MatchboxDBAdapter:
    """Create a fresh backend instance for each test."""
    backend_obj = request.getfixturevalue(backend)
    backend_obj.clear(certain=True)
    return backend_obj


@pytest.mark.parametrize("backend", backends)
@pytest.mark.docker
class TestMatchboxBackend:
    @pytest.fixture(autouse=True)
    def setup(self, backend_instance: str, sqlite_warehouse: Engine) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

    def test_properties(self) -> None:
        """Test that properties obey their protocol restrictions."""
        with self.scenario(self.backend, "index"):
            assert isinstance(self.backend.sources.list_all(), list)
            assert isinstance(self.backend.sources.count(), int)
            assert isinstance(self.backend.models.count(), int)
            assert isinstance(self.backend.data_clusters.count(), int)
            assert isinstance(self.backend.model_clusters.count(), int)
            assert isinstance(self.backend.all_clusters.count(), int)
            assert isinstance(self.backend.creates.count(), int)
            assert isinstance(self.backend.merges.count(), int)
            assert isinstance(self.backend.proposes.count(), int)

    # Retrieval

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

    # Collection management

    def test_collections(self) -> None:
        """Test creating, listing, getting and deleting collections."""
        with self.scenario(self.backend, "bare") as _:
            collections_pre = self.backend.list_collections()
            assert "test_collection" not in collections_pre

            # Create new collection and verify its initial properties
            test_collection_created = self.backend.create_collection("test_collection")

            assert test_collection_created.runs == []  # No versions yet

            collections_post = self.backend.list_collections()
            assert "test_collection" in collections_post

            # Verify duplicate creation is rejected
            with pytest.raises(MatchboxCollectionAlreadyExists):
                self.backend.create_collection("test_collection")

            test_collection = self.backend.get_collection("test_collection")

            assert test_collection == test_collection_created

            self.backend.delete_collection("test_collection", certain=True)

            # Verify collection is properly removed
            with pytest.raises(MatchboxCollectionNotFoundError):
                self.backend.get_collection("test_collection")

            with pytest.raises(MatchboxCollectionNotFoundError):
                self.backend.delete_collection("test_collection", certain=False)

    # Run management

    def test_runs(self) -> None:
        """Test creating, listing, getting and deleting runs."""
        with self.scenario(self.backend, "bare") as _:
            collections_pre = self.backend.list_collections()
            assert "test_collection" not in collections_pre

            # Create parent collection with no runs initially
            test_collection_pre = self.backend.create_collection("test_collection")
            assert len(test_collection_pre.runs) == 0  # No runs yet

            # Create first run and check default properties
            v1 = self.backend.create_run("test_collection")

            assert isinstance(v1.run_id, int)
            assert v1.is_default is False  # New runs aren't default by default
            assert v1.is_mutable is True  # New runs are mutable by default
            assert v1.resolutions == {}  # No resolutions yet

            test_collection_post = self.backend.get_collection("test_collection")
            assert v1.run_id in test_collection_post.runs

            v2 = self.backend.create_run("test_collection")

            with pytest.raises(ValueError, match="mutable"):
                self.backend.set_run_default("test_collection", v1.run_id, True)

            self.backend.set_run_mutable("test_collection", v1.run_id, False)
            self.backend.set_run_default("test_collection", v1.run_id, True)

            # Default run info also available on collection DTO
            collection = self.backend.get_collection("test_collection")
            assert collection.default_run == v1.run_id

            # Setting v2 as default should automatically unset v1 as default
            self.backend.set_run_mutable("test_collection", v2.run_id, False)
            self.backend.set_run_default("test_collection", v2.run_id, True)

            v1 = self.backend.get_run("test_collection", v1.run_id)
            v2 = self.backend.get_run("test_collection", v2.run_id)

            assert v1.is_mutable is False
            assert v1.is_default is False

            assert v2.is_mutable is False
            assert v2.is_default is True

            self.backend.delete_run("test_collection", v1.run_id, certain=True)

            # Verify run is properly removed
            with pytest.raises(MatchboxRunNotFoundError):
                self.backend.get_run("test_collection", v1.run_id)

            with pytest.raises(MatchboxRunNotFoundError):
                self.backend.delete_run("test_collection", v1.run_id, certain=False)

    def test_run_immutable(self) -> None:
        """Nothing in an immutable run can be changed."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            source_testkit = dag_testkit.sources["crn"]
            model_testkit = dag_testkit.models["naive_test_crn"]

            self.backend.set_run_mutable(
                collection=dag_testkit.dag.name,
                run_id=dag_testkit.dag.run,
                mutable=False,
            )

            with pytest.raises(MatchboxRunNotWriteable):
                self.backend.create_resolution(
                    path=source_testkit.resolution_path.model_copy(
                        update={"name": "new_source"}
                    ),
                    resolution=source_testkit.fake_run().source.to_resolution(),
                )

            with pytest.raises(MatchboxRunNotWriteable):
                self.backend.delete_resolution(
                    source_testkit.resolution_path, certain=True
                )

            with pytest.raises(MatchboxRunNotWriteable):
                self.backend.insert_source_data(
                    source_testkit.resolution_path, source_testkit.data_hashes
                )

            with pytest.raises(MatchboxRunNotWriteable):
                self.backend.insert_model_data(
                    model_testkit.resolution_path,
                    model_testkit.probabilities.to_arrow(),
                )

    # Resolution management

    def test_get_source(self) -> None:
        """Test querying data from the database."""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")

            crn_retrieved = self.backend.get_resolution(crn_testkit.resolution_path)
            assert isinstance(crn_retrieved, Resolution)
            assert crn_testkit.source_config == crn_retrieved.config

            with pytest.raises(MatchboxResolutionNotFoundError):
                self.backend.get_resolution(
                    path=ResolutionPath(collection="collection", run=1, name="foo")
                )

    def test_delete_resolution(self) -> None:
        """
        Tests the deletion of:

        * The resolution from the resolutions table
        * The source configuration attached to the resolution
        * Any models that depended on this model (descendants)
        * All probabilities for all descendants
        """
        with self.scenario(self.backend, "link") as dag_testkit:
            to_delete = dag_testkit.sources["crn"]
            total_sources = len(dag_testkit.sources)
            total_models = len(dag_testkit.models)

            source_configs_pre_delete = self.backend.sources.count()
            sources_pre_delete = self.backend.source_resolutions.count()
            models_pre_delete = self.backend.models.count()
            cluster_count_pre_delete = self.backend.model_clusters.count()
            cluster_assoc_count_pre_delete = self.backend.creates.count()
            proposed_merge_probs_pre_delete = self.backend.proposes.count()

            assert sources_pre_delete == total_sources
            assert models_pre_delete == total_models
            assert cluster_count_pre_delete > 0
            assert cluster_assoc_count_pre_delete > 0
            assert proposed_merge_probs_pre_delete > 0

            # Perform deletion
            self.backend.delete_resolution(to_delete.resolution_path, certain=True)

            source_configs_post_delete = self.backend.sources.count()
            sources_post_delete = self.backend.source_resolutions.count()
            models_post_delete = self.backend.models.count()
            cluster_count_post_delete = self.backend.model_clusters.count()
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

    def test_index_new_source(self) -> None:
        """Test that indexing a new source works."""
        with self.scenario(self.backend, "bare") as dag_testkit:
            crn_testkit: SourceTestkit = dag_testkit.sources.get("crn").fake_run()

            assert self.backend.model_clusters.count() == 0

            self.backend.create_resolution(
                crn_testkit.source.to_resolution(),
                path=crn_testkit.resolution_path,
            )

            # Resolution can't be re-added
            with pytest.raises(MatchboxResolutionAlreadyExists):
                self.backend.create_resolution(
                    crn_testkit.source.to_resolution(),
                    path=crn_testkit.resolution_path,
                )

            # After resolution metadata is present, we can add data
            self.backend.insert_source_data(
                crn_testkit.source.resolution_path, crn_testkit.data_hashes
            )

            # Data can't be re-added
            with pytest.raises(MatchboxResolutionExistingData):
                self.backend.insert_source_data(
                    crn_testkit.source.resolution_path, crn_testkit.data_hashes
                )

            # Resolution marked as complete
            assert (
                self.backend.get_resolution_stage(crn_testkit.source.resolution_path)
                == UploadStage.COMPLETE
            )

            # We can retrieve the resolution
            crn_retrieved = self.backend.get_resolution(
                crn_testkit.source.resolution_path
            )

            assert crn_testkit.source_config == crn_retrieved.config
            assert self.backend.data_clusters.count() == len(crn_testkit.data_hashes)

            assert self.backend.data_clusters.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 1

            # We can update the resolution metadata, including changes in fields
            updated_key_field = SourceField(name="new key", type=DataTypes.STRING)
            updated_index_fields = (
                SourceField(name="new field", type=DataTypes.BOOLEAN),
            )
            updated_config = SourceConfig.model_validate(
                crn_testkit.source.config.model_copy(
                    update={
                        "key_field": updated_key_field,
                        "index_fields": updated_index_fields,
                    }
                )
            )
            updated_resolution = Resolution.model_validate(
                crn_testkit.source.to_resolution().model_copy(
                    update={
                        "description": "updated",
                        "config": updated_config,
                    }
                )
            )
            self.backend.update_resolution(
                updated_resolution, path=crn_testkit.source.resolution_path
            )

            # We cannot update source resolution with different fingerprint
            with pytest.raises(MatchboxResolutionUpdateError, match="fingerprint"):
                self.backend.update_resolution(
                    updated_resolution.model_copy(update={"fingerprint": 123}),
                    path=crn_testkit.source.resolution_path,
                )

            # We cannot update source resolution with a model resolution
            with pytest.raises(MatchboxResolutionUpdateError, match="parents"):
                # Create model with same fingerprint as previous resolution
                valid_fingerprint = crn_testkit.source.to_resolution().fingerprint
                model_resolution = Resolution.model_validate(
                    model_factory()
                    .fake_run()
                    .model.to_resolution()
                    .model_copy(update={"fingerprint": valid_fingerprint})
                )
                self.backend.update_resolution(
                    model_resolution,
                    path=crn_testkit.source.resolution_path,
                )

            # We can retrieve the updated resolution
            crn_retrieved = self.backend.get_resolution(
                crn_testkit.source.resolution_path
            )
            assert crn_retrieved.description == "updated"
            assert crn_retrieved.config.key_field == updated_key_field
            assert crn_retrieved.config.index_fields == updated_index_fields

    def test_index_empty_source(self) -> None:
        """Can insert and retrieve empty source data"""
        with self.scenario(self.backend, "bare") as dag_testkit:
            crn_testkit: SourceTestkit = dag_testkit.sources.get("crn")
            crn_testkit.data_hashes = crn_testkit.data_hashes.slice(length=0)
            crn_testkit.fake_run()
            self.backend.create_resolution(
                crn_testkit.source.to_resolution(),
                path=crn_testkit.resolution_path,
            )

            self.backend.insert_source_data(
                crn_testkit.source.resolution_path, crn_testkit.data_hashes
            )
            # Resolution marked as complete
            assert (
                self.backend.get_resolution_stage(crn_testkit.source.resolution_path)
                == UploadStage.COMPLETE
            )

    def test_index_different_resolution_same_hashes(self) -> None:
        """Test that indexing data with the same hashes but different sources works."""
        with self.scenario(self.backend, "bare") as dag_testkit:
            # Prepare original source
            crn_testkit: SourceTestkit = dag_testkit.sources.get("crn").fake_run()
            # Create new source with same hashes
            dh_testkit: SourceTestkit = dag_testkit.sources.get("dh")
            dh_testkit.data_hashes = crn_testkit.data_hashes
            dh_testkit.fake_run()

            # Add original source
            self.backend.create_resolution(
                crn_testkit.source.to_resolution(), path=crn_testkit.resolution_path
            )
            self.backend.insert_source_data(
                crn_testkit.source.resolution_path, crn_testkit.data_hashes
            )
            # Add different source, with same hashes
            self.backend.create_resolution(
                dh_testkit.source.to_resolution(), path=dh_testkit.resolution_path
            )
            self.backend.insert_source_data(
                dh_testkit.source.resolution_path, crn_testkit.data_hashes
            )
            assert self.backend.data_clusters.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_resolutions.count() == 2

    def test_insert_model(self) -> None:
        """Test that models can be inserted."""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            dh_testkit = dag_testkit.sources.get("dh")
            # Assumes CRN and DH come from same LinkedSourcesTestkit
            linked = dag_testkit.source_to_linked["crn"]

            # Test deduper insertion
            models_count = self.backend.models.count()

            dedupe_1_testkit = model_factory(
                name="dedupe_1",
                description="Test deduper 1",
                dag=dag_testkit.dag,
                left_testkit=crn_testkit,
                true_entities=linked.true_entities,
            )
            self.backend.create_resolution(
                resolution=dedupe_1_testkit.fake_run().model.to_resolution(),
                path=dedupe_1_testkit.resolution_path,
            )

            dedupe_2_testkit = model_factory(
                name="dedupe_2",
                description="Test deduper 2",
                dag=dag_testkit.dag,
                left_testkit=dh_testkit,
                true_entities=linked.true_entities,
            )
            self.backend.create_resolution(
                resolution=dedupe_2_testkit.fake_run().model.to_resolution(),
                path=dedupe_2_testkit.resolution_path,
            )

            assert self.backend.models.count() == models_count + 2

            # Test linker insertion
            linker_testkit = model_factory(
                name="link_1",
                description="Test linker 1",
                left_testkit=dedupe_1_testkit,
                right_testkit=dedupe_2_testkit,
                true_entities=linked.true_entities,
            )
            self.backend.create_resolution(
                resolution=linker_testkit.fake_run().model.to_resolution(),
                path=linker_testkit.resolution_path,
            )

            assert self.backend.models.count() == models_count + 3

            # We cannot re-create under the same path
            with pytest.raises(MatchboxResolutionAlreadyExists):
                self.backend.create_resolution(
                    linker_testkit.fake_run().model.to_resolution(),
                    path=linker_testkit.resolution_path,
                )

            assert self.backend.models.count() == models_count + 3

            # Can update model resolution
            old_resolution = linker_testkit.model.to_resolution()
            updated_config = ModelConfig.model_validate(
                old_resolution.config.model_copy(
                    update={
                        "left_query": old_resolution.config.left_query.model_copy(
                            update={"threshold": 99}
                        )
                    }
                )
            )
            updated_resolution = Resolution.model_validate(
                old_resolution.model_copy(
                    update={
                        "description": "updated",
                        "truth": 33,
                        "config": updated_config,
                    }
                )
            )
            self.backend.update_resolution(
                resolution=updated_resolution,
                path=linker_testkit.resolution_path,
            )

            # We can retrieve the updated resolution
            linker_retrieved = self.backend.get_resolution(
                linker_testkit.resolution_path
            )
            assert linker_retrieved.description == "updated"
            assert linker_retrieved.truth == 33
            assert linker_retrieved.config.left_query.threshold == 99

            # We cannot change a model's inputs
            rewired_config = ModelConfig.model_validate(
                old_resolution.config.model_copy(
                    update={
                        "left_query": old_resolution.config.left_query.model_copy(
                            update={"model_resolution": "new_model"}
                        )
                    }
                )
            )
            rewired_resolution = Resolution.model_validate(
                old_resolution.model_copy(update={"config": rewired_config})
            )

            with pytest.raises(MatchboxResolutionUpdateError, match="parents"):
                self.backend.update_resolution(
                    resolution=rewired_resolution,
                    path=linker_testkit.resolution_path,
                )

            # We cannot change model results fingerprint
            with pytest.raises(MatchboxResolutionUpdateError, match="fingerprint"):
                corrupt_resolution = Resolution.model_validate(
                    updated_resolution.model_copy(update={"fingerprint": b"fake"})
                )
                self.backend.update_resolution(
                    resolution=corrupt_resolution,
                    path=linker_testkit.resolution_path,
                )

    def test_model_results_basic(self) -> None:
        """Test that a model's results data can be set and retrieved."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            # Query returns the same results as the testkit, showing
            # that processing was performed accurately.
            # (that we can query from it implies the resolution was correctly
            # marked as complete)
            res = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )
            res_clusters = query_to_cluster_entities(
                data=res,
                keys={crn_testkit.name: "key"},
            )

            identical, report = diff_results(
                expected=naive_crn_testkit.entities,
                actual=res_clusters,
            )

            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_data(
                path=naive_crn_testkit.resolution_path
            )

            assert isinstance(pre_results, pa.Table)
            assert len(pre_results) > 0

            # Validate IDs
            self.backend.validate_ids(ids=pre_results["left_id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["right_id"].to_pylist())

            # Cannot set new results
            with pytest.raises(MatchboxResolutionExistingData):
                self.backend.insert_model_data(
                    path=naive_crn_testkit.resolution_path,
                    results=naive_crn_testkit.probabilities.to_arrow(),
                )

            # Retrieve again
            post_results = self.backend.get_model_data(
                path=naive_crn_testkit.resolution_path
            )

            # Check difference
            assert pre_results == post_results

    def test_model_results_probabilistic(self) -> None:
        """Test that a probabilistic model's results data can be set and retrieved."""
        with self.scenario(self.backend, "probabilistic_dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            prob_crn_testkit = dag_testkit.models.get("probabilistic_test_crn")

            # Query returns the same results as the testkit, showing
            # that processing was performed accurately
            res = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=prob_crn_testkit.resolution_path,
            )
            res_clusters = query_to_cluster_entities(
                data=res,
                keys={crn_testkit.name: "key"},
            )

            identical, report = diff_results(
                expected=prob_crn_testkit.entities,
                actual=res_clusters,
            )
            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_data(
                path=prob_crn_testkit.resolution_path
            )

            assert isinstance(pre_results, pa.Table)
            assert len(pre_results) > 0

            # Validate IDs
            self.backend.validate_ids(ids=pre_results["left_id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["right_id"].to_pylist())

    def test_model_results_empty(self) -> None:
        """Can insert and retrieve empty model results"""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            model_testkit = model_factory(
                model_type="deduper", dag=crn_testkit.source.dag
            )
            model_testkit.probabilities = model_testkit.probabilities.head(0)
            model_testkit.fake_run()

            self.backend.create_resolution(
                model_testkit.model.to_resolution(), path=model_testkit.resolution_path
            )

            self.backend.insert_model_data(
                path=model_testkit.model.resolution_path,
                results=model_testkit.model.results.probabilities.to_arrow(),
            )

            # Querying from deduper with no results is the same as querying from source
            # (That we can query also implies that resolution marked as complete)
            source_query = self.backend.query(source=crn_testkit.resolution_path)
            dedupe_query = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=model_testkit.resolution_path,
            )

            assert source_query == dedupe_query

    def test_model_results_shared_clusters(self) -> None:
        """Test that model results data can be inserted when clusters are shared."""
        with self.scenario(self.backend, "convergent_partial") as dag_testkit:
            for model_testkit in dag_testkit.models.values():
                self.backend.create_resolution(
                    path=model_testkit.resolution_path,
                    resolution=model_testkit.fake_run().model.to_resolution(),
                )
                self.backend.insert_model_data(
                    path=model_testkit.resolution_path,
                    results=model_testkit.probabilities.to_arrow(),
                )

    # Data management

    def test_validate_ids(self) -> None:
        """Test validating data IDs."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            df_crn = self.backend.query(
                source=crn_testkit.source.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )

            ids = df_crn["id"].to_pylist()
            assert len(ids) > 0
            self.backend.validate_ids(ids=ids)

            with pytest.raises(MatchboxDataNotFound):
                self.backend.validate_ids(ids=[-6])

    def test_clear(self) -> None:
        """Test deleting all rows in the database."""
        with self.scenario(self.backend, "dedupe"):
            assert self.backend.sources.count() > 0
            assert self.backend.data_clusters.count() > 0
            assert self.backend.models.count() > 0
            assert self.backend.model_clusters.count() > 0
            assert self.backend.creates.count() > 0
            assert self.backend.merges.count() > 0
            assert self.backend.proposes.count() > 0

            self.backend.clear(certain=True)

            assert self.backend.sources.count() == 0
            assert self.backend.data_clusters.count() == 0
            assert self.backend.models.count() == 0
            assert self.backend.model_clusters.count() == 0
            assert self.backend.creates.count() == 0
            assert self.backend.merges.count() == 0
            assert self.backend.proposes.count() == 0

    def test_clear_and_restore(self) -> None:
        """Test that clearing and restoring the database works."""
        with self.scenario(self.backend, "link") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            count_funcs = [
                self.backend.sources.count,
                self.backend.models.count,
                self.backend.data_clusters.count,
                self.backend.model_clusters.count,
                self.backend.all_clusters.count,
                self.backend.merges.count,
                self.backend.creates.count,
                self.backend.proposes.count,
            ]

            def get_counts() -> list[int]:
                return [f() for f in count_funcs]

            # Verify we have data
            pre_dump_counts = get_counts()
            assert all(count > 0 for count in pre_dump_counts)

            # Get some specific IDs to verify they're restored properly
            df_crn_before = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )
            sample_ids_before = df_crn_before["id"].to_pylist()[:5]  # Take first 5 IDs

            # Dump the database
            snapshot = self.backend.dump()

        with self.scenario(self.backend, "bare") as _:
            # Verify counts match pre-dump state
            assert all(c == 0 for c in get_counts())

            # Restore from snapshot
            self.backend.restore(snapshot)

            # Verify counts match pre-dump state
            assert get_counts() == pre_dump_counts

            # Verify specific data was restored correctly
            df_crn_after = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )
            sample_ids_after = df_crn_after["id"].to_pylist()[:5]  # Take first 5 IDs

            # The same IDs should be present after restoration
            assert set(sample_ids_before) == set(sample_ids_after)

            # Test that restoring also clears the database
            self.backend.restore(snapshot)

            # Verify counts still match
            assert get_counts() == pre_dump_counts

    # User management

    def test_login(self) -> None:
        """Can swap user name with user ID."""
        with self.scenario(self.backend, "bare") as _:
            alice_id = self.backend.login("alice")
            assert alice_id == self.backend.login("alice")
            assert alice_id != self.backend.login("bob")

    # Evaluation management

    def test_insert_and_get_judgement(self) -> None:
        """Can insert and retrieve judgements."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            # To begin with, no judgements to retrieve
            judgements, expansion = self.backend.get_judgements()
            assert len(judgements) == len(expansion) == 0

            # Do some queries to find real source cluster IDs
            deduped_query = pl.from_arrow(
                self.backend.query(
                    source=crn_testkit.resolution_path,
                    point_of_truth=naive_crn_testkit.resolution_path,
                )
            )
            unique_ids = deduped_query["id"].unique()
            all_leaves = pl.from_arrow(
                self.backend.query(source=crn_testkit.resolution_path)
            )

            def get_leaf_ids(cluster_id: int) -> list[int]:
                return (
                    deduped_query.filter(pl.col("id") == cluster_id)
                    .join(all_leaves, on="key", suffix="_leaf")["id_leaf"]
                    .to_list()
                )

            alice_id = self.backend.login("alice")

            original_cluster_num = self.backend.model_clusters.count()

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
            assert self.backend.model_clusters.count() == original_cluster_num

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
            assert self.backend.model_clusters.count() == original_cluster_num + 2

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
            assert judgements.schema.equals(SCHEMA_JUDGEMENTS)
            assert expansion.schema.equals(SCHEMA_CLUSTER_EXPANSION)
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
            assert set(endorsed_dict[unique_ids[0]]) == set(clust1_leaves)
            # Other than the root we know about, there are two new ones
            assert len(set(endorsed_dict.keys())) == 3
            # The other two sets of leaves are there too
            assert set(map(frozenset, endorsed_dict.values())) == set(
                map(frozenset, [clust1_leaves, clust2_leaves[:1], clust2_leaves[1:]])
            )

    def test_compare_models_fails(self) -> None:
        """Model comparison errors with no judgement data."""
        with (
            self.scenario(self.backend, "bare"),
            pytest.raises(MatchboxNoJudgements),
        ):
            self.backend.compare_models([])

    def test_compare_models(self) -> None:
        """Can compute precision and recall for list of models."""
        with self.scenario(self.backend, "alt_dedupe") as dag_testkit:
            user_id = self.backend.login("alice")

            model_names = [
                model.resolution_path for model in dag_testkit.models.values()
            ]

            root_leaves = (
                pl.from_arrow(
                    self.backend.sample_for_eval(
                        n=10,
                        path=model_names[0],
                        user_id=user_id,
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

            pr = self.backend.compare_models(model_names)
            # Precision must be 1 for both as the second model is like the first
            # but more conservative
            assert pr[model_names[0]][0] == pr[model_names[1]][0] == 1
            # Recall must be 1 for the first model and lower for the second
            assert pr[model_names[0]][1] == 1
            assert pr[model_names[1]][1] < 1

    def test_sample_for_eval(self) -> None:
        """Can extract samples for a user and a resolution."""

        # Missing resolution raises error
        with (
            self.scenario(self.backend, "bare"),
            pytest.raises(MatchboxResolutionNotFoundError, match="naive_test_crn"),
        ):
            user_id = self.backend.login("alice")
            self.backend.sample_for_eval(
                n=10,
                path=ResolutionPath(
                    collection="collection", run=1, name="naive_test_crn"
                ),
                user_id=user_id,
            )

        # Convergent scenario allows testing we don't accidentally return metadata
        # for sources that aren't relevant for a point of truth
        with self.scenario(self.backend, "convergent") as dag_testkit:
            source_testkit = dag_testkit.sources.get("foo_a")
            model_testkit = dag_testkit.models.get("naive_test_foo_a")

            user_id = self.backend.login("alice")

            # Source clusters should not be returned
            # So if we sample from a source resolution, we get nothing
            user_id = self.backend.login("alice")
            samples_source = self.backend.sample_for_eval(
                n=10, path=source_testkit.resolution_path, user_id=user_id
            )
            assert len(samples_source) == 0

            # We now look at more interesting cases
            # Query backend to form expectations
            resolution_clusters = pl.from_arrow(
                self.backend.query(
                    source=source_testkit.resolution_path,
                    point_of_truth=model_testkit.resolution_path,
                )
            )
            source_clusters = pl.from_arrow(
                self.backend.query(source=source_testkit.resolution_path)
            )
            # We can request more than available
            assert len(resolution_clusters["id"].unique()) < 99

            samples_99 = self.backend.sample_for_eval(
                n=99, path=model_testkit.resolution_path, user_id=user_id
            )

            assert samples_99.schema.equals(SCHEMA_EVAL_SAMPLES)

            # We can reconstruct the expected sample from resolution and source queries
            expected_sample = (
                resolution_clusters.join(source_clusters, on="key", suffix="_source")
                .rename({"id": "root", "id_source": "leaf"})
                .with_columns(pl.lit("foo_a").alias("source"))
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
                n=5, path=model_testkit.resolution_path, user_id=user_id
            )
            assert len(samples_5["root"].unique()) == 5

            # If user has recent judgements, exclude clusters
            first_cluster_id = resolution_clusters["id"][0]
            first_cluster = resolution_clusters.filter(pl.col("id") == first_cluster_id)
            first_cluster_leaves = (
                first_cluster.join(source_clusters, on="key", suffix="_source")[
                    "id_source"
                ]
                .unique()  # multiple keys can map to same cluster
                .to_list()
            )

            self.backend.insert_judgement(
                judgement=Judgement(
                    user_id=user_id,
                    shown=first_cluster_id,
                    endorsed=[first_cluster_leaves],
                ),
            )

            samples_without_cluster = self.backend.sample_for_eval(
                n=99, path=model_testkit.resolution_path, user_id=user_id
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
                cluster_leaves = (
                    cluster.join(source_clusters, on="key", suffix="_source")[
                        "id_source"
                    ]
                    .unique()  # multiple keys can map to same cluster
                    .to_list()
                )

                self.backend.insert_judgement(
                    judgement=Judgement(
                        user_id=user_id,
                        shown=cluster_id,
                        endorsed=[cluster_leaves],
                    ),
                )

            samples_all_done = self.backend.sample_for_eval(
                n=99, path=model_testkit.resolution_path, user_id=user_id
            )
            assert len(samples_all_done) == 0

    def test_delete_orphans(self) -> None:
        """Can delete orphaned clusters."""
        with self.scenario(self.backend, "link") as dag_testkit:
            # delete resolution to create orphans
            res = dag_testkit.models["naive_test_crn"].resolution_path
            self.backend.delete_resolution(res, certain=True)

            # check clusters in scenario
            clusters = self.backend.all_clusters.count()
            assert clusters == 85

            # delete orphans
            orphans = self.backend.delete_orphans()
            assert orphans == 30

            # check clusters again
            clusters = self.backend.all_clusters.count()
            assert clusters == 55
