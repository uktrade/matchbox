"""Test the backend adapter's admin functions."""

from functools import partial

import pyarrow as pa
import pytest
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.datatypes import DataTypes
from matchbox.common.dtos import (
    ModelConfig,
    Resolution,
    ResolutionPath,
    SourceConfig,
    SourceField,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionExistingData,
    MatchboxResolutionNotFoundError,
    MatchboxResolutionUpdateError,
    MatchboxRunNotFoundError,
    MatchboxRunNotWriteable,
)
from matchbox.common.factories.entities import diff_results, query_to_cluster_entities
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.common.factories.sources import SourceTestkit
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.docker
class TestMatchboxCollectionsBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

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
            assert self.backend.source_clusters.count() == len(crn_testkit.data_hashes)

            assert self.backend.source_clusters.count() == len(crn_testkit.data_hashes)
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
            assert self.backend.source_clusters.count() == len(crn_testkit.data_hashes)
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
