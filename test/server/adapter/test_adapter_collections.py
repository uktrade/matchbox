"""Test the backend adapter's admin functions."""

from functools import partial

import pyarrow as pa
import pytest
from sqlalchemy import Engine
from test.fixtures.db import SERVER_BACKENDS

from matchbox.client.queries import Query
from matchbox.common.datatypes import DataTypes
from matchbox.common.dtos import (
    DefaultGroup,
    GroupName,
    ModelConfig,
    PermissionGrant,
    PermissionType,
    QueryCombineType,
    SourceConfig,
    SourceField,
    Step,
    StepPath,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxRunNotFoundError,
    MatchboxRunNotWriteable,
    MatchboxStepAlreadyExists,
    MatchboxStepExistingData,
    MatchboxStepNotFoundError,
    MatchboxStepTypeError,
    MatchboxStepUpdateError,
)
from matchbox.common.factories.entities import diff_entities, query_to_cluster_entities
from matchbox.common.factories.models import model_factory, query_to_model_factory
from matchbox.common.factories.resolvers import resolver_factory
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.common.factories.sources import SourceTestkit
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", SERVER_BACKENDS)
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
            default_permissions = [
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.READ,
                ),
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.WRITE,
                ),
            ]
            test_collection_created = self.backend.create_collection(
                name="test_collection",
                permissions=default_permissions,
            )

            assert test_collection_created.runs == []  # No versions yet

            collections_post = self.backend.list_collections()
            assert "test_collection" in collections_post

            # Verify duplicate creation is rejected
            with pytest.raises(MatchboxCollectionAlreadyExists):
                self.backend.create_collection(
                    name="test_collection",
                    permissions=default_permissions,
                )

            test_collection = self.backend.get_collection("test_collection")

            assert test_collection == test_collection_created

            self.backend.delete_collection("test_collection", certain=True)

            # Verify collection is properly removed
            with pytest.raises(MatchboxCollectionNotFoundError):
                self.backend.get_collection("test_collection")

            with pytest.raises(MatchboxCollectionNotFoundError):
                self.backend.delete_collection("test_collection", certain=False)

    def test_collection_with_custom_permissions(self) -> None:
        """Test creating a collection with custom permissions."""
        with self.scenario(self.backend, "closed_collection") as _:
            # The 'restricted' collection already has custom permissions
            # Verify permissions were set correctly
            grants = self.backend.get_permissions("restricted")

            # Should have READ for readers, WRITE for writers
            assert len(grants) == 3
            assert (
                PermissionGrant(
                    group_name=GroupName("readers"), permission=PermissionType.READ
                )
                in grants
            )
            assert (
                PermissionGrant(
                    group_name=GroupName("writers"), permission=PermissionType.WRITE
                )
                in grants
            )

    def test_collection_default_permissions(self) -> None:
        """Test that collections get default public read/write permissions."""
        with self.scenario(self.backend, "bare") as _:
            # Create collection with default permissions
            default_permissions = [
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.READ,
                ),
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.WRITE,
                ),
            ]
            self.backend.create_collection(
                name="default_perms_collection",
                permissions=default_permissions,
            )

            # Verify default permissions were set
            grants = self.backend.get_permissions("default_perms_collection")

            assert len(grants) == 2
            assert (
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.READ,
                )
                in grants
            )
            assert (
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.WRITE,
                )
                in grants
            )

            self.backend.delete_collection("default_perms_collection", certain=True)

    def test_collection_empty_permissions(self) -> None:
        """Test creating a collection with no permissions."""
        with self.scenario(self.backend, "closed_collection") as _:
            # Create collection with explicitly empty permissions
            self.backend.create_collection(
                name="no_perms_collection",
                permissions=[],
            )

            # Verify no permissions were set
            grants = self.backend.get_permissions("no_perms_collection")
            assert len(grants) == 0

            # Verify that dave (who is only in public group) cannot access it
            assert not self.backend.check_permission(
                "dave", PermissionType.READ, "no_perms_collection"
            )

            self.backend.delete_collection("no_perms_collection", certain=True)

    # Run management

    def test_runs(self) -> None:
        """Test creating, listing, getting and deleting runs."""
        with self.scenario(self.backend, "bare") as _:
            collections_pre = self.backend.list_collections()
            assert "test_collection" not in collections_pre

            # Create parent collection with no runs initially
            default_permissions = [
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.READ,
                ),
                PermissionGrant(
                    group_name=GroupName(DefaultGroup.PUBLIC),
                    permission=PermissionType.WRITE,
                ),
            ]
            test_collection_pre = self.backend.create_collection(
                name="test_collection",
                permissions=default_permissions,
            )
            assert len(test_collection_pre.runs) == 0  # No runs yet

            # Create first run and check default properties
            v1 = self.backend.create_run("test_collection")

            assert isinstance(v1.run_id, int)
            assert v1.is_default is False  # New runs aren't default by default
            assert v1.is_mutable is True  # New runs are mutable by default
            assert v1.steps == {}  # No steps yet

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
                self.backend.create_step(
                    path=source_testkit.path.model_copy(update={"name": "new_source"}),
                    step=source_testkit.fake_run().source.to_dto(),
                )

            with pytest.raises(MatchboxRunNotWriteable):
                self.backend.delete_step(source_testkit.path, certain=True)

            with pytest.raises(MatchboxRunNotWriteable):
                self.backend.insert_source_data(
                    source_testkit.path, source_testkit.data_hashes
                )

            with pytest.raises(MatchboxRunNotWriteable):
                self.backend.insert_model_data(
                    model_testkit.path,
                    model_testkit.scores.to_arrow(),
                )

    # Step management

    def test_get_source(self) -> None:
        """Test querying data from the database."""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")

            crn_retrieved = self.backend.get_step(crn_testkit.path)
            assert isinstance(crn_retrieved, Step)
            assert crn_testkit.source_config == crn_retrieved.config

            with pytest.raises(MatchboxStepNotFoundError):
                self.backend.get_step(
                    path=StepPath(collection="collection", run=1, name="foo")
                )

    def test_delete_step(self) -> None:
        """
        Tests the deletion of:

        * The step from the steps table
        * The source configuration attached to the step
        * Any models that depended on this model (descendants)
        * All scores for all descendants
        """
        with self.scenario(self.backend, "link") as dag_testkit:
            to_delete = dag_testkit.sources["crn"]
            total_sources = len(dag_testkit.sources)
            total_models = len(dag_testkit.models)

            source_configs_pre_delete = self.backend.sources.count()
            sources_pre_delete = self.backend.source_steps.count()
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
            self.backend.delete_step(to_delete.path, certain=True)

            source_configs_post_delete = self.backend.sources.count()
            sources_post_delete = self.backend.source_steps.count()
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
        with self.scenario(self.backend, "preindex") as dag_testkit:
            crn_testkit: SourceTestkit = dag_testkit.sources.get("crn").fake_run()

            assert self.backend.model_clusters.count() == 0

            self.backend.create_step(
                crn_testkit.source.to_dto(),
                path=crn_testkit.path,
            )

            # Step can't be re-added
            with pytest.raises(MatchboxStepAlreadyExists):
                self.backend.create_step(
                    crn_testkit.source.to_dto(),
                    path=crn_testkit.path,
                )

            # After step metadata is present, we can add data
            self.backend.insert_source_data(
                crn_testkit.source.path, crn_testkit.data_hashes
            )

            # Data can't be re-added
            with pytest.raises(MatchboxStepExistingData):
                self.backend.insert_source_data(
                    crn_testkit.source.path, crn_testkit.data_hashes
                )

            # Step marked as complete
            assert (
                self.backend.get_step_stage(crn_testkit.source.path)
                == UploadStage.COMPLETE
            )

            # We can retrieve the step
            crn_retrieved = self.backend.get_step(crn_testkit.source.path)

            assert crn_testkit.source_config == crn_retrieved.config
            assert self.backend.source_clusters.count() == len(crn_testkit.data_hashes)

            assert self.backend.source_clusters.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_steps.count() == 1

            # We can update the step metadata, including changes in fields
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
            updated_step = Step.model_validate(
                crn_testkit.source.to_dto().model_copy(
                    update={
                        "description": "updated",
                        "config": updated_config,
                    }
                )
            )
            self.backend.update_step(updated_step, path=crn_testkit.source.path)

            # We cannot update source step with different fingerprint
            with pytest.raises(MatchboxStepUpdateError, match="fingerprint"):
                self.backend.update_step(
                    updated_step.model_copy(update={"fingerprint": 123}),
                    path=crn_testkit.source.path,
                )

            # We cannot update source step with a model step
            with pytest.raises(MatchboxStepUpdateError, match="parents"):
                # Create model with same fingerprint as previous step
                valid_fingerprint = crn_testkit.source.to_dto().fingerprint
                model_step = Step.model_validate(
                    model_factory()
                    .fake_run()
                    .model.to_dto()
                    .model_copy(update={"fingerprint": valid_fingerprint})
                )
                self.backend.update_step(
                    model_step,
                    path=crn_testkit.source.path,
                )

            # We can retrieve the updated step
            crn_retrieved = self.backend.get_step(crn_testkit.source.path)
            assert crn_retrieved.description == "updated"
            assert crn_retrieved.config.key_field == updated_key_field
            assert crn_retrieved.config.index_fields == updated_index_fields

    def test_index_empty_source(self) -> None:
        """Can insert and retrieve empty source data"""
        with self.scenario(self.backend, "preindex") as dag_testkit:
            crn_testkit: SourceTestkit = dag_testkit.sources.get("crn")
            crn_testkit.data_hashes = crn_testkit.data_hashes.slice(length=0)
            crn_testkit.fake_run()
            self.backend.create_step(
                crn_testkit.source.to_dto(),
                path=crn_testkit.path,
            )

            self.backend.insert_source_data(
                crn_testkit.source.path, crn_testkit.data_hashes
            )
            # Step marked as complete
            assert (
                self.backend.get_step_stage(crn_testkit.source.path)
                == UploadStage.COMPLETE
            )

    def test_index_different_step_same_hashes(self) -> None:
        """Test that indexing data with the same hashes but different sources works."""
        with self.scenario(self.backend, "preindex") as dag_testkit:
            # Prepare original source
            crn_testkit: SourceTestkit = dag_testkit.sources.get("crn").fake_run()
            # Create new source with same hashes
            dh_testkit: SourceTestkit = dag_testkit.sources.get("dh")
            dh_testkit.data_hashes = crn_testkit.data_hashes
            dh_testkit.fake_run()

            # Add original source
            self.backend.create_step(crn_testkit.source.to_dto(), path=crn_testkit.path)
            self.backend.insert_source_data(
                crn_testkit.source.path, crn_testkit.data_hashes
            )
            # Add different source, with same hashes
            self.backend.create_step(dh_testkit.source.to_dto(), path=dh_testkit.path)
            self.backend.insert_source_data(
                dh_testkit.source.path, crn_testkit.data_hashes
            )
            assert self.backend.source_clusters.count() == len(crn_testkit.data_hashes)
            assert self.backend.source_steps.count() == 2

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
            self.backend.create_step(
                step=dedupe_1_testkit.fake_run().model.to_dto(),
                path=dedupe_1_testkit.path,
            )

            dedupe_2_testkit = model_factory(
                name="dedupe_2",
                description="Test deduper 2",
                dag=dag_testkit.dag,
                left_testkit=dh_testkit,
                true_entities=linked.true_entities,
            )
            self.backend.create_step(
                step=dedupe_2_testkit.fake_run().model.to_dto(),
                path=dedupe_2_testkit.path,
            )

            for dedupe_testkit in (dedupe_1_testkit, dedupe_2_testkit):
                resolver_testkit = resolver_factory(
                    dag=dag_testkit.dag,
                    name=f"resolver_{dedupe_testkit.name}",
                    inputs=[dedupe_testkit],
                ).fake_run()
                self.backend.create_step(
                    step=resolver_testkit.resolver.to_dto(),
                    path=resolver_testkit.resolver.path,
                )

            assert self.backend.models.count() == models_count + 2

            # Test linker insertion
            linker_testkit = model_factory(
                name="link_1",
                description="Test linker 1",
                left_testkit=crn_testkit,
                right_testkit=dh_testkit,
                true_entities=linked.true_entities,
            )
            self.backend.create_step(
                step=linker_testkit.fake_run().model.to_dto(),
                path=linker_testkit.path,
            )

            assert self.backend.models.count() == models_count + 3

            # We cannot re-create under the same path
            with pytest.raises(MatchboxStepAlreadyExists):
                self.backend.create_step(
                    linker_testkit.fake_run().model.to_dto(),
                    path=linker_testkit.path,
                )

            assert self.backend.models.count() == models_count + 3

            # Can update model step
            old_step = linker_testkit.model.to_dto()
            updated_config = ModelConfig.model_validate(
                old_step.config.model_copy(
                    update={
                        "left_query": old_step.config.left_query.model_copy(
                            update={"combine_type": QueryCombineType.SET_AGG}
                        )
                    }
                )
            )
            updated_step = Step.model_validate(
                old_step.model_copy(
                    update={
                        "description": "updated",
                        "config": updated_config,
                    }
                )
            )
            self.backend.update_step(
                step=updated_step,
                path=linker_testkit.path,
            )

            # We can retrieve the updated step
            linker_retrieved = self.backend.get_step(linker_testkit.path)
            assert linker_retrieved.description == "updated"
            assert (
                linker_retrieved.config.left_query.combine_type
                == QueryCombineType.SET_AGG
            )

            # We cannot change a model's inputs
            rewired_config = ModelConfig.model_validate(
                old_step.config.model_copy(
                    update={
                        "left_query": old_step.config.left_query.model_copy(
                            update={"sources": ("new_source",)}
                        )
                    }
                )
            )
            rewired_step = Step.model_validate(
                old_step.model_copy(update={"config": rewired_config})
            )

            with pytest.raises(MatchboxStepUpdateError, match="parents"):
                self.backend.update_step(
                    step=rewired_step,
                    path=linker_testkit.path,
                )

            # We cannot change model results fingerprint
            with pytest.raises(MatchboxStepUpdateError, match="fingerprint"):
                corrupt_step = Step.model_validate(
                    updated_step.model_copy(update={"fingerprint": b"fake"})
                )
                self.backend.update_step(
                    step=corrupt_step,
                    path=linker_testkit.path,
                )

    def test_insert_model_rejects_model_parent(self) -> None:
        """Model parents must be source or resolver steps."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            source_testkit = dag_testkit.sources.get("crn")
            invalid_model = model_factory(
                name="invalid_model_parent",
                dag=dag_testkit.dag,
                left_testkit=source_testkit,
                true_entities=dag_testkit.source_to_linked["crn"].true_entities,
            ).fake_run()

            bad_parent = dag_testkit.models.get("naive_test_crn").name
            invalid_config = ModelConfig.model_validate(
                invalid_model.model.config.model_copy(
                    update={
                        "left_query": invalid_model.model.config.left_query.model_copy(
                            update={
                                "resolver": bad_parent,
                            }
                        )
                    }
                )
            )
            invalid_step = Step.model_validate(
                invalid_model.model.to_dto().model_copy(
                    update={"config": invalid_config}
                )
            )

            with pytest.raises(
                MatchboxStepTypeError,
                match="Expected one of: source, resolver",
            ):
                self.backend.create_step(
                    step=invalid_step,
                    path=invalid_model.path,
                )

    def test_insert_resolver(self) -> None:
        """Test that resolvers can be inserted and managed."""
        with self.scenario(self.backend, "link") as dag_testkit:
            # Setup
            linked = dag_testkit.source_to_linked["crn"]

            dedupe_crn_testkit = dag_testkit.models["naive_test_crn"]
            dedupe_dh_testkit = dag_testkit.models["naive_test_dh"]
            linker_testkit = dag_testkit.models[
                "deterministic_naive_test_crn_naive_test_dh"
            ]

            resolvers_count = self.backend.resolvers.count()

            # Insert a resolver over a single model
            resolver_single_testkit = resolver_factory(
                dag=dag_testkit.dag,
                name="resolver_dedupe_crn_alt",
                description="Test resolver for a single dedupe model",
                inputs=[dedupe_crn_testkit],
                true_entities=linked.true_entities,
            ).fake_run()

            self.backend.create_step(
                step=resolver_single_testkit.resolver.to_dto(),
                path=resolver_single_testkit.resolver.path,
            )
            assert self.backend.resolvers.count() == resolvers_count + 1

            self.backend.insert_resolver_data(
                path=resolver_single_testkit.resolver.path,
                data=resolver_single_testkit.resolver.results.to_arrow(),
            )
            assert (
                self.backend.get_step_stage(resolver_single_testkit.resolver.path)
                == UploadStage.COMPLETE
            )

            # Insert a resolver over a mutliple models
            resolver_multi_testkit = resolver_factory(
                dag=dag_testkit.dag,
                name="resolver_multi",
                description="Test resolver combining two dedupe models",
                inputs=[dedupe_crn_testkit, dedupe_dh_testkit],
                true_entities=linked.true_entities,
            ).fake_run()

            self.backend.create_step(
                step=resolver_multi_testkit.resolver.to_dto(),
                path=resolver_multi_testkit.resolver.path,
            )
            assert self.backend.resolvers.count() == resolvers_count + 2

            self.backend.insert_resolver_data(
                path=resolver_multi_testkit.resolver.path,
                data=resolver_multi_testkit.resolver.results.to_arrow(),
            )

            # Insert a resolver that creates diamond-shaped dependencies
            resolver_diamond_testkit = resolver_factory(
                dag=dag_testkit.dag,
                name="resolver_diamond",
                description="Test resolver with diamond dependency",
                inputs=[dedupe_crn_testkit, linker_testkit],
                true_entities=linked.true_entities,
            ).fake_run()

            self.backend.create_step(
                step=resolver_diamond_testkit.resolver.to_dto(),
                path=resolver_diamond_testkit.resolver.path,
            )
            assert self.backend.resolvers.count() == resolvers_count + 3

            self.backend.insert_resolver_data(
                path=resolver_diamond_testkit.resolver.path,
                data=resolver_diamond_testkit.resolver.results.to_arrow(),
            )

            # We cannot re-create under the same path
            with pytest.raises(MatchboxStepAlreadyExists):
                self.backend.create_step(
                    step=resolver_single_testkit.resolver.to_dto(),
                    path=resolver_single_testkit.resolver.path,
                )

            assert self.backend.resolvers.count() == resolvers_count + 3

            # We cannot re-insert over existing data
            with pytest.raises(MatchboxStepExistingData):
                self.backend.insert_resolver_data(
                    path=resolver_single_testkit.resolver.path,
                    data=resolver_single_testkit.resolver.results.to_arrow(),
                )

            # Can update resolver step
            old_step = resolver_single_testkit.resolver.to_dto()
            updated_config = old_step.config.model_copy(
                update={"resolver_settings": {"threshold": 0.8}}
            )
            updated_step = Step.model_validate(
                old_step.model_copy(
                    update={
                        "description": "updated resolver description",
                        "config": updated_config,
                    }
                )
            )

            self.backend.update_step(
                step=updated_step,
                path=resolver_single_testkit.resolver.path,
            )

            # We can retrieve the updated step
            resolver_retrieved = self.backend.get_step(
                resolver_single_testkit.resolver.path
            )
            assert resolver_retrieved.description == "updated resolver description"
            assert resolver_retrieved.config.resolver_settings == {"threshold": 0.8}

            # We cannot change a resolver's inputs
            rewired_config = old_step.config.model_copy(
                update={"inputs": (dedupe_dh_testkit.name,)}
            )
            rewired_step = Step.model_validate(
                old_step.model_copy(update={"config": rewired_config})
            )

            with pytest.raises(MatchboxStepUpdateError, match="parents"):
                self.backend.update_step(
                    step=rewired_step,
                    path=resolver_single_testkit.resolver.path,
                )

            # We cannot change a resolver's fingerprint
            with pytest.raises(MatchboxStepUpdateError, match="fingerprint"):
                corrupt_step = Step.model_validate(
                    updated_step.model_copy(update={"fingerprint": b"fake"})
                )
                self.backend.update_step(
                    step=corrupt_step,
                    path=resolver_single_testkit.resolver.path,
                )

    def test_insert_resolver_rejects_non_model_input(self) -> None:
        """Resolver inputs must be model steps."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            invalid_resolver_path = StepPath(
                collection=dag_testkit.dag.name,
                run=dag_testkit.dag.run,
                name="resolver_invalid_parent",
            )
            invalid_resolver_step = Step(
                description="Invalid resolver parent",
                step_type="resolver",
                config={
                    "resolver_class": "Components",
                    "inputs": ("crn",),
                    "resolver_settings": {},
                },
                fingerprint=b"invalid_parent",
            )

            with pytest.raises(
                MatchboxStepTypeError,
                match="depend on model",
            ):
                self.backend.create_step(
                    step=invalid_resolver_step,
                    path=invalid_resolver_path,
                )

    def test_model_results_basic(self) -> None:
        """Test that a model's results data can be set and retrieved."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")
            naive_crn_resolver_path = dag_testkit.resolvers[
                f"resolver_{naive_crn_testkit.name}"
            ].resolver.path

            # Query returns the same results as the testkit, showing
            # that processing was performed accurately.
            # (that we can query from it implies the step was correctly
            # marked as complete)
            res = self.backend.query(
                source=crn_testkit.path,
                resolver=naive_crn_resolver_path,
            )
            res_clusters = query_to_cluster_entities(
                data=res,
                keys={crn_testkit.name: "key"},
            )

            identical, report = diff_entities(
                expected=naive_crn_testkit.entities,
                actual=res_clusters,
            )

            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_data(path=naive_crn_testkit.path)

            assert isinstance(pre_results, pa.Table)
            assert len(pre_results) > 0

            # Validate IDs
            self.backend.validate_ids(ids=pre_results["left_id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["right_id"].to_pylist())

            # Cannot set new results
            with pytest.raises(MatchboxStepExistingData):
                self.backend.insert_model_data(
                    path=naive_crn_testkit.path,
                    results=naive_crn_testkit.scores.to_arrow(),
                )

            # Retrieve again
            post_results = self.backend.get_model_data(path=naive_crn_testkit.path)

            # Check difference
            assert pre_results == post_results

    def test_model_results_scored(self) -> None:
        """Test that a scored model's results data can be set and retrieved."""
        with self.scenario(self.backend, "scored_dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            score_crn_testkit = dag_testkit.models.get("scored_test_crn")
            score_crn_resolver_path = dag_testkit.resolvers[
                f"resolver_{score_crn_testkit.name}"
            ].resolver.path

            # Query returns the same results as the testkit, showing
            # that processing was performed accurately
            res = self.backend.query(
                source=crn_testkit.path,
                resolver=score_crn_resolver_path,
            )
            res_clusters = query_to_cluster_entities(
                data=res,
                keys={crn_testkit.name: "key"},
            )

            identical, report = diff_entities(
                expected=score_crn_testkit.entities,
                actual=res_clusters,
            )
            assert identical, report

            # Retrieve
            pre_results = self.backend.get_model_data(path=score_crn_testkit.path)

            assert isinstance(pre_results, pa.Table)
            assert len(pre_results) > 0

            # Validate IDs
            self.backend.validate_ids(ids=pre_results["left_id"].to_pylist())
            self.backend.validate_ids(ids=pre_results["right_id"].to_pylist())

    def test_model_results_empty(self) -> None:
        """Can insert and retrieve empty model results"""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            linked = dag_testkit.source_to_linked["crn"]
            source_query = self.backend.query(source=crn_testkit.path)
            model_testkit = query_to_model_factory(
                left_query=Query(crn_testkit.source, dag=dag_testkit.dag),
                left_data=source_query,
                left_keys={crn_testkit.name: "key"},
                true_entities=tuple(linked.true_entities),
                name="empty_test_crn",
                description="Empty dedupe test model",
                score_range=(1.0, 1.0),
            )
            model_testkit.fake_run()
            assert model_testkit.model.results is not None
            model_testkit.model.results = model_testkit.model.results.head(0)

            self.backend.create_step(
                model_testkit.model.to_dto(), path=model_testkit.path
            )

            self.backend.insert_model_data(
                path=model_testkit.model.path,
                results=model_testkit.model.results.to_arrow(),
            )

            resolver_testkit = resolver_factory(
                dag=dag_testkit.dag,
                name=f"resolver_{model_testkit.name}",
                inputs=[model_testkit],
            )
            resolver_testkit.fake_run()
            assert resolver_testkit.resolver.results is not None
            resolver_testkit.resolver.results = resolver_testkit.resolver.results.head(
                0
            )
            self.backend.create_step(
                step=resolver_testkit.resolver.to_dto(),
                path=resolver_testkit.resolver.path,
            )
            self.backend.insert_resolver_data(
                path=resolver_testkit.resolver.path,
                data=resolver_testkit.resolver.results.to_arrow(),
            )

            # Querying from deduper with no results is the same as querying from source
            # (That we can query also implies that step marked as complete)
            dedupe_query = self.backend.query(
                source=crn_testkit.path,
                resolver=resolver_testkit.resolver.path,
            )

            source_entities = query_to_cluster_entities(
                data=source_query,
                keys={crn_testkit.name: "key"},
            )
            dedupe_entities = query_to_cluster_entities(
                data=dedupe_query,
                keys={crn_testkit.name: "key"},
            )
            identical, report = diff_entities(
                expected=list(source_entities),
                actual=list(dedupe_entities),
            )
            assert identical, report

    def test_model_results_shared_clusters(self) -> None:
        """Test that model results data can be inserted when clusters are shared."""
        with self.scenario(self.backend, "convergent_partial") as dag_testkit:
            for model_testkit in dag_testkit.models.values():
                self.backend.create_step(
                    path=model_testkit.path,
                    step=model_testkit.fake_run().model.to_dto(),
                )
                self.backend.insert_model_data(
                    path=model_testkit.path,
                    results=model_testkit.scores.to_arrow(),
                )
