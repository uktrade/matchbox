"""Test the backend adapter's admin functions."""

from functools import partial

import pytest
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.exceptions import MatchboxDataNotFound
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.docker
class TestMatchboxAdminBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

    # User management

    def test_login(self) -> None:
        """Can swap user name with user ID."""
        with self.scenario(self.backend, "bare") as _:
            alice_id = self.backend.login("alice")
            assert alice_id == self.backend.login("alice")
            assert alice_id != self.backend.login("bob")

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
            assert self.backend.source_clusters.count() > 0
            assert self.backend.models.count() > 0
            assert self.backend.model_clusters.count() > 0
            assert self.backend.creates.count() > 0
            assert self.backend.merges.count() > 0
            assert self.backend.proposes.count() > 0

            self.backend.clear(certain=True)

            assert self.backend.sources.count() == 0
            assert self.backend.source_clusters.count() == 0
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
                self.backend.source_clusters.count,
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

    def test_delete_orphans(self) -> None:
        """Can delete orphaned clusters."""
        with self.scenario(self.backend, "link") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            # Get number of clusters
            initial_all_clusters = self.backend.all_clusters.count()

            # Delete orphans, none should be deleted yet
            orphans = self.backend.delete_orphans()
            assert orphans == 0
            assert initial_all_clusters == self.backend.all_clusters.count()

            # TODO: insert judgement for cluster, check that it is not deleted when
            # deleting model resolution. Then deleting the judgement should cause
            # exactly 1 orphan.

            model_res = naive_crn_testkit.resolution_path
            self.backend.delete_resolution(model_res, certain=True)

            # Delete orphans, some should be deleted and total clusters should reduce
            orphans = self.backend.delete_orphans()
            assert orphans > 0
            all_clusters_2 = self.backend.all_clusters.count()
            assert initial_all_clusters > all_clusters_2

            # Delete source resolution crn
            source_res = crn_testkit.resolution_path
            self.backend.delete_resolution(source_res, certain=True)

            # Delete orphans again and check number of clusters has reduced
            orphans = self.backend.delete_orphans()
            assert orphans > 0
            all_clusters_3 = self.backend.all_clusters.count()
            assert all_clusters_2 > all_clusters_3
