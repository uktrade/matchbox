"""Test the backend adapter's evaluation functions."""

from functools import partial

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES,
    SCHEMA_JUDGEMENTS,
)
from matchbox.common.dtos import ResolutionPath, User
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.docker
class TestMatchboxEvaluationBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

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

            alice: User = self.backend.login(User(sub="alice"))

            original_cluster_num = self.backend.model_clusters.count()

            # Can endorse the same cluster that is shown
            clust1_leaves = get_leaf_ids(unique_ids[0])
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_name=alice.user_name,
                    shown=clust1_leaves,
                    endorsed=[clust1_leaves],
                ),
            )
            # Can send redundant data
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_name=alice.user_name,
                    shown=clust1_leaves,
                    endorsed=[clust1_leaves],
                ),
            )
            assert self.backend.model_clusters.count() == original_cluster_num

            # Can tag judgement
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_name=alice.user_name,
                    shown=clust1_leaves,
                    endorsed=[clust1_leaves],
                    tag="eval_session1",
                ),
            )

            # Now split a cluster
            clust2_leaves = get_leaf_ids(unique_ids[1])
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_name=alice.user_name,
                    shown=clust2_leaves,
                    endorsed=[clust2_leaves[:1], clust2_leaves[1:]],
                ),
            )
            # Now, two new clusters should have been created
            assert self.backend.model_clusters.count() == original_cluster_num + 2

            # Insert a judgement with a "shown" that doesn't exist on the server
            self.backend.insert_judgement(
                judgement=Judgement(
                    user_name=alice.user_name,
                    shown=clust1_leaves + clust2_leaves,
                    endorsed=[clust1_leaves + clust2_leaves],
                ),
            )
            # A single extra cluster is created
            assert self.backend.model_clusters.count() == original_cluster_num + 3

            # Let's check failures
            # First, confirm that the following leaves don't exist
            fake_leaves = [10000, 10001]
            with pytest.raises(MatchboxDataNotFound):
                self.backend.validate_ids(fake_leaves)
            # Now, let's test an exception is raised
            with pytest.raises(MatchboxDataNotFound):
                self.backend.insert_judgement(
                    judgement=Judgement(
                        user_name=alice.user_name,
                        shown=fake_leaves,
                        endorsed=[fake_leaves],
                    ),
                )

            # Now, let's try to get the judgements back
            # Data gets back in the right shape
            judgements, expansion = self.backend.get_judgements()
            assert judgements.schema.equals(SCHEMA_JUDGEMENTS)
            assert expansion.schema.equals(SCHEMA_CLUSTER_EXPANSION)
            # Only one user ID was used
            assert judgements["user_name"].unique().to_pylist() == [alice.user_name]
            # The set of shown judgements has the two IDs we know, plus a new one
            assert set(judgements["shown"].to_pylist()) > {unique_ids[0], unique_ids[1]}

            # On the other hand, the root-leaf mapping table has no duplicates
            # 2 shown clusters + 2 new endorsed clusters + 1 new shown cluster
            assert len(expansion) == 5

            # We can use tag to filter judgements
            judgements_tag, expansion_tag = self.backend.get_judgements("eval_session1")
            assert judgements_tag.num_rows == 1
            assert expansion_tag.num_rows == 1

            # We now ensure the expansion has all we need
            # We get expansion for two known shown cluster IDs
            expansion_df = pl.from_arrow(expansion)
            assert sorted(
                expansion_df.filter(pl.col("root") == unique_ids[0])
                .select("leaves")
                .to_series()
                .to_list()[0]
            ) == sorted(clust1_leaves)

            assert sorted(
                expansion_df.filter(pl.col("root") == unique_ids[1])
                .select("leaves")
                .to_series()
                .to_list()[0]
            ) == sorted(clust2_leaves)

            # We get expansions of new clusters whose IDs are unknown
            expansion_leaf_sets = [
                sorted(ls[0]) for ls in pl.from_arrow(expansion).select("leaves").rows()
            ]
            assert sorted(clust2_leaves[:1]) in expansion_leaf_sets
            assert sorted(clust2_leaves[1:]) in expansion_leaf_sets
            assert sorted(clust1_leaves + clust2_leaves) in expansion_leaf_sets

    def test_sample_for_eval(self) -> None:
        """Can extract samples for a user and a resolution."""

        # Missing resolution raises error
        with (
            self.scenario(self.backend, "bare"),
            pytest.raises(MatchboxResolutionNotFoundError, match="naive_test_crn"),
        ):
            alice: User = self.backend.login(User(sub="alice"))
            self.backend.sample_for_eval(
                n=10,
                path=ResolutionPath(
                    collection="collection", run=1, name="naive_test_crn"
                ),
                user_name=alice.user_name,
            )

        # Convergent scenario allows testing we don't accidentally return metadata
        # for sources that aren't relevant for a point of truth
        with self.scenario(self.backend, "convergent") as dag_testkit:
            source_testkit = dag_testkit.sources.get("foo_a")
            model_testkit = dag_testkit.models.get("naive_test_foo_a")

            alice: User = self.backend.login(User(sub="alice"))

            # Source clusters should not be returned
            # So if we sample from a source resolution, we get nothing
            samples_source = self.backend.sample_for_eval(
                n=10, path=source_testkit.resolution_path, user_name=alice.user_name
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
                n=99, path=model_testkit.resolution_path, user_name=alice.user_name
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
                n=5, path=model_testkit.resolution_path, user_name=alice.user_name
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
                    user_name=alice.user_name,
                    shown=first_cluster_leaves,
                    endorsed=[first_cluster_leaves],
                ),
            )

            samples_without_cluster = self.backend.sample_for_eval(
                n=99, path=model_testkit.resolution_path, user_name=alice.user_name
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
                        user_name=alice.user_name,
                        shown=cluster_leaves,
                        endorsed=[cluster_leaves],
                    ),
                )

            samples_all_done = self.backend.sample_for_eval(
                n=99, path=model_testkit.resolution_path, user_name=alice.user_name
            )
            assert len(samples_all_done) == 0
