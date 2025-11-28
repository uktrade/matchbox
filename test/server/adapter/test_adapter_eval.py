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
from matchbox.common.dtos import ResolutionPath
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxNoJudgements,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.docker
class TestMatchboxEvaluationBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

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
