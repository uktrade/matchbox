"""Test the backend adapter's evaluation functions."""

from functools import partial

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyarrow import Table
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES_DOWNLOAD,
    SCHEMA_EVAL_SAMPLES_UPLOAD,
    SCHEMA_JUDGEMENTS,
)
from matchbox.common.dtos import ResolutionPath
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
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

    def test_insert_and_get_samples(self) -> None:
        """Can insert sample sets for evaluation, and then retrieve them."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            # Find three existing model clusters
            crn_path = dag_testkit.sources["crn"].source.resolution_path
            crn_query = pl.from_arrow(self.backend.query(crn_path))
            crn_query_dict = {
                row["key"]: row["id"] for row in crn_query.iter_rows(named=True)
            }
            true_entities = dag_testkit.source_to_linked["crn"].true_entity_subset(
                "crn"
            )
            cluster1, cluster2, cluster3 = [
                list(true_entities[i].keys["crn"]) for i in range(3)
            ]

            # Sample existing cluster
            leaves1 = [crn_query_dict[leaf] for leaf in cluster1]
            samples_data = []
            for leaf in leaves1:
                samples_data.append({"root": -1, "leaf": leaf, "weight": 1})

            # Sample new cluster (merge of two other clusters)
            leaves2 = [crn_query_dict[leaf] for leaf in cluster2]
            leaves3 = [crn_query_dict[leaf] for leaf in cluster3]
            leaves23 = leaves2 + leaves3
            for leaf in leaves23:
                samples_data.append({"root": -2, "leaf": leaf, "weight": 1})

            # Can accept empty sample (without consequences)
            no_samples = Table.from_pylist([], schema=SCHEMA_EVAL_SAMPLES_UPLOAD)
            self.backend.insert_samples(
                samples=no_samples, name="sampleset", collection=dag_testkit.dag.name
            )

            # Insert sample data
            samples = Table.from_pylist(samples_data, schema=SCHEMA_EVAL_SAMPLES_UPLOAD)
            self.backend.insert_samples(
                samples=samples, name="sampleset", collection=dag_testkit.dag.name
            )

            # Cannot insert with the same name
            with pytest.raises(ValueError, match="name"):
                self.backend.insert_samples(
                    samples=samples, name="sampleset", collection=dag_testkit.dag.name
                )

            # But we can re-insert with a different name
            self.backend.insert_samples(
                samples=samples, name="sampleset2", collection=dag_testkit.dag.name
            )

            assert set(self.backend.list_sample_sets(dag_testkit.dag.name)) == {
                "sampleset",
                "sampleset2",
            }

    def test_delete_sample_set(self) -> None:
        """Can delete sample sets after inserting."""
        with self.scenario(self.backend, "index") as dag_testkit:
            crn_path = dag_testkit.sources["crn"].source.resolution_path
            crn_query = pl.from_arrow(self.backend.query(crn_path))["id"].to_list()
            samples = pl.DataFrame(
                [
                    {"root": -1, "leaf": crn_query[0], "weight": 1},
                    {"root": -1, "leaf": crn_query[1], "weight": 1},
                ]
            ).to_arrow()
            self.backend.insert_samples(
                samples=samples, name="sampleset", collection=dag_testkit.dag.name
            )

            assert self.backend.list_sample_sets(dag_testkit.dag.name) == ["sampleset"]

            with pytest.raises(MatchboxDeletionNotConfirmed):
                self.backend.delete_sample_set(
                    collection=dag_testkit.dag.name, name="sampleset"
                )

            assert self.backend.list_sample_sets(dag_testkit.dag.name) == ["sampleset"]

            self.backend.delete_sample_set(
                collection=dag_testkit.dag.name, name="sampleset", certain=True
            )

            assert self.backend.list_sample_sets(dag_testkit.dag.name) == []

    def test_insert_and_get_judgement(self) -> None:
        """Can insert and retrieve judgements."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            samples = Table.from_pylist(
                [{"root": -1, "leaf": 1}, {"root": -1, "leaf": 2}],
                schema=SCHEMA_EVAL_SAMPLES_UPLOAD,
            )
            self.backend.insert_samples(
                samples=samples, name="sampleset", collection=dag_testkit.dag.name
            )
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            # To begin with, no judgements to retrieve
            judgements, expansion = self.backend.get_judgements(
                collection=dag_testkit.dag.name, sample_set="sampleset"
            )
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
                collection=dag_testkit.dag.name,
                sample_set="sampleset",
                judgement=Judgement(
                    user_id=alice_id, shown=unique_ids[0], endorsed=[clust1_leaves]
                ),
            )
            # Can send redundant data
            self.backend.insert_judgement(
                collection=dag_testkit.dag.name,
                sample_set="sampleset",
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
                collection=dag_testkit.dag.name,
                sample_set="sampleset",
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
                    collection=dag_testkit.dag.name,
                    sample_set="sampleset",
                    judgement=Judgement(
                        user_id=alice_id, shown=unique_ids[0], endorsed=[fake_leaves]
                    ),
                )

            # Now, let's try to get the judgements back
            # Data gets back in the right shape
            judgements, expansion = self.backend.get_judgements(
                collection=dag_testkit.dag.name, sample_set="sampleset"
            )
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

            assert samples_99.schema.equals(SCHEMA_EVAL_SAMPLES_DOWNLOAD)

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
