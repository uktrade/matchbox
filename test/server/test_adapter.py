from typing import Callable

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame

from matchbox.client.helpers.selector import match, query, select
from matchbox.client.results import Results
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import HASH_FUNC
from matchbox.common.sources import Match, Source, SourceAddress, SourceColumn
from matchbox.server.base import MatchboxDBAdapter, MatchboxModelAdapter

from ..fixtures.db import SetupDatabaseCallable
from ..fixtures.models import (
    dedupe_data_test_params,
    link_data_test_params,
)

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


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
class TestMatchboxBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        request: pytest.FixtureRequest,
        backend_instance: str,
        setup_database: Callable,
        warehouse_data: list[Source],
    ):
        self.backend: MatchboxDBAdapter = backend_instance
        self.setup_database: SetupDatabaseCallable = lambda level: setup_database(
            self.backend, warehouse_data, level
        )
        self.warehouse_data: list[Source] = warehouse_data

    def test_properties(self):
        """Test that properties obey their protocol restrictions."""
        self.setup_database("index")
        assert isinstance(self.backend.datasets.list(), list)
        assert isinstance(self.backend.datasets.count(), int)
        assert isinstance(self.backend.models.count(), int)
        assert isinstance(self.backend.data.count(), int)
        assert isinstance(self.backend.clusters.count(), int)
        assert isinstance(self.backend.creates.count(), int)
        assert isinstance(self.backend.merges.count(), int)
        assert isinstance(self.backend.proposes.count(), int)

    def test_model_properties(self):
        """Test that model properties obey their protocol restrictions."""
        self.setup_database("dedupe")
        naive_crn = self.backend.get_model(model="naive_test.crn")
        assert naive_crn.id
        assert naive_crn.hash
        assert naive_crn.name
        assert naive_crn.results
        assert isinstance(naive_crn.truth, float)  # otherwise we assert 0.0
        assert naive_crn.ancestors
        assert isinstance(naive_crn.ancestors_cache, dict)  # otherwise we assert {}

    def test_validate_ids(self):
        """Test validating data IDs."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]

        select_crn = select(
            {crn.address.full_name: ["company_name", "crn"]},
            engine=crn.engine,
        )

        df_crn = query(
            select_crn,
            resolution_name="naive_test.crn",
            return_type="pandas",
        )

        ids = df_crn.id.to_list()
        assert len(ids) > 0
        self.backend.validate_ids(ids=ids)

        with pytest.raises(MatchboxDataNotFound):
            self.backend.validate_ids(ids=[-6])

    def test_validate_hashes(self):
        """Test validating data hashes."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]
        select_crn = select(
            {crn.address.full_name: ["company_name", "crn"]},
            engine=crn.engine,
        )

        df_crn = query(
            select_crn,
            resolution_name="naive_test.crn",
            return_type="pandas",
        )

        ids = df_crn.id.to_list()
        hashes = list(self.backend.cluster_id_to_hash(ids=ids).values())
        assert len(hashes) > 0
        self.backend.validate_hashes(hashes=hashes)

        with pytest.raises(MatchboxDataNotFound):
            self.backend.validate_hashes(hashes=[HASH_FUNC(b"nonexistent").digest()])

    def test_cluster_id_to_hash(self):
        """Test getting ID to Cluster hash lookup from the database."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]

        select_crn = select(
            {crn.address.full_name: ["company_name", "crn"]},
            engine=crn.engine,
        )

        df_crn = query(
            select_crn,
            resolution_name="naive_test.crn",
            return_type="pandas",
        )

        ids = df_crn.id.to_list()
        assert len(ids) > 0

        hashes = self.backend.cluster_id_to_hash(ids=ids)
        assert len(hashes) == len(set(ids))
        assert set(ids) == set(hashes.keys())
        assert all(isinstance(h, bytes) for h in hashes.values())

        assert self.backend.cluster_id_to_hash(ids=[-6]) == {-6: None}

    def test_get_source(self):
        """Test querying data from the database."""
        self.setup_database("index")

        crn = self.warehouse_data[0]

        crn_retrieved = self.backend.get_source(crn.address)
        # Equality between the two is False because one lacks the Engine
        assert crn.model_dump() == crn_retrieved.model_dump()

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

        self.setup_database("link")

        graph = self.backend.get_resolution_graph()
        # Nodes: 3 datasets, 3 dedupers, and 2 linkers
        # Edges: 1 per deduper, 2 per linker
        assert len(graph.nodes) == 8
        assert len(graph.edges) == 7

    def test_get_model(self):
        """Test getting a model from the database."""
        self.setup_database("dedupe")

        model = self.backend.get_model(model="naive_test.crn")
        assert isinstance(model, MatchboxModelAdapter)

        with pytest.raises(MatchboxResolutionNotFoundError):
            self.backend.get_model(model="nonexistant")

    def test_get_resolution_id(self):
        self.setup_database("dedupe")
        resolution_id = self.backend.get_resolution_id("naive_test.crn")
        assert isinstance(resolution_id, int)

        with pytest.raises(MatchboxResolutionNotFoundError):
            self.backend.get_resolution_id("nonexistent")

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
        self.setup_database("link")

        # Expect it to delete itself, its probabilities,
        # its parents, and their probabilities
        deduper_to_delete = "naive_test.crn"
        total_models = len(dedupe_data_test_params) + len(link_data_test_params)

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

        # Deletes deduper and parent linkers: 3 models gone
        assert models_post_delete == models_pre_delete - 3

        # Cluster, dedupe and link count unaffected
        assert cluster_count_post_delete == cluster_count_pre_delete
        assert actual_merges_post_delete == actual_merges_pre_delete

        # But count of propose and create edges has dropped
        assert cluster_assoc_count_post_delete < cluster_assoc_count_pre_delete
        assert proposed_merge_probs_post_delete < proposed_merge_probs_pre_delete

    def test_insert_model(self):
        """Test that models can be inserted."""
        self.setup_database("index")

        crn = self.warehouse_data[0]
        duns = self.warehouse_data[1]

        # Test deduper insertion
        models_count = self.backend.models.count()

        self.backend.insert_model(
            "dedupe_1", left=crn.alias, description="Test deduper 1"
        )
        self.backend.insert_model(
            "dedupe_2", left=duns.alias, description="Test deduper 1"
        )

        assert self.backend.models.count() == models_count + 2

        # Test linker insertion
        self.backend.insert_model(
            "link_1", left="dedupe_1", right="dedupe_2", description="Test linker 1"
        )

        assert self.backend.models.count() == models_count + 3

        # Test model upsert
        self.backend.insert_model(
            "link_1", left="dedupe_1", right="dedupe_2", description="Test upsert"
        )

        assert self.backend.models.count() == models_count + 3

    def test_model_results(self):
        """Test that a model's Results can be set and retrieved."""
        self.setup_database("dedupe")
        naive_crn = self.backend.get_model(model="naive_test.crn")

        # Retrieve
        pre_results = naive_crn.results

        assert isinstance(pre_results, Results)
        assert len(pre_results.probabilities) > 0
        assert pre_results.metadata.name == "naive_test.crn"

        self.backend.validate_ids(ids=pre_results.probabilities["id"].to_pylist())
        self.backend.validate_ids(ids=pre_results.probabilities["left_id"].to_pylist())
        self.backend.validate_ids(ids=pre_results.probabilities["right_id"].to_pylist())

        # Set
        target_row = pre_results.probabilities.to_pylist()[0]
        target_id = target_row["id"]
        target_left_id = target_row["left_id"]
        target_right_id = target_row["right_id"]

        matches_id_mask = pc.not_equal(pre_results.probabilities["id"], target_id)
        matches_left_mask = pc.not_equal(
            pre_results.probabilities["left_id"], target_left_id
        )
        matches_right_mask = pc.not_equal(
            pre_results.probabilities["right_id"], target_right_id
        )

        combined_mask = pc.and_(
            pc.and_(matches_id_mask, matches_left_mask), matches_right_mask
        )
        df_probabilities_truncated = pre_results.probabilities.filter(combined_mask)

        results = Results(
            probabilities=df_probabilities_truncated.select(
                ["left_id", "right_id", "probability"]
            ),
            model=pre_results.model,
            metadata=pre_results.metadata,
        )

        naive_crn.results = results

        # Retrieve again
        post_results = naive_crn.results

        # Check difference
        assert len(pre_results.probabilities) != len(post_results.probabilities)

        # Check similarity
        assert pre_results.metadata.name == post_results.metadata.name

    def test_model_truth(self):
        """Test that a model's truth can be set and retrieved."""
        self.setup_database("dedupe")
        naive_crn = self.backend.get_model(model="naive_test.crn")
        # Retrieve
        pre_truth = naive_crn.truth

        # Set
        naive_crn.truth = 0.5

        # Retrieve again
        post_truth = naive_crn.truth

        # Check difference
        assert pre_truth != post_truth

    def test_model_ancestors(self):
        """Test that a model's ancestors can be retrieved."""
        self.setup_database("link")
        linker_name = "deterministic_naive_test.crn_naive_test.duns"
        linker = self.backend.get_model(model=linker_name)

        assert isinstance(linker.ancestors, dict)

        truth_found = False
        for model, truth in linker.ancestors.items():
            if isinstance(truth, float):
                # Not all ancestors have truth values, but one must
                truth_found = True
            assert isinstance(model, str)
            assert isinstance(truth, (float, type(None)))

        assert truth_found

    def test_model_ancestors_cache(self):
        """Test that a model's ancestors cache can be set and retrieved."""
        self.setup_database("link")
        linker_name = "deterministic_naive_test.crn_naive_test.duns"
        linker = self.backend.get_model(model=linker_name)

        # Retrieve
        pre_ancestors_cache = linker.ancestors_cache

        # Set
        updated_ancestors_cache = {k: 0.5 for k in pre_ancestors_cache.keys()}
        linker.ancestors_cache = updated_ancestors_cache

        # Retrieve again
        post_ancestors_cache = linker.ancestors_cache

        # Check difference
        assert pre_ancestors_cache != post_ancestors_cache
        assert post_ancestors_cache == updated_ancestors_cache

    def test_index(
        self,
        crn_companies: DataFrame,
        duns_companies: DataFrame,
        cdms_companies: DataFrame,
    ):
        """Test that indexing data works."""
        assert self.backend.data.count() == 0

        self.setup_database("index")

        def count_deduplicates(df: DataFrame) -> int:
            return df.drop(columns=["id"]).drop_duplicates().shape[0]

        unique = sum(
            count_deduplicates(df)
            for df in [crn_companies, duns_companies, cdms_companies]
        )

        assert self.backend.data.count() == unique

    def test_index_new_source(self):
        address = SourceAddress(full_name="foo", warehouse_hash=b"bar")
        source = Source(
            address=address,
            db_pk="pk",
            columns=[SourceColumn(name="a", type="TEXT")],
        )
        data_hashes = pa.table(
            {
                "source_pk": pa.array([["1"], ["2"]]),
                "hash": pa.array([b"1", b"2"]),
            }
        )

        assert self.backend.clusters.count() == 0

        self.backend.index(source, data_hashes)

        assert self.backend.get_source(address) == source
        assert self.backend.get_resolution_id("foo")
        assert self.backend.data.count() == 2
        # I can add it again with no consequences
        self.backend.index(source, data_hashes)
        assert self.backend.data.count() == 2
        assert self.backend.source_resolutions.count() == 1

    def test_index_duplicate_clusters(self):
        address = SourceAddress(full_name="foo", warehouse_hash=b"bar")
        source = Source(
            address=address,
            db_pk="pk",
            columns=[SourceColumn(name="a", type="TEXT")],
        )
        data_hashes1 = pa.table(
            {
                "source_pk": pa.array([["1"], ["2"]]),
                "hash": pa.array([b"1", b"2"]),
            }
        )
        data_hashes2 = pa.table(
            {
                "source_pk": pa.array([["1"], ["2"], ["3"]]),
                "hash": pa.array([b"1", b"2", b"3"]),
            }
        )

        assert self.backend.data.count() == 0
        self.backend.index(source, data_hashes1)
        assert self.backend.data.count() == 2
        self.backend.index(source, data_hashes2)
        assert self.backend.data.count() == 3
        assert self.backend.source_resolutions.count() == 1

    def test_index_same_resolution(self):
        source1 = Source(
            address=SourceAddress(full_name="foo", warehouse_hash=b"bar"),
            db_pk="pk",
            columns=[SourceColumn(name="a", type="TEXT")],
        )
        source2 = Source(
            address=SourceAddress(full_name="foo", warehouse_hash=b"bar2"),
            db_pk="pk",
            columns=[SourceColumn(name="a", type="TEXT")],
        )
        data_hashes = pa.table(
            {
                "source_pk": pa.array([["1"], ["2"]]),
                "hash": pa.array([b"1", b"2"]),
            }
        )

        self.backend.index(source1, data_hashes)
        self.backend.index(source2, data_hashes)

        assert self.backend.data.count() == 2
        assert self.backend.source_resolutions.count() == 1

    def test_index_different_resolution_same_hashes(self):
        source1 = Source(
            address=SourceAddress(full_name="foo", warehouse_hash=b"bar"),
            db_pk="pk",
            columns=[SourceColumn(name="a", type="TEXT")],
        )
        source2 = Source(
            address=SourceAddress(full_name="foo2", warehouse_hash=b"bar2"),
            db_pk="pk",
            columns=[SourceColumn(name="a", type="TEXT")],
        )
        data_hashes = pa.table(
            {
                "source_pk": pa.array([["1"], ["2"]]),
                "hash": pa.array([b"1", b"2"]),
            }
        )

        self.backend.index(source1, data_hashes)
        # TODO: this will now error, and it shouldn't
        with pytest.raises(NotImplementedError):
            self.backend.index(source2, data_hashes)

    def test_select_warning(self):
        """Tests selecting non-indexed fields warns the user."""
        self.setup_database("index")

        crn = self.warehouse_data[0]

        with pytest.warns(Warning):
            select(
                {crn.address.full_name: ["id", "crn"]},
                engine=crn.engine,
            )

    def test_query_only_source(self):
        """Test querying data from a link point of truth."""
        self.setup_database("index")

        crn_wh = self.warehouse_data[0]

        select_crn = select(
            {crn_wh.address.full_name: ["crn"]},
            engine=crn_wh.engine,
        )

        df_crn_sample = query(
            select_crn,
            return_type="pandas",
            limit=10,
        )

        assert isinstance(df_crn_sample, DataFrame)
        assert df_crn_sample.shape[0] == 10

        df_crn_full = query(
            select_crn,
            return_type="pandas",
        )

        assert df_crn_full.shape[0] == 3000
        assert set(df_crn_full.columns) == {
            "id",
            "test_crn_crn",
        }

    def test_query_with_dedupe_model(self):
        """Test querying data from a deduplication point of truth."""
        self.setup_database("dedupe")

        crn = self.warehouse_data[0]

        select_crn = select(
            {crn.address.full_name: ["company_name", "crn"]},
            engine=crn.engine,
        )

        df_crn = query(
            select_crn,
            resolution_name="naive_test.crn",
            return_type="pandas",
        )

        assert isinstance(df_crn, DataFrame)
        assert df_crn.shape[0] == 3000
        assert set(df_crn.columns) == {
            "id",
            "test_crn_crn",
            "test_crn_company_name",
        }
        assert df_crn.id.nunique() == 1000

    def test_query_with_link_model(self):
        """Test querying data from a link point of truth."""
        self.setup_database("link")

        linker_name = "deterministic_naive_test.crn_naive_test.duns"

        crn_wh = self.warehouse_data[0]
        duns_wh = self.warehouse_data[1]

        select_crn = select(
            {crn_wh.address.full_name: ["crn"]},
            engine=crn_wh.engine,
        )

        select_duns = select(
            {duns_wh.address.full_name: ["duns"]},
            engine=duns_wh.engine,
        )

        crn_duns = query(
            select_crn,
            select_duns,
            resolution_name=linker_name,
            return_type="pandas",
        )

        assert isinstance(crn_duns, DataFrame)
        assert crn_duns.shape[0] == 3500
        assert set(crn_duns.columns) == {
            "id",
            "test_crn_crn",
            "test_duns_duns",
        }
        assert crn_duns["id"].nunique() == 1000

    def test_match_one_to_many(self, revolution_inc: dict[str, list[str]]):
        """Test that matching data works when the target has many IDs."""
        self.setup_database("link")

        crn_x_duns = "deterministic_naive_test.crn_naive_test.duns"
        crn_wh = self.warehouse_data[0]
        duns_wh = self.warehouse_data[1]

        res = match(
            backend=self.backend,
            source_pk=revolution_inc["duns"][0],
            source=duns_wh.address,
            target=crn_wh.address,
            resolution=crn_x_duns,
        )

        assert isinstance(res, Match)
        assert res.source == duns_wh.address
        assert res.target == crn_wh.address
        assert res.cluster is not None
        assert res.source_id == set(revolution_inc["duns"])
        assert res.target_id == set(revolution_inc["crn"])

    def test_match_many_to_one(self, revolution_inc: dict[str, list[str]]):
        """Test that matching data works when the source has more possible IDs."""
        self.setup_database("link")

        crn_x_duns = "deterministic_naive_test.crn_naive_test.duns"
        crn_wh = self.warehouse_data[0]
        duns_wh = self.warehouse_data[1]

        res = match(
            backend=self.backend,
            source_pk=revolution_inc["crn"][0],
            source=crn_wh.address,
            target=duns_wh.address,
            resolution=crn_x_duns,
        )

        assert isinstance(res, Match)
        assert res.source == crn_wh.address
        assert res.target == duns_wh.address
        assert res.cluster is not None
        assert res.source_id == set(revolution_inc["crn"])
        assert res.target_id == set(revolution_inc["duns"])

    def test_match_one_to_none(self, winner_inc: dict[str, list[str]]):
        """Test that matching data work when the target has no IDs."""
        self.setup_database("link")

        crn_x_duns = "deterministic_naive_test.crn_naive_test.duns"
        crn_wh = self.warehouse_data[0]
        duns_wh = self.warehouse_data[1]

        res = match(
            backend=self.backend,
            source_pk=winner_inc["crn"][0],
            source=crn_wh.address,
            target=duns_wh.address,
            resolution=crn_x_duns,
        )

        assert isinstance(res, Match)
        assert res.source == crn_wh.address
        assert res.target == duns_wh.address
        assert res.cluster is not None
        assert res.source_id == set(winner_inc["crn"])
        assert res.target_id == set() == set(winner_inc["duns"])

    def test_match_none_to_none(self):
        """Test that matching data work when the supplied key doesn't exist."""
        self.setup_database("link")

        crn_x_duns = "deterministic_naive_test.crn_naive_test.duns"
        crn_wh = self.warehouse_data[0]
        duns_wh = self.warehouse_data[1]

        res = match(
            backend=self.backend,
            source_pk="foo",
            source=crn_wh.address,
            target=duns_wh.address,
            resolution=crn_x_duns,
        )

        assert isinstance(res, Match)
        assert res.source == crn_wh.address
        assert res.target == duns_wh.address
        assert res.cluster is None
        assert res.source_id == set()
        assert res.target_id == set()

    def test_clear(self):
        """Test clearing the database."""
        self.setup_database("dedupe")

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
