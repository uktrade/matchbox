import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.results import ModelResults, ResolvedMatches
from matchbox.client.sources import Source
from matchbox.common.arrow import SCHEMA_QUERY_WITH_LEAVES
from matchbox.common.factories.sources import source_factory, source_from_tuple


class TestModelResults:
    """Test ModelResult objects."""

    def test_duplicate_removal(self) -> None:
        """Removes redundant pairs, keeping lowest probability."""
        simple_duplicate = pl.DataFrame(
            [
                {"left_id": 4, "right_id": 5, "probability": 50},
                {"left_id": 4, "right_id": 5, "probability": 100},
            ]
        )

        assert_frame_equal(
            ModelResults(probabilities=simple_duplicate).probabilities,
            simple_duplicate.tail(1),
            check_row_order=False,
            check_column_order=False,
            check_dtypes=False,
        )

        symmetric_duplicate = pl.DataFrame(
            [
                {"left_id": 5, "right_id": 4, "probability": 50},
                {"left_id": 4, "right_id": 5, "probability": 100},
            ]
        )

        assert_frame_equal(
            ModelResults(probabilities=symmetric_duplicate).probabilities,
            symmetric_duplicate.tail(1),
            check_row_order=False,
            check_column_order=False,
            check_dtypes=False,
        )

        no_duplicates = pl.DataFrame(
            [
                {"left_id": 4, "right_id": 6, "probability": 50},
                {"left_id": 4, "right_id": 5, "probability": 100},
            ]
        )

        assert_frame_equal(
            ModelResults(probabilities=no_duplicates).probabilities,
            no_duplicates,
            check_row_order=False,
            check_column_order=False,
            check_dtypes=False,
        )

    def test_clusters_and_root_leaf(self) -> None:
        """From a results object, we can derive clusters at various levels."""
        # Prepare dummy data and model
        left_root_leaf = pl.DataFrame(
            [
                # Two keys per root
                {"id": 10, "leaf_id": 1},
                {"id": 10, "leaf_id": 1},
                # Two leaves per root (same representation)
                {"id": 20, "leaf_id": 2},
                {"id": 20, "leaf_id": 3},
                # Singleton cluster with two keys
                {"id": 4, "leaf_id": 4},
                {"id": 4, "leaf_id": 4},
            ]
        )

        right_root_leaf = pl.DataFrame(
            # For simplicity, all these are singleton clusters
            [
                {"id": 5, "leaf_id": 5},
                {"id": 6, "leaf_id": 6},
                {"id": 7, "leaf_id": 7},
                {"id": 8, "leaf_id": 8},
            ]
        )

        probabilities = pl.DataFrame(
            [
                # Simple left-right merge
                {"left_id": 4, "right_id": 5, "probability": 100},
                # Dedupe through linking
                {"left_id": 10, "right_id": 6, "probability": 100},
                {"left_id": 10, "right_id": 7, "probability": 100},
            ]
        )

        results = ModelResults(
            probabilities=probabilities,
            left_root_leaf=left_root_leaf,
            right_root_leaf=right_root_leaf,
        )

        # Check two ways of representing clusters
        clusters = results.clusters
        grouped_children = {
            tuple(sorted(group))
            for group in clusters.group_by("parent").agg("child")["child"]
        }
        # Only input IDs referenced in probabilities are present, in the right groups
        assert grouped_children == {(4, 5), (6, 7, 10)}

        root_leaf = results.root_leaf()
        grouped_leaves = {
            tuple(sorted(group))
            for group in root_leaf.group_by("root_id").agg("leaf_id")["leaf_id"]
        }
        # Only single-digits are present, and all of them, in the right groups
        assert grouped_leaves == {(1, 6, 7), (2, 3), (4, 5), (8,)}

        # To check edge cases, look at no probabilities returned
        empty_results = ModelResults(
            probabilities=pl.DataFrame(
                {"left_id": [], "right_id": [], "probability": []}
            ),
            left_root_leaf=left_root_leaf,
            right_root_leaf=right_root_leaf,
        )

        assert len(empty_results.clusters) == 0
        expected_empty_root_leaf = pl.concat(
            [
                left_root_leaf.rename({"id": "root_id"}),
                right_root_leaf.rename({"id": "root_id"}),
            ]
        ).unique()
        assert_frame_equal(
            empty_results.root_leaf(),
            expected_empty_root_leaf,
            check_column_order=False,
            check_row_order=False,
        )

        # The above was only possible because leaf IDs were present in the inputs
        only_prob_results = ModelResults(probabilities=probabilities)
        with pytest.raises(RuntimeError, match="instantiated for validation"):
            only_prob_results.root_leaf()


class TestResolvedMatches:
    """Test ResolvedMatches objects."""

    @pytest.fixture(scope="function")  # warehouse is function-scoped
    def dummy_data(
        self, sqlite_in_memory_warehouse: Engine
    ) -> tuple[Source, Source, pl.DataFrame, pl.DataFrame]:
        """Create foo, bar and associated matches."""
        foo = (
            source_from_tuple(
                name="foo",
                engine=sqlite_in_memory_warehouse,
                data_keys=["1", "2", "2b", "3"],
                data_tuple=(
                    {"field_a": 10},
                    {"field_a": 20},
                    {"field_a": 20},
                    {"field_a": 30},
                ),
            )
            .write_to_location()
            .source
        )
        bar = (
            source_from_tuple(
                name="bar",
                engine=sqlite_in_memory_warehouse,
                data_keys=["a", "b", "c", "d"],
                data_tuple=(
                    {"field_a": "1x", "field_b": "1y"},
                    {"field_a": "2x", "field_b": "2y"},
                    {"field_a": "3x", "field_b": "3y"},
                    {"field_a": "4x", "field_b": "4y"},
                ),
            )
            .write_to_location()
            .source
        )

        # Both foo and bar have a record that's not linked
        # Foo has two keys for one leaf ID
        # Foo and bar have links across; bar also has link within
        foo_query_data = pl.DataFrame(
            [
                {"id": 14, "leaf_id": 1, "key": "1"},
                {"id": 2, "leaf_id": 2, "key": "2"},
                {"id": 2, "leaf_id": 2, "key": "2b"},
                {"id": 356, "leaf_id": 3, "key": "3"},
            ],
            schema=pl.Schema(SCHEMA_QUERY_WITH_LEAVES),
        )

        bar_query_data = pl.DataFrame(
            [
                {"id": 14, "leaf_id": 4, "key": "a"},
                {"id": 356, "leaf_id": 5, "key": "b"},
                {"id": 356, "leaf_id": 6, "key": "c"},
                {"id": 7, "leaf_id": 7, "key": "d"},
            ],
            schema=pl.Schema(SCHEMA_QUERY_WITH_LEAVES),
        )

        return foo, bar, foo_query_data, bar_query_data

    def test_from_dump(
        self, dummy_data: tuple[Source, Source, pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Can initialise ResolvedMatches from concatenated dataframe representation."""
        foo, bar, foo_query_data, bar_query_data = dummy_data
        # These won't be the same sources as above, but we only need them them
        # to have the same name
        dag = DAG("companies")
        dag.source(**source_factory(name="foo").into_dag())
        dag.source(**source_factory(name="bar").into_dag())

        original = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        )

        new = ResolvedMatches.from_dump(cluster_key_map=original.as_dump(), dag=dag)

        assert new.sources[0] == dag.get_source("foo")
        assert new.sources[1] == dag.get_source("bar")

        assert_frame_equal(
            foo_query_data,
            new.query_results[0],
            check_row_order=False,
            check_column_order=False,
        )

        assert_frame_equal(
            bar_query_data,
            new.query_results[1],
            check_row_order=False,
            check_column_order=False,
        )

    def test_as_lookup(
        self, dummy_data: tuple[Source, Source, pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Lookup can be generated from resolved data."""
        foo, bar, foo_query_data, bar_query_data = dummy_data

        # Because of FULL OUTER JOIN, we expect some nulls, and some explosions
        expected_foo_bar_mapping = pl.DataFrame(
            [
                {"id": 14, "foo_key": "1", "bar_key": "a"},
                {"id": 2, "foo_key": "2", "bar_key": None},
                {"id": 2, "foo_key": "2b", "bar_key": None},
                {"id": 356, "foo_key": "3", "bar_key": "b"},
                {"id": 356, "foo_key": "3", "bar_key": "c"},
                {"id": 7, "foo_key": None, "bar_key": "d"},
            ]
        )

        # When selecting single source, we won't explode
        expected_foo_mapping = pl.DataFrame(
            [
                {"id": 14, "foo_key": "1"},
                {"id": 2, "foo_key": "2"},
                {"id": 2, "foo_key": "2b"},
                {"id": 356, "foo_key": "3"},
            ]
        )

        # Retrieve single table
        foo_mapping = ResolvedMatches(
            sources=[foo], query_results=[foo_query_data]
        ).as_lookup()

        assert_frame_equal(
            foo_mapping,
            expected_foo_mapping,
            check_row_order=False,
            check_column_order=False,
        )

        # Retrieve multiple tables
        foo_bar_mapping = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        ).as_lookup()

        assert_frame_equal(
            foo_bar_mapping,
            expected_foo_bar_mapping,
            check_row_order=False,
            check_column_order=False,
        )

    def test_as_dump(
        self, dummy_data: tuple[Source, Source, pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Mapping across root, leaf, source and key can be generated."""
        foo, bar, foo_query_data, bar_query_data = dummy_data

        mapping = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        ).as_dump()

        expected_mapping = pl.DataFrame(
            [
                {"source": "foo", "id": 14, "leaf_id": 1, "key": "1"},
                {"source": "foo", "id": 2, "leaf_id": 2, "key": "2"},
                {"source": "foo", "id": 2, "leaf_id": 2, "key": "2b"},
                {"source": "foo", "id": 356, "leaf_id": 3, "key": "3"},
                {"source": "bar", "id": 14, "leaf_id": 4, "key": "a"},
                {"source": "bar", "id": 356, "leaf_id": 5, "key": "b"},
                {"source": "bar", "id": 356, "leaf_id": 6, "key": "c"},
                {"source": "bar", "id": 7, "leaf_id": 7, "key": "d"},
            ]
        )

        assert_frame_equal(
            mapping, expected_mapping, check_row_order=False, check_column_order=False
        )

    def test_as_leaf_sets(
        self, dummy_data: tuple[Source, Source, pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Can generate grouping of lead IDs."""
        foo, bar, foo_query_data, bar_query_data = dummy_data
        leaf_sets = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        ).as_leaf_sets()

        assert sorted(leaf_sets) == sorted([[1, 4], [2], [3, 5, 6], [7]])

    def test_view_cluster(
        self, dummy_data: tuple[Source, Source, pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Single cluster can be viewed with source data."""
        foo, bar, foo_query_data, bar_query_data = dummy_data

        cluster = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        ).view_cluster(356)

        # Expanded representation
        expected_cluster = pl.DataFrame(
            {
                "foo_key": ["3", None, None],
                "foo_field_a": [30, None, None],
                "bar_key": [None, "b", "c"],
                "bar_field_a": [None, "2x", "3x"],
                "bar_field_b": [None, "2y", "3y"],
            }
        )

        assert_frame_equal(
            cluster, expected_cluster, check_row_order=False, check_column_order=False
        )

        # Compact representation
        cluster_merged = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        ).view_cluster(356, merge_fields=True)

        # Note: 30 gets cast to a string
        expected_cluster_merged = pl.DataFrame(
            [
                {"foo_key": "3", "bar_key": None, "field_a": "30", "field_b": None},
                {"foo_key": None, "bar_key": "b", "field_a": "2x", "field_b": "2y"},
                {"foo_key": None, "bar_key": "c", "field_a": "3x", "field_b": "3y"},
            ]
        )

        assert_frame_equal(
            cluster_merged,
            expected_cluster_merged,
            check_row_order=False,
            check_column_order=False,
        )

        # View cluster with multiple keys per leaf, but only one source
        cluster_convergent = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        ).view_cluster(2)
        expected_cluster_convergent = pl.DataFrame(
            [
                # No columns from "bar"
                {"foo_key": "2", "foo_field_a": 20},
                {"foo_key": "2b", "foo_field_a": 20},
            ]
        )

        assert_frame_equal(
            cluster_convergent,
            expected_cluster_convergent,
            check_row_order=False,
            check_column_order=False,
        )

    def test_merge(
        self, dummy_data: tuple[Source, Source, pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Can merge two instances of resolved matches."""
        # Define first resolved matches
        foo, bar, foo_query_data, bar_query_data = dummy_data

        resolved_one = ResolvedMatches(
            sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
        )

        # Define alternative resolved matches
        alt_foo_query_data = pl.DataFrame(
            [
                {"id": 12, "leaf_id": 1, "key": "1"},
                {"id": 12, "leaf_id": 2, "key": "2"},
                {"id": 12, "leaf_id": 2, "key": "2b"},
                {"id": 35, "leaf_id": 3, "key": "3"},
            ],
            schema=pl.Schema(SCHEMA_QUERY_WITH_LEAVES),
        )

        alt_bar_query_data = pl.DataFrame(
            [
                {"id": 4, "leaf_id": 4, "key": "a"},
                {"id": 35, "leaf_id": 5, "key": "b"},
                {"id": 6, "leaf_id": 6, "key": "c"},
            ],
            schema=pl.Schema(SCHEMA_QUERY_WITH_LEAVES),
        )

        resolved_two = ResolvedMatches(
            sources=[foo, bar], query_results=[alt_foo_query_data, alt_bar_query_data]
        )

        # Merge the two
        merged_resolved = resolved_one.merge(resolved_two)
        new_clusters = merged_resolved.as_dump().group_by("id").agg("leaf_id")
        # The sources are unchanged
        assert merged_resolved.sources == [foo, bar]
        # All cluster IDs now have negative integers
        assert (new_clusters["id"] < 0).all()
        # New clusters contain leaves we expect
        leaf_sets = []
        for ls in new_clusters["leaf_id"].to_list():
            # We remove duplicates created by multiple keys for same leaf
            leaf_sets.append(sorted(set(ls)))

        assert sorted(leaf_sets) == [[1, 2, 4], [3, 5, 6], [7]]
