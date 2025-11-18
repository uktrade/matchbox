import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from matchbox.client.results import ResolvedData, Results
from matchbox.common.arrow import SCHEMA_QUERY_WITH_LEAVES
from matchbox.common.factories.sources import source_factory


def test_clusters_and_root_leaf() -> None:
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
            # simple left-right merge
            {"left_id": 4, "right_id": 5, "probability": 100},
            # dedupe through linking
            {"left_id": 10, "right_id": 6, "probability": 100},
            {"left_id": 10, "right_id": 7, "probability": 100},
        ]
    )

    results = Results(
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
    # Only the input IDs referenced in probabilities are present, in the right groups
    assert grouped_children == {(4, 5), (6, 7, 10)}

    root_leaf = results.root_leaf()
    grouped_leaves = {
        tuple(sorted(group))
        for group in root_leaf.group_by("root_id").agg("leaf_id")["leaf_id"]
    }
    # Only single-digits are present, and all of them, in the right groups
    assert grouped_leaves == {(1, 6, 7), (2, 3), (4, 5), (8,)}

    # To check edge cases, look at no probabilities returned
    empty_results = Results(
        probabilities=pl.DataFrame({"left_id": [], "right_id": [], "probability": []}),
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
    only_prob_results = Results(probabilities=probabilities)
    with pytest.raises(RuntimeError, match="instantiated for validation"):
        only_prob_results.root_leaf()


def test_get_lookup() -> None:
    """Lookup can be generated from resolved data."""
    foo = source_factory(name="foo").source
    bar = source_factory(name="bar").source

    foo_query_data = pa.Table.from_pylist(
        [
            {"id": 14, "leaf_id": 1, "key": "1"},
            {"id": 2, "leaf_id": 2, "key": "2"},
            {"id": 356, "leaf_id": 3, "key": "3"},
        ],
        schema=SCHEMA_QUERY_WITH_LEAVES,
    )

    bar_query_data = pa.Table.from_pylist(
        [
            {"id": 14, "leaf_id": 4, "key": "a"},
            {"id": 356, "leaf_id": 5, "key": "b"},
            {"id": 356, "leaf_id": 6, "key": "c"},
        ],
        schema=SCHEMA_QUERY_WITH_LEAVES,
    )

    # Because of FULL OUTER JOIN, we expect some values to be null, and some explosions
    expected_foo_bar_mapping = pl.DataFrame(
        [
            {"id": 14, "foo_key": "1", "bar_key": "a"},
            {"id": 2, "foo_key": "2", "bar_key": None},
            {"id": 356, "foo_key": "3", "bar_key": "b"},
            {"id": 356, "foo_key": "3", "bar_key": "c"},
        ]
    )

    # When selecting single source, we won't explode
    expected_foo_mapping = expected_foo_bar_mapping.select(["id", "foo_key"]).unique()

    # Retrieve single table
    foo_mapping = ResolvedData(
        sources=[foo], query_results=[foo_query_data]
    ).get_lookup()

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # Retrieve multiple tables
    foo_bar_mapping = ResolvedData(
        sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
    ).get_lookup()

    assert_frame_equal(
        pl.from_arrow(foo_bar_mapping),
        expected_foo_bar_mapping,
        check_row_order=False,
        check_column_order=False,
    )
