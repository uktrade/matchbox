import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyarrow import Table

from matchbox.client.results import Results


def test_clusters_and_root_leaf():
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

    probabilities = Table.from_pylist(
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
        left_root_leaf=left_root_leaf.to_arrow(),
        right_root_leaf=right_root_leaf.to_arrow(),
    )

    # Check two ways of representing clusters
    clusters = pl.from_arrow(results.clusters)
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
        probabilities=Table.from_pydict(
            {"left_id": [], "right_id": [], "probability": []}
        ),
        left_root_leaf=left_root_leaf.to_arrow(),
        right_root_leaf=right_root_leaf.to_arrow(),
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
