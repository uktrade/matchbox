import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyarrow import Table

from matchbox.client.models.linkers.base import Linker
from matchbox.client.models.models import Model
from matchbox.client.results import Results
from matchbox.common.dtos import ModelConfig, ModelType


def test_clusters_and_root_leaf():
    """From a results object, we can derive clusters at various levels."""
    # Prepare dummy data and model
    left_data = pl.DataFrame(
        [
            # Two keys per root
            {"foo": "a", "id": 10, "leaf_id": 1, "key": "1"},
            {"foo": "a", "id": 10, "leaf_id": 1, "key": "1bis"},
            # Two leaves per root
            {"foo": "b", "id": 20, "leaf_id": 2, "key": "2"},
            {"foo": "c", "id": 20, "leaf_id": 3, "key": "3"},
            # Singleton cluster with two keys
            {"foo": "d", "id": 4, "leaf_id": 4, "key": "4"},
            {"foo": "d", "id": 4, "leaf_id": 4, "key": "4bis"},
        ]
    )

    right_data = pl.DataFrame(
        # For simplicity, all these are singleton clusters
        [
            {"foo": "d", "id": 5, "leaf_id": 5, "key": "5"},
            {"foo": "a", "id": 6, "leaf_id": 6, "key": "6"},
            {"foo": "a", "id": 7, "leaf_id": 7, "key": "7"},
            {"foo": "b", "id": 8, "leaf_id": 8, "key": "8"},
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

    model_config = ModelConfig(
        name="model",
        description="description",
        type=ModelType.LINKER,
        left_resolution="source_a",
        right_resolution="source_b",
    )
    model = Model(
        metadata=model_config,
        model_instance=Linker,
        left_data=left_data,
        right_data=right_data,
    )
    results = Results(probabilities=probabilities, metadata=model_config, model=model)

    # Check two ways of representing clusters
    clusters = results.clusters_to_polars()
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
        metadata=model_config,
        model=model,
    )

    assert len(empty_results.clusters_to_polars()) == 0
    expected_empty_root_leaf = pl.concat(
        [
            left_data.select(["id", "leaf_id"]).rename({"id": "root_id"}),
            right_data.select(["id", "leaf_id"]).rename({"id": "root_id"}),
        ]
    ).unique()
    assert_frame_equal(
        empty_results.root_leaf(),
        expected_empty_root_leaf,
        check_column_order=False,
        check_row_order=False,
    )

    # The above was only possible because leaf IDs were present in the inputs
    left_data.drop_in_place("leaf_id")
    right_data.drop_in_place("leaf_id")
    with pytest.raises(RuntimeError, match="must contain leaf IDs"):
        results.root_leaf()
