import polars as pl
import pyarrow as pa
import pytest

from matchbox.common.arrow import SCHEMA_CLUSTER_EXPANSION, SCHEMA_JUDGEMENTS
from matchbox.common.eval import Judgement, precision_recall, process_judgements


def test_judgement_validation():
    """Judgement validates source cluster IDs."""
    with pytest.raises(ValueError):
        Judgement(user_id=1, shown=10, endorsed=[[1, 2, 3], [3, 4, 5]])

    with pytest.raises(ValueError):
        Judgement(user_id=1, shown=10, endorsed=[[1, 2, 3, 1]])

    # Something sensible does work
    Judgement(user_id=1, shown=10, endorsed=[[1, 2, 3], [4, 5]])


def test_precision_recall_fails():
    """Test instances where PR computation raises."""
    # No judgements
    model = pa.Table.from_pylist([{"root": 12, "leaf": 1}, {"root": 12, "leaf": 2}])
    empty_judgements = pa.Table.from_pylist([], schema=SCHEMA_JUDGEMENTS)
    empty_expansion = pa.Table.from_pylist([], schema=SCHEMA_CLUSTER_EXPANSION)

    with pytest.raises(ValueError, match="Judgements data"):
        precision_recall(
            models_root_leaf=[model],
            judgements=empty_judgements,
            expansion=empty_expansion,
        )

    # No model root-leaf
    empty_model = pa.Table.from_pylist([])
    judgements = pa.Table.from_pylist([{"shown": 12, "endorsed": 12}])
    expansion = pa.Table.from_pylist([{"root": 12, "leaves": [1, 2]}])

    with pytest.raises(ValueError, match="Model data"):
        precision_recall(
            models_root_leaf=[empty_model], judgements=judgements, expansion=expansion
        )


def test_precision_recall():
    """Test calculation of precision and recall from root-leaf tables."""
    # In this test, one-digit cluster IDs are for source clusters.
    # Multiple-digit cluster IDs decompose to source cluster IDs.
    # For example, 123 maps to 1,2,3
    model1 = pa.Table.from_pylist(
        [
            # (1,2,3)
            {"root": 123, "leaf": 1},
            {"root": 123, "leaf": 2},
            {"root": 123, "leaf": 3},
            # (4,5)
            {"root": 45, "leaf": 4},
            {"root": 45, "leaf": 5},
            # (6,7)
            {"root": 67, "leaf": 6},
            {"root": 67, "leaf": 7},
            # (8,9)
            {"root": 89, "leaf": 8},
            {"root": 89, "leaf": 9},
        ]
    )

    model2 = pa.Table.from_pylist(
        [
            # (1,3)
            {"root": 13, "leaf": 1},
            {"root": 13, "leaf": 3},
            # (2)
            {"root": 2, "leaf": 2},
            # (4)
            {"root": 4, "leaf": 4},
            # (5)
            {"root": 5, "leaf": 5},
            # (6,7)
            {"root": 67, "leaf": 6},
            {"root": 67, "leaf": 7},
        ]
    )

    judgement_cluster_expansion = pa.Table.from_pylist(
        [
            {"root": 123, "leaves": [1, 2, 3]},
            {"root": 67, "leaves": [6, 7]},
            {"root": 45, "leaves": [4, 5]},
            {"root": 12, "leaves": [1, 2]},
        ]
    )

    judgements = pa.Table.from_pylist(
        [
            # Ambiguous but more positive than negative
            {"shown": 123, "endorsed": 12},
            {"shown": 123, "endorsed": 3},
            {"shown": 123, "endorsed": 12},
            {"shown": 123, "endorsed": 3},
            {"shown": 123, "endorsed": 1},
            {"shown": 123, "endorsed": 2},
            {"shown": 123, "endorsed": 3},
            # Ambiguous but more negative than positive
            {"shown": 45, "endorsed": 45},
            {"shown": 45, "endorsed": 4},
            {"shown": 45, "endorsed": 4},
            {"shown": 45, "endorsed": 5},
            {"shown": 45, "endorsed": 5},
            # The following neutralise each other
            {"shown": 67, "endorsed": 67},
            {"shown": 67, "endorsed": 6},
            {"shown": 67, "endorsed": 7},
        ]
    )

    pr_scores = precision_recall(
        models_root_leaf=[model1, model2],
        judgements=judgements,
        expansion=judgement_cluster_expansion,
    )

    # Validation pairs: (12) (as (45) is more negative; (67) is neutralised)
    # Model 1 pairs: (12), (13), (23) (as (67) is neutralised; (89) has extra leaves)
    assert pr_scores[0][:2] == (1 / 4, 1)  # Check precision and recall, ignore CI
    # Model 2 pairs: (13) (as (67) is neutralised)
    assert pr_scores[1][:2] == (0, 0)  # Check precision and recall, ignore CI


def test_process_judgements():
    """Can convert judgements and expansion to pairs, pair counts and set of leaves."""
    # In this test:
    # cluster IDs < 100 are source clusters
    # 1xx clusters IDs are model clusters
    # 2xx clusters IDs are new judgement clusters (splintered model clusters)

    judgement_cluster_expansion = pl.DataFrame(
        [
            # Shown clusters
            {"root": 100, "leaves": [1, 2]},
            {"root": 101, "leaves": [3, 4, 5]},
            {"root": 102, "leaves": [6, 7]},
            {"root": 103, "leaves": [8, 9]},
            {"root": 104, "leaves": [10, 11]},
            {"root": 105, "leaves": [12, 13]},
            {"root": 106, "leaves": [12, 13, 14]},
            # New "splintered" clusters
            {"root": 200, "leaves": [3, 4]},
        ]
    )

    judgements = pl.DataFrame(
        [
            # A) cluster confirmed as is
            {"shown": 100, "endorsed": 100},  # (1,2)->(1,2)
            # B) cluster splintered
            {"shown": 101, "endorsed": 200},  # (3,4,5)->(3,4)
            {"shown": 101, "endorsed": 5},  # (3,4,5)->(5)
            # C) net positive opinion
            {"shown": 102, "endorsed": 102},  # (6,7)->(6,7)
            {"shown": 102, "endorsed": 102},  # repeated...
            {"shown": 102, "endorsed": 6},  # ...and negated: (6,7)->(6)
            {"shown": 102, "endorsed": 7},  # (6,7)->(7)
            # D) net negative opinion
            {"shown": 103, "endorsed": 103},  # (8,9)->(8,9)
            {"shown": 103, "endorsed": 8},  # negated: (8,9)->(8)
            {"shown": 103, "endorsed": 9},  # (8,9)->(9)
            {"shown": 103, "endorsed": 8},  # negated again: (8,9)->(8)
            {"shown": 103, "endorsed": 9},  # (8,9)->(9)
            # E) neutral opinion
            {"shown": 104, "endorsed": 104},  # (10,11)->(10,11)
            {"shown": 104, "endorsed": 10},  # negated: (10,11)->(10)
            {"shown": 104, "endorsed": 11},  # (10,11)->(11)
            # F) judgement has seen less than a previous one
            {"shown": 105, "endorsed": 105},  # (12,13) -> (12,13)
            {"shown": 106, "endorsed": 106},  # (12,13,14) -> (12,13,14)
        ]
    )

    pairs, net_counts, leaves = process_judgements(
        judgements, judgement_cluster_expansion
    )
    assert pairs == {
        (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (12, 14),
        (13, 14),
    }

    for pair in [(1, 2), (3, 4), (6, 7), (12, 14), (13, 14)]:
        assert net_counts[pair] == 1

    for pair in [(3, 5), (4, 5), (8, 9)]:
        assert net_counts[pair] == -1

    for pair in [(10, 11)]:
        assert net_counts[pair] == 0

    for pair in [(12, 13)]:
        assert net_counts[pair] == 2

    assert leaves == set(range(1, 15))


def test_evaldata_unified_pr_calculation():
    """Test EvalData unified PR calculation with single and multi-threshold modes."""
    import unittest.mock

    # Create mock handler that returns our test data
    judgements = pa.table(
        {
            "user_id": [1, 1],
            "shown": [100, 100],
            "endorsed": [200, 3],
        }
    )

    expansion = pa.table(
        {
            "root": [100, 200, 3],
            "leaves": [[1, 2, 3], [1, 2], [3]],
        }
    )

    model_root_leaf = pa.table({"root": [1, 2], "leaf": [1, 2]})

    probabilities = pa.table(
        {
            "left_id": [1, 2],
            "right_id": [2, 3],
            "probability": [90, 70],
        }
    )

    # Mock the handler to return our test data
    with unittest.mock.patch(
        "matchbox.client.cli.eval.utils._handler.download_eval_data"
    ) as mock_download:
        mock_download.return_value = (judgements, expansion)

        from matchbox.client.cli.eval.utils import EvalData

        # Create EvalData instance
        eval_data = EvalData(
            root_leaf=model_root_leaf,
            thresholds=[0.9, 0.7],
            probabilities=probabilities,
        )

        # Test single threshold mode
        single_results = eval_data.precision_recall([0.8])
        assert len(single_results) == 1
        assert single_results[0][0] == 0.8  # threshold

        # Test multi-threshold mode (uses optimization)
        multi_results = eval_data.precision_recall([0.9, 0.7])
        assert len(multi_results) == 2
        assert multi_results[0][0] == 0.7  # sorted ascending
        assert multi_results[1][0] == 0.9

        # Test using instance thresholds
        default_results = eval_data.precision_recall()
        assert len(default_results) == 2  # Uses instance thresholds


def test_evaldata_matches_precision_recall():
    """Test that EvalData sweep method produces identical results."""
    import unittest.mock

    from matchbox.client.cli.eval.utils import EvalData
    from matchbox.common.eval import precision_recall

    # Use the exact same test data as test_precision_recall
    model1 = pa.Table.from_pylist(
        [
            # (1,2,3)
            {"root": 123, "leaf": 1},
            {"root": 123, "leaf": 2},
            {"root": 123, "leaf": 3},
            # (4,5)
            {"root": 45, "leaf": 4},
            {"root": 45, "leaf": 5},
            # (6,7)
            {"root": 67, "leaf": 6},
            {"root": 67, "leaf": 7},
            # (8,9)
            {"root": 89, "leaf": 8},
            {"root": 89, "leaf": 9},
        ]
    )

    judgement_cluster_expansion = pa.Table.from_pylist(
        [
            {"root": 123, "leaves": [1, 2, 3]},
            {"root": 67, "leaves": [6, 7]},
            {"root": 45, "leaves": [4, 5]},
            {"root": 12, "leaves": [1, 2]},
        ]
    )

    judgements = pa.Table.from_pylist(
        [
            # Ambiguous but more positive than negative
            {"shown": 123, "endorsed": 12, "user_id": 1},
            {"shown": 123, "endorsed": 3, "user_id": 1},
            {"shown": 123, "endorsed": 12, "user_id": 1},
            {"shown": 123, "endorsed": 3, "user_id": 1},
            {"shown": 123, "endorsed": 1, "user_id": 1},
            {"shown": 123, "endorsed": 2, "user_id": 1},
            {"shown": 123, "endorsed": 3, "user_id": 1},
            # Ambiguous but more negative than positive
            {"shown": 45, "endorsed": 45, "user_id": 1},
            {"shown": 45, "endorsed": 4, "user_id": 1},
            {"shown": 45, "endorsed": 4, "user_id": 1},
            {"shown": 45, "endorsed": 5, "user_id": 1},
            {"shown": 45, "endorsed": 5, "user_id": 1},
            # The following neutralise each other
            {"shown": 67, "endorsed": 67, "user_id": 1},
            {"shown": 67, "endorsed": 6, "user_id": 1},
            {"shown": 67, "endorsed": 7, "user_id": 1},
        ]
    )

    # Get results from original precision_recall function
    original_results = precision_recall(
        models_root_leaf=[model1],
        judgements=judgements,
        expansion=judgement_cluster_expansion,
    )

    # Mock the handler to return our test data
    with unittest.mock.patch(
        "matchbox.client.cli.eval.utils._handler.download_eval_data"
    ) as mock_download:
        mock_download.return_value = (judgements, judgement_cluster_expansion)

        # Create EvalData instance and get results using sweep method
        eval_data = EvalData(
            root_leaf=model1,
            thresholds=[1.0],  # Single threshold for direct comparison
            probabilities=None,  # No probabilities means include all pairs
        )

        sweep_results = eval_data.precision_recall([1.0])

        # Extract values from sweep results (threshold, p, r, p_ci, r_ci)
        _, sweep_p, sweep_r, sweep_p_ci, sweep_r_ci = sweep_results[0]

        # Extract values from original results (p, r, p_ci, r_ci)
        orig_p, orig_r, orig_p_ci, orig_r_ci = original_results[0]

        # Verify they match exactly
        assert sweep_p == orig_p, f"Precision mismatch: {sweep_p} != {orig_p}"
        assert sweep_r == orig_r, f"Recall mismatch: {sweep_r} != {orig_r}"
        assert (
            sweep_p_ci == orig_p_ci
        ), f"Precision CI mismatch: {sweep_p_ci} != {orig_p_ci}"
        assert (
            sweep_r_ci == orig_r_ci
        ), f"Recall CI mismatch: {sweep_r_ci} != {orig_r_ci}"

        # Also verify the expected values from test_precision_recall
        assert sweep_p == 1 / 4, f"Expected precision 1/4, got {sweep_p}"
        assert sweep_r == 1, f"Expected recall 1, got {sweep_r}"


def test_evaldata_optimization_consistency():
    """Test that optimized multi-threshold matches single-threshold results."""
    import unittest.mock

    # Simple test case with known results
    judgements = pa.table(
        {
            "user_id": [1],
            "shown": [100],
            "endorsed": [100],
        }
    )

    expansion = pa.table(
        {
            "root": [100],
            "leaves": [[1, 2]],
        }
    )

    model_root_leaf = pa.table({"root": [1, 1], "leaf": [1, 2]})

    # Mock the handler
    with unittest.mock.patch(
        "matchbox.client.cli.eval.utils._handler.download_eval_data"
    ) as mock_download:
        mock_download.return_value = (judgements, expansion)

        from matchbox.client.cli.eval.utils import EvalData

        eval_data = EvalData(
            root_leaf=model_root_leaf, thresholds=[0.8], probabilities=None
        )

        # Get results from both single and multi-threshold paths
        single_result = eval_data.precision_recall([0.8])[0]
        multi_result = eval_data.precision_recall([0.8, 0.6])[1]  # 0.8 result

        # Results should be identical (excluding threshold value)
        assert single_result[1:] == multi_result[1:]  # precision, recall, CIs


def test_evaldata_handles_edge_cases():
    """Test EvalData correctly handles neutral and negative pairs."""
    import unittest.mock

    from matchbox.client.cli.eval.utils import EvalData
    from matchbox.common.eval import precision_recall

    # Model with various clusters
    model = pa.Table.from_pylist(
        [
            # Cluster that will be endorsed
            {"root": 12, "leaf": 1},
            {"root": 12, "leaf": 2},
            # Cluster that will be neutral (net count = 0)
            {"root": 34, "leaf": 3},
            {"root": 34, "leaf": 4},
            # Cluster that will be negative (rejected)
            {"root": 56, "leaf": 5},
            {"root": 56, "leaf": 6},
        ]
    )

    expansion = pa.Table.from_pylist(
        [
            {"root": 12, "leaves": [1, 2]},
            {"root": 34, "leaves": [3, 4]},
            {"root": 56, "leaves": [5, 6]},
        ]
    )

    judgements = pa.Table.from_pylist(
        [
            # Positive endorsement
            {"shown": 12, "endorsed": 12, "user_id": 1},
            # Neutral (one endorsement, one rejection)
            {"shown": 34, "endorsed": 34, "user_id": 1},
            {"shown": 34, "endorsed": 3, "user_id": 1},
            {"shown": 34, "endorsed": 4, "user_id": 1},
            # Negative (more rejections than endorsements)
            {"shown": 56, "endorsed": 56, "user_id": 1},
            {"shown": 56, "endorsed": 5, "user_id": 1},
            {"shown": 56, "endorsed": 6, "user_id": 1},
            {"shown": 56, "endorsed": 5, "user_id": 1},
            {"shown": 56, "endorsed": 6, "user_id": 1},
        ]
    )

    # Get results from original function
    original_results = precision_recall(
        models_root_leaf=[model],
        judgements=judgements,
        expansion=expansion,
    )

    # Mock and test with EvalData
    with unittest.mock.patch(
        "matchbox.client.cli.eval.utils._handler.download_eval_data"
    ) as mock_download:
        mock_download.return_value = (judgements, expansion)

        eval_data = EvalData(
            root_leaf=model,
            thresholds=[1.0],
            probabilities=None,
        )

        sweep_results = eval_data.precision_recall([1.0])
        _, sweep_p, sweep_r, sweep_p_ci, sweep_r_ci = sweep_results[0]
        orig_p, orig_r, orig_p_ci, orig_r_ci = original_results[0]

        # Verify exact match
        assert sweep_p == orig_p
        assert sweep_r == orig_r
        assert sweep_p_ci == orig_p_ci
        assert sweep_r_ci == orig_r_ci

        # Model pairs include (1,2) and (5,6) - neutral excluded, negative kept
        # Validation pairs only include (1,2) since it's the only positive
        # So precision = 1/2 = 0.5 and recall = 1/1 = 1.0
        assert sweep_p == 0.5, f"Expected precision 0.5, got {sweep_p}"
        assert sweep_r == 1.0, f"Expected recall 1.0, got {sweep_r}"
