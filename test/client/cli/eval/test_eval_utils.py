"""Tests for eval utils functionality, specifically EvalData and PR curve generation."""

import time
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest
from textual_plotext import PlotextPlot

from matchbox.client.cli.eval.plot.widget import PRCurveDisplay
from matchbox.client.cli.eval.state import EvaluationState
from matchbox.client.cli.eval.utils import EvalData
from matchbox.common.arrow import SCHEMA_CLUSTER_EXPANSION, SCHEMA_JUDGEMENTS


class TestEvalData:
    """Test the EvalData class functionality."""

    def test_sweep_efficiency_for_multiple_thresholds(self):
        """Test that sweep method is more efficient for multiple thresholds."""

        # Create a larger test dataset
        model_data = []
        for cluster_id in range(100):  # 100 clusters
            for leaf_id in range(
                cluster_id * 3, cluster_id * 3 + 3
            ):  # 3 leaves per cluster
                model_data.append({"root": cluster_id, "leaf": leaf_id})

        model_root_leaf = pa.Table.from_pylist(model_data)

        # Create judgements for some clusters
        judgements_data = []
        expansion_data = []
        for cluster_id in range(0, 50):  # Judge half the clusters
            judgements_data.append(
                {
                    "shown": cluster_id,
                    "endorsed": cluster_id,
                    "user_id": 1,
                }
            )
            expansion_data.append(
                {
                    "root": cluster_id,
                    "leaves": list(range(cluster_id * 3, cluster_id * 3 + 3)),
                }
            )

        judgements = pa.Table.from_pylist(judgements_data)
        expansion = pa.Table.from_pylist(expansion_data)

        # Create probabilities for different thresholds
        prob_data = []
        for i in range(0, 300, 2):  # Every other leaf pair
            prob_data.append(
                {
                    "left_id": i,
                    "right_id": i + 1,
                    "probability": 50 + (i % 50),  # Vary from 50 to 99
                }
            )
        probabilities = pa.Table.from_pylist(prob_data)

        with patch(
            "matchbox.client.cli.eval.utils._handler.download_eval_data"
        ) as mock_handler:
            mock_handler.return_value = (judgements, expansion)

            # Test with multiple thresholds
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

            # Create EvalData and time the sweep method
            eval_data = EvalData(
                root_leaf=model_root_leaf,
                thresholds=thresholds,
                probabilities=probabilities,
            )

            start_time = time.time()
            sweep_results = eval_data.precision_recall(thresholds)
            sweep_time = time.time() - start_time

            # Verify we got results for all thresholds
            assert len(sweep_results) == len(thresholds)

            # Verify results are sorted by threshold
            result_thresholds = [t for t, _, _, _, _ in sweep_results]
            assert result_thresholds == sorted(thresholds)

            # Test individual threshold calls would be slower (simulate)
            # The sweep processes all thresholds in one pass through the data
            # Individual calls would need to process the data separately each time

            # Just verify the sweep completed quickly (under 1 second)
            assert sweep_time < 1.0, f"Sweep took {sweep_time:.3f}s, should be fast"

            # Verify confidence intervals are calculated
            for _, p, r, p_ci, r_ci in sweep_results:
                assert isinstance(p, float) and 0 <= p <= 1
                assert isinstance(r, float) and 0 <= r <= 1
                assert isinstance(p_ci, float) and p_ci >= 0
                assert isinstance(r_ci, float) and r_ci >= 0

    def test_get_pr_curve_data_with_empty_judgements(self):
        """Test get_pr_curve_data() when judgements table is empty."""
        # Create empty judgements and expansion tables with correct schemas
        empty_judgements = pa.table(
            {
                "user_id": pa.array([], type=pa.uint64()),
                "endorsed": pa.array([], type=pa.uint64()),
                "shown": pa.array([], type=pa.uint64()),
            },
            schema=SCHEMA_JUDGEMENTS,
        )

        empty_expansion = pa.table(
            {
                "root": pa.array([], type=pa.uint64()),
                "leaves": pa.array([], type=pa.list_(pa.uint64())),
            },
            schema=SCHEMA_CLUSTER_EXPANSION,
        )

        # Create mock root_leaf data
        root_leaf_data = pa.table(
            {
                "root": pa.array([1, 1, 2]),
                "leaf": pa.array([10, 11, 20]),
            }
        )

        # Mock the _handler.download_eval_data to return empty data
        with patch(
            "matchbox.client.cli.eval.utils._handler.download_eval_data"
        ) as mock_handler:
            mock_handler.return_value = (empty_judgements, empty_expansion)

            # Create EvalData instance
            thresholds = [0.1, 0.5, 0.8]
            eval_data = EvalData(root_leaf_data, thresholds)

            # With empty judgements, should return zeros
            pr_data = eval_data.precision_recall()
            assert len(pr_data) == 3  # One for each threshold
            for _threshold, p, r, p_ci, r_ci in pr_data:
                assert p == 0.0
                assert r == 0.0
                assert p_ci == 0.0
                assert r_ci == 0.0

    def test_get_pr_curve_data_with_valid_judgements(self):
        """Test get_pr_curve_data() with valid judgements data."""
        # Create mock judgements data - user is shown cluster 1 and endorses all of it
        judgements_data = pa.table(
            {
                "user_id": pa.array([1], type=pa.uint64()),
                "endorsed": pa.array([1], type=pa.uint64()),  # endorses root cluster 1
                "shown": pa.array([1], type=pa.uint64()),  # shown root cluster 1
            },
            schema=SCHEMA_JUDGEMENTS,
        )

        # Create mock expansion data - maps root clusters to their leaf components
        expansion_data = pa.table(
            {
                "root": pa.array([1], type=pa.uint64()),
                "leaves": pa.array([[10, 11]], type=pa.list_(pa.uint64())),
            },
            schema=SCHEMA_CLUSTER_EXPANSION,
        )

        # Create mock root_leaf data
        root_leaf_data = pa.table(
            {
                "root": pa.array([1, 1]),
                "leaf": pa.array([10, 11]),
            }
        )

        # Mock the _handler.download_eval_data
        with patch(
            "matchbox.client.cli.eval.utils._handler.download_eval_data"
        ) as mock_handler:
            mock_handler.return_value = (judgements_data, expansion_data)

            # Create EvalData instance
            thresholds = [0.1, 0.5, 0.8]
            eval_data = EvalData(root_leaf_data, thresholds)

            # This should work with valid data
            try:
                pr_data = eval_data.precision_recall()

                # Check the return format
                assert isinstance(pr_data, list)
                assert len(pr_data) == len(thresholds)

                for threshold, precision, recall, precision_ci, recall_ci in pr_data:
                    assert isinstance(threshold, float)
                    assert isinstance(precision, int | float)
                    assert isinstance(recall, int | float)
                    assert isinstance(precision_ci, int | float)
                    assert isinstance(recall_ci, int | float)
                    assert 0.0 <= precision <= 1.0
                    assert 0.0 <= recall <= 1.0
                    assert precision_ci >= 0.0
                    assert recall_ci >= 0.0

                # Successfully got PR data with valid judgements

            except Exception:
                # Unexpected error - re-raise for debugging
                # Let the test fail so we can see what went wrong
                raise

    def test_get_pr_curve_data_return_format(self):
        """Test that get_pr_curve_data() returns the correct format."""
        # Create minimal mock data
        judgements_data = pa.table(
            {
                "user_id": pa.array([1], type=pa.uint64()),
                "endorsed": pa.array([1], type=pa.uint64()),  # endorses root cluster 1
                "shown": pa.array([1], type=pa.uint64()),  # shown root cluster 1
            },
            schema=SCHEMA_JUDGEMENTS,
        )

        expansion_data = pa.table(
            {
                "root": pa.array([1], type=pa.uint64()),
                "leaves": pa.array(
                    [[10, 11]], type=pa.list_(pa.uint64())
                ),  # cluster 1 contains leaves 10,11
            },
            schema=SCHEMA_CLUSTER_EXPANSION,
        )

        root_leaf_data = pa.table(
            {
                "root": pa.array([1, 1]),
                "leaf": pa.array([10, 11]),
            }
        )

        with patch(
            "matchbox.client.cli.eval.utils._handler.download_eval_data"
        ) as mock_handler:
            mock_handler.return_value = (judgements_data, expansion_data)

            thresholds = [0.5, 0.8]
            eval_data = EvalData(root_leaf_data, thresholds)

            try:
                pr_data = eval_data.precision_recall()

                # Verify exact format: list[tuple[float, float, float, float, float]]
                assert isinstance(pr_data, list)
                assert len(pr_data) == 2  # Same as number of thresholds

                for item in pr_data:
                    assert isinstance(item, tuple)
                    assert len(item) == 5
                    threshold, precision, recall, precision_ci, recall_ci = item
                    print(
                        f"Threshold: {threshold}, Precision: {precision}, "
                        f"Recall: {recall}, P_CI: {precision_ci}, R_CI: {recall_ci}"
                    )

            except Exception as e:
                # Re-raise with better context
                raise AssertionError(f"Format test error: {e}") from e

    def test_refresh_judgements_method(self):
        """Test that refresh_judgements() updates the judgements data."""
        # Initial judgements
        initial_judgements = pa.table(
            {
                "user_id": pa.array([1], type=pa.uint64()),
                "endorsed": pa.array([1], type=pa.uint64()),  # endorses cluster 1
                "shown": pa.array([1], type=pa.uint64()),  # shown cluster 1
            },
            schema=SCHEMA_JUDGEMENTS,
        )

        initial_expansion = pa.table(
            {
                "root": pa.array([1], type=pa.uint64()),
                "leaves": pa.array([[10]], type=pa.list_(pa.uint64())),
            },
            schema=SCHEMA_CLUSTER_EXPANSION,
        )

        # Updated judgements
        updated_judgements = pa.table(
            {
                "user_id": pa.array([1, 1], type=pa.uint64()),
                "endorsed": pa.array(
                    [1, 2], type=pa.uint64()
                ),  # endorses clusters 1, 2
                "shown": pa.array([1, 2], type=pa.uint64()),  # shown clusters 1, 2
            },
            schema=SCHEMA_JUDGEMENTS,
        )

        updated_expansion = pa.table(
            {
                "root": pa.array([1, 2], type=pa.uint64()),
                "leaves": pa.array([[10], [11]], type=pa.list_(pa.uint64())),
            },
            schema=SCHEMA_CLUSTER_EXPANSION,
        )

        root_leaf_data = pa.table(
            {
                "root": pa.array([1, 1]),
                "leaf": pa.array([10, 11]),
            }
        )

        with patch(
            "matchbox.client.cli.eval.utils._handler.download_eval_data"
        ) as mock_handler:
            # First call returns initial data
            mock_handler.return_value = (initial_judgements, initial_expansion)

            thresholds = [0.5]
            eval_data = EvalData(root_leaf_data, thresholds)

            # Verify initial state
            assert len(eval_data.judgements) == 1

            # Update the mock to return updated data
            mock_handler.return_value = (updated_judgements, updated_expansion)

            # Call refresh_judgements
            eval_data.refresh_judgements()

            # Verify updated state
            assert len(eval_data.judgements) == 2
            # refresh_judgements() successfully updated data


class TestPRCurveDisplay:
    """Test the PRCurveDisplay widget functionality."""

    def test_update_plot_when_plot_hidden(self):
        """Test update_plot() when show_plot is False."""
        # Create mock state with show_plot = False
        state = Mock(spec=EvaluationState)
        state.show_plot = False
        state.add_listener = Mock()
        state.is_loading_eval_data = False
        state.eval_data = None
        state.eval_data_error = None

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not raise exception
        display.update_plot()
        assert display._has_plotted is False

    def test_update_plot_loading_state(self):
        """Test update_plot() shows loading message when eval data is loading."""
        # Create mock state with loading = True
        state = Mock(spec=EvaluationState)
        state.show_plot = True
        state.is_loading_eval_data = True
        state.add_listener = Mock()

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not crash and should set has_plotted to False
        display.update_plot()
        assert display._has_plotted is False

    def test_update_plot_error_state(self):
        """Test update_plot() handles error state when eval_data_error is set."""
        # Create mock state with error
        state = Mock(spec=EvaluationState)
        state.show_plot = True
        state.is_loading_eval_data = False
        state.eval_data_error = "Some error message"
        state.add_listener = Mock()

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not crash and should set has_plotted to False
        display.update_plot()
        assert display._has_plotted is False

    def test_update_plot_no_eval_data(self):
        """Test update_plot() handles case when eval_data is None."""
        # Create mock state with no eval data
        state = Mock(spec=EvaluationState)
        state.show_plot = True
        state.is_loading_eval_data = False
        state.eval_data_error = None
        state.eval_data = None
        state.add_listener = Mock()

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not crash and should set has_plotted to False
        display.update_plot()
        assert display._has_plotted is False

    def test_update_plot_empty_pr_data(self):
        """Test update_plot() handles empty PR data gracefully."""
        # Create mock eval_data that returns empty list
        mock_eval_data = Mock()
        mock_eval_data.precision_recall.return_value = []

        # Create mock state with empty eval data
        state = Mock(spec=EvaluationState)
        state.show_plot = True
        state.is_loading_eval_data = False
        state.eval_data_error = None
        state.eval_data = mock_eval_data
        state.add_listener = Mock()

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not crash and should set has_plotted to False
        display.update_plot()
        assert display._has_plotted is False

    def test_update_plot_with_valid_pr_data(self):
        """Test update_plot() creates plot when valid PR data is available."""
        # Create mock eval_data that returns valid PR data
        mock_eval_data = Mock()
        mock_eval_data.precision_recall.return_value = [
            (0.1, 0.9, 0.5, 0.1, 0.1),  # (threshold, precision, recall, p_ci, r_ci)
            (0.5, 0.8, 0.7, 0.1, 0.1),
            (0.8, 0.6, 0.9, 0.1, 0.1),
        ]

        # Create mock state with valid eval data
        state = Mock(spec=EvaluationState)
        state.show_plot = True
        state.is_loading_eval_data = False
        state.eval_data_error = None
        state.eval_data = mock_eval_data
        state.add_listener = Mock()

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not crash and should set has_plotted to True
        display.update_plot()
        assert display._has_plotted is True

    def test_update_plot_exception_handling(self):
        """Test update_plot() handles exceptions gracefully."""
        # Create mock eval_data that raises exception
        mock_eval_data = Mock()
        mock_eval_data.precision_recall.side_effect = Exception("Test error")

        # Create mock state
        state = Mock(spec=EvaluationState)
        state.show_plot = True
        state.is_loading_eval_data = False
        state.eval_data_error = None
        state.eval_data = mock_eval_data
        state.add_listener = Mock()
        state.set_eval_data_error = Mock()
        state.update_status = Mock()

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not crash and should handle error gracefully
        display.update_plot()
        assert display._has_plotted is False
        # The key behavior is that it doesn't crash - error handling details may vary

    def test_update_plot_empty_judgements_error(self):
        """Test update_plot() handles empty judgements error with user-friendly
        response."""
        # Create mock eval_data that raises judgements empty error
        mock_eval_data = Mock()
        mock_eval_data.precision_recall.side_effect = Exception(
            "Judgements data cannot be empty."
        )

        # Create mock state
        state = Mock(spec=EvaluationState)
        state.show_plot = True
        state.is_loading_eval_data = False
        state.eval_data_error = None
        state.eval_data = mock_eval_data
        state.add_listener = Mock()

        # Create PRCurveDisplay
        display = PRCurveDisplay(state)

        # Should not crash and should handle empty judgements gracefully
        display.update_plot()
        assert display._has_plotted is False

    def test_textual_plotext_integration(self):
        """Test that textual-plotext integration works correctly."""
        # Create a simple widget to test textual-plotext functionality
        plot_widget = PlotextPlot()

        # Test that we can access the plt property
        assert hasattr(plot_widget, "plt")
        assert plot_widget.plt is not None

        # Test basic plotting operations work without error
        try:
            plot_widget.plt.clear_data()
            plot_widget.plt.clear_figure()
            plot_widget.plt.scatter([1, 2, 3], [1, 2, 3])
            plot_widget.plt.title("Test Plot")
            # This should not raise an exception
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"Basic textual-plotext operations failed: {e}")
