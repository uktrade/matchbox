"""Minimal coverage tests for eval plot functionality."""

from matchbox.client.cli.eval.plot import (
    _deduplicate_recall_values,  # noqa: PLC2701
    compute_pr_envelope,
    interpolate_pr_curve,
)


class TestPlotFunctions:
    """Basic tests for plot functions."""

    def test_compute_pr_envelope_with_valid_data(self):
        """Test envelope computation with valid data."""
        pr_data = [(0.9, 0.8, 0.6, 0.05, 0.03), (0.7, 0.7, 0.7, 0.04, 0.04)]
        r_grid, p_upper, p_lower = compute_pr_envelope(pr_data)

        assert len(r_grid) == 101
        assert len(p_upper) == 101
        assert len(p_lower) == 101
        assert all(0 <= r <= 1 for r in r_grid)
        assert all(0 <= p <= 1 for p in p_upper)
        assert all(0 <= p <= 1 for p in p_lower)

    def test_compute_pr_envelope_empty_data(self):
        """Test envelope computation with empty data."""
        r_grid, p_upper, p_lower = compute_pr_envelope([])

        assert len(r_grid) == 0
        assert len(p_upper) == 0
        assert len(p_lower) == 0

    def test_interpolate_pr_curve_with_valid_data(self):
        """Test PR curve interpolation with valid data."""
        pr_data = [(0.9, 0.8, 0.6, 0.05, 0.03), (0.7, 0.7, 0.7, 0.04, 0.04)]
        r_curve, p_curve, is_extrapolated = interpolate_pr_curve(pr_data)

        assert len(r_curve) == 101
        assert len(p_curve) == 101
        assert len(is_extrapolated) == 101
        assert all(0 <= r <= 1 for r in r_curve)
        assert all(0 <= p <= 1 for p in p_curve)
        assert is_extrapolated[0] in [True, False]

    def test_interpolate_pr_curve_empty_data(self):
        """Test PR curve interpolation with empty data."""
        r_curve, p_curve, is_extrapolated = interpolate_pr_curve([])

        assert len(r_curve) == 0
        assert len(p_curve) == 0
        assert len(is_extrapolated) == 0

    def test_deduplicate_recall_values_with_duplicates(self):
        """Test deduplication handles duplicate recall values correctly."""
        # Create data with duplicate recall values
        pr_data = [
            (0.9, 0.85, 0.6, 0.05, 0.03),  # threshold=0.9, precision=0.85, recall=0.6
            (
                0.8,
                0.82,
                0.6,
                0.04,
                0.02,
            ),  # threshold=0.8, precision=0.82, recall=0.6 (duplicate!)
            (0.7, 0.75, 0.7, 0.06, 0.03),  # threshold=0.7, precision=0.75, recall=0.7
        ]

        result = _deduplicate_recall_values(pr_data)

        # Should have 2 unique points (0.6 and 0.7 recall)
        assert len(result) == 2

        # Check recall values are unique and sorted
        recall_values = [r[2] for r in result]
        assert recall_values == [0.6, 0.7]

        # Check averaged precision for duplicate recall (0.85 + 0.82) / 2 = 0.835
        first_point = result[0]  # recall=0.6
        assert first_point[1] == 0.835  # averaged precision

        # Check single point preserved
        second_point = result[1]  # recall=0.7
        assert second_point[1] == 0.75  # original precision

    def test_interpolate_pr_curve_with_duplicate_recalls(self):
        """Test interpolation works with data that has duplicate recall values."""
        # This would previously fail with "x must be strictly increasing"
        pr_data = [
            (0.9, 0.85, 0.6, 0.05, 0.03),
            (0.8, 0.82, 0.6, 0.04, 0.02),  # duplicate recall
            (0.7, 0.75, 0.7, 0.06, 0.03),
        ]

        # Should not raise ValueError anymore
        r_curve, p_curve, is_extrapolated = interpolate_pr_curve(pr_data)

        assert len(r_curve) == 101
        assert len(p_curve) == 101
        assert len(is_extrapolated) == 101
        assert all(0 <= r <= 1 for r in r_curve)
        assert all(0 <= p <= 1 for p in p_curve)
