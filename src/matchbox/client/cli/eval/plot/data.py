"""Plot data validation and preparation logic."""

import logging

logger = logging.getLogger(__name__)


def can_show_plot(state) -> tuple[bool, str]:
    """Single source of truth for plot display readiness.

    Args:
        state: EvaluationState instance

    Returns:
        Tuple of (can_show: bool, status_message: str)
        If can_show is False, status_message explains why.
        If can_show is True, status_message is empty.
    """
    # Basic state checks
    if state.is_loading_eval_data:
        return False, "⏳ Loading"

    if state.eval_data_error:
        return False, "⚠ Error"

    if state.eval_data is None:
        return False, "⚠ No data"

    # Check data sufficiency without try/except
    pr_data = state.eval_data.precision_recall()
    if pr_data is None or len(pr_data) < 2:
        return False, "∅ Sparse"

    return True, ""


def refresh_judgements_for_plot(state) -> tuple[bool, str]:
    """Refresh judgements data for plotting.

    Args:
        state: EvaluationState instance

    Returns:
        Tuple of (success: bool, status_message: str)
    """
    try:
        state.eval_data.refresh_judgements()

        # Update status based on judgements count
        judgements_count = (
            len(state.eval_data.judgements)
            if state.eval_data.judgements is not None
            else 0
        )

        if judgements_count > 0:
            return True, f"📊 Got {judgements_count}"
        else:
            return True, "◯ Empty"

    except Exception as e:  # noqa: BLE001
        error_str = str(e).lower()

        if (
            ("cannot be empty" in error_str and "judgement" in error_str)
            or ("empty judgement" in error_str)
            or ("cannot compute metrics with empty judgement" in error_str)
            or ("judgements dataset must not be empty" in error_str)
        ):
            return True, "◯ Empty"  # This is expected, not a failure
        else:
            logger.error(f"Failed to refresh judgements: {e}")
            return False, "⚠ Error"
