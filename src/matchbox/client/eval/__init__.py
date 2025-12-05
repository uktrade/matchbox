"""Public evaluation helpers for Matchbox clients."""

from matchbox.client.eval.samples import (
    EvalData,
    EvaluationFieldMetadata,
    EvaluationItem,
    ModelComparison,
    compare_models,
    create_evaluation_item,
    create_judgement,
    get_samples,
    precision_recall,
)

__all__ = [
    "EvalData",
    "EvaluationFieldMetadata",
    "EvaluationItem",
    "compare_models",
    "create_evaluation_item",
    "create_judgement",
    "get_samples",
    "ModelComparison",
    "precision_recall",
]
