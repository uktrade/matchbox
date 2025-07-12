"""Common operations to produce model evaluation scores."""

from itertools import combinations
from typing import TypeAlias

import polars as pl
from pydantic import BaseModel, Field

from matchbox.common.dtos import ModelResolutionName

Pairs: TypeAlias = set[tuple[int, int]]
PrecisionRecall: TypeAlias = tuple[float, float]
ModelComparison: TypeAlias = dict[ModelResolutionName, PrecisionRecall]


class Judgement(BaseModel):
    """User determination on how to group source clusters from a model cluster."""

    user_id: int
    shown: int = Field(description="ID of the model cluster shown to the user")
    endorsed: list[list[int]] = Field(
        description="""Groups of source cluster IDs that user thinks belong together"""
    )


def contains_to_pairs(contains: pl.DataFrame) -> dict[int, Pairs]:
    """Convert clusters dataframe with parent, leaf columns to sets of pairs.

    Args:
        contains: A polars version of the judgements table produced by the server.

    Returns:
        A dict from user IDs to sets of pairs proposed, where pairs are all the possible
        pair-wise edges between source clusters with the same parent according to the
        user.
    """
    pairs: dict[int, Pairs] = {}
    all_users = contains["user_id"].unique().to_list()
    for user in all_users:
        pairs[user] = set()

        clusters = (
            contains.filter(pl.col("user_id") == user)
            .group_by("parent")
            .agg(pl.col("child").alias("children"))
            .select("children")
            .to_series()
            .to_list()
        )

        for c in clusters:
            pairs[user].update(combinations(sorted(c), r=2))

    return pairs


def precision_recall(
    model_pairs: Pairs, eval_pairs_by_user: dict[int, Pairs]
) -> PrecisionRecall:
    """Compute precision and recall scores, averaging across users."""
    all_precision = []
    all_recall = []

    if not model_pairs or not eval_pairs_by_user:
        raise ValueError("Both eval pairs and model pairs needed for precision-recall")

    for user_pairs in eval_pairs_by_user.values():
        if not user_pairs:
            raise ValueError("Every user must have at least one pair")
        true_positives = len(user_pairs.intersection(model_pairs))
        all_precision.append(true_positives / len(user_pairs))
        all_recall.append(true_positives / len(model_pairs))

    def rounded_average(score: float) -> float:
        return round(sum(score) / len(score), 3)

    return rounded_average(all_precision), rounded_average(all_recall)
