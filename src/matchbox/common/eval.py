"""Common operations to produce model evaluation scores."""

from itertools import combinations
from typing import TypeAlias

import polars as pl
from pydantic import BaseModel

from matchbox.common.dtos import ModelResolutionName

Pairs: TypeAlias = set[tuple[int, int]]
PrecisionRecall: TypeAlias = tuple[float, float]
ModelComparison: TypeAlias = dict[ModelResolutionName, PrecisionRecall]


class Judgement(BaseModel):
    """Representation of how to split a set of entities into clusters."""

    user_id: int
    clusters: list[list[int]]


def contains_to_pairs(contains: pl.DataFrame) -> Pairs:
    """Convert clusters dataframe with parent, leaf columns to sets of pairs."""
    pairs = set()
    clusters = (
        contains.group_by("parent")
        .agg(pl.col("leaf").alias("leaves"))
        .select("leaves")
        .to_series()
        .to_list()
    )
    for c in clusters:
        pairs.update(combinations(sorted(c), r=2))

    return pairs


def precision_recall(model_pairs: Pairs, eval_pairs: Pairs) -> PrecisionRecall:
    """Compute precision and recall scores."""
    true_positives = len(eval_pairs.intersection(model_pairs))
    precision = true_positives / len(model_pairs)
    recall = true_positives / len(eval_pairs)

    return round(precision, 3), round(recall, 3)
