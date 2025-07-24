"""Common operations to produce model evaluation scores."""

from itertools import chain, combinations
from typing import TypeAlias

import polars as pl
from pyarrow import Table
from pydantic import BaseModel, Field, field_validator

from matchbox.common.dtos import ModelResolutionName

Pair: TypeAlias = tuple[int, int]
Pairs: TypeAlias = set[Pair]

PrecisionRecall: TypeAlias = tuple[float, float]
ModelComparison: TypeAlias = dict[ModelResolutionName, PrecisionRecall]


class Judgement(BaseModel):
    """User determination on how to group source clusters from a model cluster."""

    user_id: int
    shown: int = Field(description="ID of the model cluster shown to the user")
    endorsed: list[list[int]] = Field(
        description="""Groups of source cluster IDs that user thinks belong together"""
    )

    @field_validator("endorsed", mode="before")
    @classmethod
    def check_endorsed(cls, value: list[list[int]]):
        """Ensure no cluster IDs are repeated in the endorsement."""
        concat_ids = list(chain(*value))
        if len(concat_ids) != len(set(concat_ids)):
            raise ValueError(
                "One or more cluster IDs were repeated in the endorsement data."
            )

        return value


def precision_recall(
    models_root_leaf: list[Table], judgements: Table, expansion: Table
) -> list[PrecisionRecall]:
    """From models and eval data, compute scores inspired by precision-recall.

    This function does the following:

    - Convert model and judgement clusters to implied pair-wise connections.
        For judgments, this includes the pairs shown to users, but rejected.
        Sum how many times pairs were endorsed (+1) or rejected (-1).
    - Keep only the pairs where leaves are present in all models and in the judgements,
    so the comparison is fair.
    - If a validation pair was rejected as many times as it was endorsed, discard it
        from both model and validation pairs.
    - If a validation pair was rejected more times than it was endorsed, remove it from
        validation pairs, but keep it in model pairs.
    - Precision and recall are computed for each model against validation pairs.

    At the moment, this function ignores user IDs.

    Args:
        models_root_leaf: list of tables with root and leaf columns, one per model.
            They must include all the clusters that resolve from a model, all the way
            to the original source clusters if no model in the lineage merged them.
        judgements: Dataframe following `matchbox.common.arrow.SCHEMA_JUDGEMENTS`.
        expansion: Dataframe following `matchbox.common.arrow.SCHEMA_CLUSTER_EXPANSION`.

    Returns:
        List of tuples of precision and recall scores, one per model.
    """
    leaves_per_set: list[set[int]] = []  # one entry for each model, one for judgements
    pairs_per_model: list[Pairs] = []

    if not len(judgements):
        raise ValueError("Judgements data cannot be empty.")

    # Process models and judgements
    for root_leaf in models_root_leaf:
        if not len(root_leaf):
            raise ValueError("Model data cannot be empty.")
        leaves_per_set.append(set(root_leaf["leaf"].to_pylist()))
        clusters = (
            pl.from_arrow(root_leaf)
            .group_by("root")
            .agg(pl.col("leaf").alias("leaves"))
            .select("leaves")
            .to_series()
            .to_list()
        )
        model_pairs = set()
        for c in clusters:
            model_pairs.update(combinations(sorted(c), r=2))
        pairs_per_model.append(model_pairs)

    validation_pairs, validation_net_count, validation_leaves = process_judgements(
        pl.from_arrow(judgements), pl.from_arrow(expansion)
    )
    leaves_per_set.append(validation_leaves)

    # Filter pairs based on overlap between leaves
    # For example, if model1 has (1,2),(1,3),(2,3); model2 has (1,10),(2,20);
    # judgements have (1),(2,3); then only keep (1,2)
    shared_leaves = set.intersection(*leaves_per_set)

    for i, model_pairs in enumerate(pairs_per_model):
        pairs_per_model[i] = {
            (a, b)
            for a, b in model_pairs
            if (
                a in shared_leaves
                and b in shared_leaves
                # remove neutrally-judged pairs from the model
                and validation_net_count[(a, b)] != 0
            )
        }

    validation_pairs = {
        (a, b)
        for a, b in validation_pairs
        if a in shared_leaves
        and b in shared_leaves
        # remove all neutrally or negatively judged pairs
        and validation_net_count[(a, b)] > 0
    }

    # Compute PR scores for each model
    pr_scores: list[PrecisionRecall] = []
    for model_pairs in pairs_per_model:
        true_positive_pairs = model_pairs & validation_pairs
        pr_scores.append(
            (
                len(true_positive_pairs) / len(model_pairs),
                len(true_positive_pairs) / len(validation_pairs),
            )
        )

    return pr_scores


def process_judgements(
    judgements: pl.DataFrame, expansion: pl.DataFrame
) -> tuple[Pairs, dict[Pair, float], set[int]]:
    """Convert judgements to pairs, net counts per pair, and set of source cluster IDs.

    In general, pairs include all (sorted) pair-wise combinations of elements in a list
    of cluster IDs. For example, (123) will give us (1,2), (1,3), (2,3). We, however,
    need to capture the difference between when a user is shown (12) and endorses (12),
    vs. when the user is shown (123) and endorses (12). In the second case, the user
    implies a negative judgement over pairs (1,3) and (2,3). We return the net value of
    pairs by summing 1 for an endorsement and subtracting 1 for a rejection.

    This function relies on the input data being well-formed. For example,

    - All shown cluster IDs need to have an expansion.
    - All endorsed cluster IDs need to have an expansion, unless they're leaves.
    - No partial splintered clusters, i.e. if (123)->(12) is in the data, (123)->(3)
        must be as well.

    Args:
        judgements: Dataframe following `matchbox.common.arrow.SCHEMA_JUDGEMENTS`.
        expansion: Dataframe following `matchbox.common.arrow.SCHEMA_CLUSTER_EXPANSION`.

    Returns:
        Tuple of:

        - Set of pairs, for all endorsements and rejections.
        - Dict mapping pairs to net (positive or negative) number of judgements.
        - Set of all cluster IDs shown to users.
    """
    expanded_judgements = (
        judgements.join(expansion, left_on="shown", right_on="root")
        .rename({"leaves": "shown_leaves"})
        .join(
            expansion, left_on="endorsed", right_on="root", how="left"
        )  # left join as singleton leaves won't be expanded
        .rename({"leaves": "endorsed_leaves"})
        # if missing expansion, assume we're dealing with singleton leaves
        .with_columns(
            pl.when(pl.col("endorsed_leaves").is_null())
            .then(pl.col("endorsed").map_elements(lambda x: [x]))
            .otherwise(pl.col("endorsed_leaves"))
            .alias("endorsed_leaves")
        )
    )

    all_leaves = set(
        chain.from_iterable(expanded_judgements["endorsed_leaves"].to_list())
    )

    if len(expanded_judgements) != len(judgements):
        raise ValueError("Malformed judgements / expansion data received.")

    validation_net_count: dict[Pair, float] = {}
    for shown, endorsed in zip(
        expanded_judgements["shown_leaves"].to_list(),
        expanded_judgements["endorsed_leaves"].to_list(),
        strict=True,
    ):
        positive_pairs = set(combinations(sorted(endorsed), r=2))
        potential_pairs = set(combinations(sorted(shown), r=2))
        negative_pairs = potential_pairs - positive_pairs

        # For every pair implicitly rejected, we subtract from its count the proportion
        # of the shown cluster endorsed in this row. This is to avoid double counting.
        # For example, if user shown (123) and endorses (12) and (3), the pairs (13) and
        # (23), which are rejected on two rows, will subtract -2/3 and -1/3 respectively
        negative_adjustment = len(endorsed) / len(shown)
        validation_net_count.update(
            {
                p: validation_net_count.get(p, 0) - negative_adjustment
                for p in negative_pairs
            }
        )

        # For every pair explicitly endorsed, we add 1 to its count, plus the proportion
        # of the shown cluster endorsed in this row. This is to avoid under-counting.
        # For example, if user shown (1234) and endorses (1) and (234), the pairs
        # (23), (24) and (34) will all be initially subtacted 1/4, and thus need to be
        # then added 1 + 1/4 = 1 + (proportion of cluster endorsed in other rows)
        positive_adjustment = 1 + ((len(shown) - len(endorsed)) / len(shown))
        validation_net_count.update(
            {
                p: validation_net_count.get(p, 0) + positive_adjustment
                for p in positive_pairs
            }
        )

    all_pairs = set(validation_net_count.keys())

    return all_pairs, validation_net_count, all_leaves
