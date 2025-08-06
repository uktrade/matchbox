"""Common operations to produce model evaluation scores."""

from itertools import chain, combinations
from typing import TypeAlias

import polars as pl
from pyarrow import Table
from pydantic import BaseModel, Field, field_validator

from matchbox.common.graph import ModelResolutionName

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
            .then(
                pl.col("endorsed").map_elements(
                    lambda x: [x], return_dtype=pl.List(pl.UInt64)
                )
            )
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

    for row in expanded_judgements.rows(named=True):
        shown = row["shown_leaves"]
        endorsed = row["endorsed_leaves"]

        positive_pairs = set(combinations(sorted(endorsed), r=2))
        potential_pairs = set(combinations(sorted(shown), r=2))
        negative_pairs = potential_pairs - positive_pairs

        # -- EXAMPLE --
        # User is shown cluster (1234) and endorses three groups: (1), (23), and (4).
        # This means: pair (2,3) is GOOD, but (1,2), (1,3), (1,4), (2,4), (3,4) are BAD.
        # We want to return: +1 for (2,3), -1 for each rejected pair.

        # THE CHALLENGE: Data arrives as separate rows, not as a complete judgement
        # Instead of one row with the full judgement, we get 3 separate rows:
        # Row A: [shown: (1234), endorsed: (1)]
        # Row B: [shown: (1234), endorsed: (23)]
        # Row C: [shown: (1234), endorsed: (4)]
        # These rows might be mixed with other users' judgements!

        # THE SOLUTION: Process each row individually using weighted scoring

        # Processing Row A: [shown: (1234), endorsed: (1)]
        # - We see potential pairs: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        # - We see endorsed pairs: none (single item can't form pairs)
        # - We tentatively reject ALL potential pairs with weight -1/4
        #   (Why 1/4? Because this endorsed group has 1 item out of 4 total items)
        # - Current scores: all pairs = -0.25

        # Processing Row B: [shown: (1234), endorsed: (23)]
        # - Potential pairs: same as before
        # - Endorsed pairs: (2,3)
        # - For endorsed pair (2,3): Add +1, PLUS compensation for negative scoring from
        #   other rows for this judgement (i.e. Row A and Row C)
        #   Compensation = +2/4 (non-endorsed group size / total size)
        #   Final addition: +1 + 0.5 = +1.5
        # - For rejected pairs: Add more negative weight = -2/4 = -0.5
        # - Current scores: (2,3) = -0.25 + 1.5 = +1.25, others = -0.25 - 0.5 = -0.75

        # Processing Row C: [shown: (1234), endorsed: (4)]
        # - Add final negative weight of -1/4 to all potential pairs
        # - Final scores: (2,3) = +1.25 - 0.25 = +1.0, others = -0.75 - 0.25 = -1.0

        # RESULT: Perfect! (2,3) gets +1 (endorsed), rejected pairs get -1 each.
        # The maths works regardless of row order or interleaving with other judgements
        # as long as no partial or splintered clusters are present.

        # As in the example above:
        # Subtract negative weight for all shown pairs not endorsed on this row
        negative_adjustment = len(endorsed) / len(shown)
        validation_net_count.update(
            {
                p: validation_net_count.get(p, 0) - negative_adjustment
                for p in negative_pairs
            }
        )

        # Add 1 plus positive weight for all pairs endorsed on this row
        positive_adjustment = 1 + ((len(shown) - len(endorsed)) / len(shown))
        validation_net_count.update(
            {
                p: validation_net_count.get(p, 0) + positive_adjustment
                for p in positive_pairs
            }
        )

    all_pairs = set(validation_net_count.keys())

    return all_pairs, validation_net_count, all_leaves
