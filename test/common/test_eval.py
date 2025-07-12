import polars as pl
import pytest

from matchbox.common.arrow import SCHEMA_JUDGEMENTS
from matchbox.common.eval import Pairs, contains_to_pairs, precision_recall


def test_contains_to_pairs():
    contains = pl.DataFrame(
        [
            # -- First user --
            # Source clusters
            [1, 1, None],
            [1, 2, None],
            [1, 3, None],
            # Model cluster
            [1, 4, 1],
            [1, 4, 2],
            [1, 4, 3],
            # Model cluster (order invariance)
            [1, 5, 3],
            [1, 5, 1],
            [1, 5, 2],
        ],
        schema=SCHEMA_JUDGEMENTS.names,
    )

    expected_pairs = {(1, 2), (2, 3), (1, 3)}

    pairs_per_user = contains_to_pairs(contains)

    assert set(pairs_per_user.keys()) == {1}
    assert pairs_per_user[1] == expected_pairs


@pytest.mark.parametrize(
    ["model_pairs", "eval_pairs_by_user", "prec", "recall"],
    [
        [
            {(1, 2), (1, 3), (2, 3)},  # model
            {1: {(1, 2)}},  # user
            1,  # precision
            pytest.approx(1 / 3, 0.001),  # recall
        ],
        [
            {(1, 2)},  # model
            {1: {(1, 2), (1, 3), (2, 3)}},  # user
            pytest.approx(1 / 3, 0.001),  # precision
            1,  # recall
        ],
        [
            {(1, 2), (1, 3), (2, 3)},  # model
            {1: {(1, 2)}},  # user, who's never seen 3
            1,  # precision
            pytest.approx(1 / 3, 0.001),  # recall
        ],
        [
            {(1, 2)},  # model
            {
                1: {(1, 2)},  # user 1
                2: {(3, 4)},  # user 2
            },
            0.5,  # precision: mean of 1 and 0
            0.5,  # recall: mean of 1 and 0
        ],
    ],
    ids=["sub_recall", "sub_precision", "unseen_clusters", "multi_user"],
)
def test_precision_recall(
    model_pairs: Pairs, eval_pairs_by_user: dict[int, Pairs], prec: float, recall: float
):
    p, r = precision_recall(
        model_pairs=model_pairs, eval_pairs_by_user=eval_pairs_by_user
    )
    assert p == prec
    assert r == recall
