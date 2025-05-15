import polars as pl
import pytest

from matchbox.common.eval import Pairs, contains_to_pairs, precision_recall


def test_contains_to_pairs():
    contains = pl.DataFrame(
        [
            # Source clusters
            [1, None],
            [2, None],
            [3, None],
            # Model cluster
            [4, 1],
            [4, 2],
            [4, 3],
            # Model cluster (order invariance)
            [5, 3],
            [5, 1],
            [5, 2],
        ],
        schema=["parent", "leaf"],
    )

    expected_pairs = {(1, 2), (2, 3), (1, 3)}

    pairs = contains_to_pairs(contains)

    assert pairs == expected_pairs


@pytest.mark.parametrize(
    ["model_pairs", "eval_pairs", "prec", "recall"],
    [
        [
            {(1, 2), (1, 3), (2, 3)},
            {(1, 2)},
            1 / 3,
            1,
        ],
        [
            {(1, 2)},
            {(1, 2), (1, 3), (2, 3)},
            1,
            1 / 3,
        ],
    ],
    ids=["sub_precision", "sub_recall"],
)
def test_precision_recall(
    model_pairs: Pairs, eval_pairs: Pairs, prec: float, recall: float
):
    p, r = precision_recall(model_pairs=model_pairs, eval_pairs=eval_pairs)
    assert p == prec
    assert r == recall
