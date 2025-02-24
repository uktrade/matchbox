from functools import lru_cache
from itertools import chain
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from matchbox.common.factories.models import generate_dummy_probabilities
from matchbox.common.transform import (
    attach_components_to_probabilities,
    component_to_hierarchy,
    to_hierarchical_clusters,
)


@lru_cache(maxsize=None)
def _combine_strings(*n: str) -> str:
    """
    Combine n strings into a single string, with a cache.
    Meant to replace `matchbox.common.hash.IntMap.index`

    Args:
        *args: Variable number of strings to combine

    Returns:
        A single string
    """
    letters = set(chain.from_iterable(n))
    return "".join(sorted(letters))


@pytest.mark.parametrize(
    ("parameters"),
    [
        {
            "left_range": (0, 1_000),
            "right_range": (1_000, 2_000),
            "prob_range": (0.6, 0.8),
            "num_components": 10,
            "total_rows": 100_000,
        },
    ],
    ids=["simple"],
)
def test_attach_components_to_probabilities(parameters: dict[str, Any]):
    left_values = tuple(range(*parameters["left_range"]))
    right_values = tuple(range(*parameters["right_range"]))

    probabilities = generate_dummy_probabilities(
        left_values=left_values,
        right_values=right_values,
        prob_range=parameters["prob_range"],
        num_components=parameters["num_components"],
        total_rows=parameters["total_rows"],
    )

    with_components = attach_components_to_probabilities(probabilities=probabilities)

    assert len(pc.unique(with_components["component"])) == parameters["num_components"]


def test_empty_attach_components_to_probabilities():
    probabilities = pa.table(
        {
            "left_id": [],
            "right_id": [],
            "probability": [],
        }
    )

    with_components = attach_components_to_probabilities(probabilities=probabilities)

    assert len(with_components) == 0


@pytest.mark.parametrize(
    ("probabilities", "hierarchy"),
    [
        # Test case 1: Equal probabilities
        (
            {
                "left_id": ["a", "b", "c"],
                "right_id": ["b", "c", "d"],
                "probability": [100, 100, 100],
            },
            {
                ("ab", "a", 100),
                ("ab", "b", 100),
                ("bc", "b", 100),
                ("bc", "c", 100),
                ("cd", "c", 100),
                ("cd", "d", 100),
                ("abcd", "ab", 100),
                ("abcd", "bc", 100),
                ("abcd", "cd", 100),
            },
        ),
        # Test case 2: Asymmetric probabilities
        (
            {
                "left_id": ["w", "x", "y"],
                "right_id": ["x", "y", "z"],
                "probability": [90, 85, 80],
            },
            {
                ("wx", "w", 90),
                ("wx", "x", 90),
                ("xy", "x", 85),
                ("xy", "y", 85),
                ("wxy", "wx", 85),
                ("wxy", "xy", 85),
                ("yz", "y", 80),
                ("yz", "z", 80),
                ("wxyz", "wxy", 80),
                ("wxyz", "yz", 80),
            },
        ),
        # Test case 3: Single two-item component
        (
            {
                "left_id": ["x"],
                "right_id": ["y"],
                "probability": [90],
            },
            {
                ("xy", "x", 90),
                ("xy", "y", 90),
            },
        ),
        # Test case 4: A component larger than two remains unchanged
        # at a successive threshold
        (
            {
                "left_id": ["x", "y", "a"],
                "right_id": ["y", "z", "b"],
                "probability": [90, 90, 85],
            },
            {
                ("xy", "x", 90),
                ("xy", "y", 90),
                ("yz", "y", 90),
                ("yz", "z", 90),
                ("xyz", "xy", 90),
                ("xyz", "yz", 90),
                ("ab", "a", 85),
                ("ab", "b", 85),
            },
        ),
    ],
    ids=["equal", "asymmetric", "single", "unchanged"],
)
def test_component_to_hierarchy(
    probabilities: dict[str, list[str | float]], hierarchy: set[tuple[str, str, int]]
):
    probabilities_table = (
        pa.Table.from_pydict(probabilities)
        .cast(
            pa.schema(
                [
                    ("left_id", pa.string()),
                    ("right_id", pa.string()),
                    ("probability", pa.uint8()),
                ]
            )
        )
        .sort_by([("probability", "descending")])
    )

    parents, children, probs = zip(*hierarchy, strict=False)

    hierarchy_true = (
        pa.table([parents, children, probs], names=["parent", "child", "probability"])
        .cast(
            pa.schema(
                [
                    ("parent", pa.string()),
                    ("child", pa.string()),
                    ("probability", pa.uint8()),
                ]
            )
        )
        .sort_by(
            [
                ("probability", "descending"),
                ("parent", "ascending"),
                ("child", "ascending"),
            ]
        )
        .filter(pc.is_valid(pc.field("parent")))
    )

    hierarchy = component_to_hierarchy(
        table=probabilities_table, dtype=pa.string, hash_func=_combine_strings
    ).sort_by(
        [
            ("probability", "descending"),
            ("parent", "ascending"),
            ("child", "ascending"),
        ]
    )

    assert hierarchy.equals(hierarchy_true)


@pytest.mark.parametrize(
    ("input_data", "expected_hierarchy"),
    [
        # Single component test case
        (
            {
                "component": [1, 1, 1],
                "left_id": ["a", "b", "c"],
                "right_id": ["b", "c", "d"],
                "probability": [90, 85, 80],
            },
            {
                ("ab", "a", 90),
                ("ab", "b", 90),
                ("bc", "b", 85),
                ("bc", "c", 85),
                ("abc", "ab", 85),
                ("abc", "bc", 85),
                ("cd", "c", 80),
                ("cd", "d", 80),
                ("abcd", "abc", 80),
                ("abcd", "cd", 80),
            },
        ),
        # Multiple components test case
        (
            {
                "component": [1, 1, 2, 2],
                "left_id": ["a", "b", "x", "y"],
                "right_id": ["b", "c", "y", "z"],
                "probability": [90, 85, 95, 92],
            },
            {
                ("xy", "x", 95),
                ("xy", "y", 95),
                ("yz", "y", 92),
                ("yz", "z", 92),
                ("xyz", "xy", 92),
                ("xyz", "yz", 92),
                ("ab", "a", 90),
                ("ab", "b", 90),
                ("bc", "b", 85),
                ("bc", "c", 85),
                ("abc", "ab", 85),
                ("abc", "bc", 85),
            },
        ),
    ],
    ids=["single_component", "multiple_components"],
)
def test_hierarchical_clusters(input_data, expected_hierarchy):
    # Create input table
    probabilities = pa.table(
        input_data,
        schema=pa.schema(
            [
                ("component", pa.uint64()),
                ("left_id", pa.string()),
                ("right_id", pa.string()),
                ("probability", pa.uint8()),
            ]
        ),
    )

    # Create expected output table
    parents, children, probs = zip(*expected_hierarchy, strict=False)
    expected = (
        pa.table([parents, children, probs], names=["parent", "child", "probability"])
        .cast(
            pa.schema(
                [
                    ("parent", pa.string()),
                    ("child", pa.string()),
                    ("probability", pa.uint8()),
                ]
            )
        )
        .sort_by(
            [
                ("probability", "descending"),
                ("parent", "ascending"),
                ("child", "ascending"),
            ]
        )
    )

    # Run and compare
    result = to_hierarchical_clusters(
        probabilities,
        dtype=pa.string,
        proc_func=component_to_hierarchy,
        hash_func=_combine_strings,
    )

    result = result.sort_by(
        [
            ("probability", "descending"),
            ("parent", "ascending"),
            ("child", "ascending"),
        ]
    )

    assert result.schema == expected.schema
    assert result.equals(expected)
