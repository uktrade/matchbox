from typing import Any

import numpy as np
import polars as pl
import pytest

from matchbox.common.arrow import SCHEMA_RESULTS
from matchbox.common.factories.entities import (
    ClusterEntity,
    EntityReference,
    SourceEntity,
)
from matchbox.common.factories.models import (
    calculate_min_max_edges,
    component_report,
    generate_dummy_probabilities,
    generate_entity_probabilities,
)
from matchbox.common.transform import DisjointSet
from test.common.factories.test_entity_factory import (
    make_cluster_entity,
    make_source_entity,
)


@pytest.mark.parametrize(
    ("left_n", "right_n", "n_components", "true_min", "true_max"),
    [
        (10, None, 2, 8, 20),
        (11, None, 2, 9, 25),
        (9, 9, 3, 15, 27),
        (8, 4, 3, 9, 11),
        (4, 8, 3, 9, 11),
        (8, 8, 3, 13, 22),
    ],
    ids=[
        "dedupe_no_mod",
        "dedup_mod",
        "link_no_mod",
        "link_left_mod",
        "link_right_mod",
        "link_same_mod",
    ],
)
def test_calculate_min_max_edges(
    left_n: int,
    right_n: int | None,
    n_components: int,
    true_min: int,
    true_max: int,
) -> None:
    deduplicate = False
    if not right_n:
        deduplicate = True
        right_n = left_n
    min_edges, max_edges = calculate_min_max_edges(
        left_n, right_n, n_components, deduplicate
    )

    assert true_min == min_edges
    assert true_max == max_edges


@pytest.mark.parametrize(
    ("parameters"),
    [
        pytest.param(
            {
                "left_count": 5,
                "right_count": None,
                "prob_range": (0.6, 0.8),
                "num_components": 3,
                "total_rows": 2,
            },
            id="dedupe_no_edges",
        ),
        pytest.param(
            {
                "left_count": 1_000,
                "right_count": None,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1_000, 1_000, 10, True)[0],
            },
            id="dedupe_min",
        ),
        pytest.param(
            {
                "left_count": 1_000,
                "right_count": None,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1_000, 1_000, 10, True)[1],
            },
            id="dedupe_max",
        ),
        pytest.param(
            {
                "left_count": 1_000,
                "right_count": 1_000,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1_000, 1_000, 10, False)[0],
            },
            id="link_min",
        ),
        pytest.param(
            {
                "left_count": 1_000,
                "right_count": 1_000,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": calculate_min_max_edges(1_000, 1_000, 10, False)[1],
            },
            id="link_max",
        ),
        pytest.param(
            {
                "left_count": 1_000,
                "right_count": 1_000,
                "prob_range": (0.6, 0.8),
                "num_components": 10,
                "total_rows": None,
            },
            id="blank_total_rows",
        ),
    ],
)
def test_generate_dummy_probabilities(parameters: dict[str, Any]) -> None:
    len_left = parameters["left_count"]
    len_right = parameters["right_count"]
    if len_right:
        total_len = len_left + len_right
        len_right = parameters["right_count"]
        rand_vals = np.random.choice(a=total_len, replace=False, size=total_len)
        left_values = tuple(rand_vals[:len_left])
        right_values = tuple(rand_vals[len_left:])
    else:
        rand_vals = np.random.choice(a=len_left, replace=False, size=len_left)
        left_values = tuple(rand_vals[:len_left])
        right_values = None

    n_components = parameters["num_components"]
    total_rows = parameters["total_rows"]
    min_edges, _ = calculate_min_max_edges(
        parameters["left_count"],
        parameters["right_count"] or parameters["left_count"],
        parameters["num_components"],
        right_values is None,
    )
    total_rows = total_rows or min_edges

    probabilities = generate_dummy_probabilities(
        left_values=left_values,
        right_values=right_values,
        prob_range=parameters["prob_range"],
        num_components=n_components,
        total_rows=total_rows,
    )
    report = component_report(table=probabilities, all_nodes=rand_vals)
    p_left = probabilities["left_id"].to_list()
    p_right = probabilities["right_id"].to_list()

    assert report["num_components"] == n_components

    # Link job
    if right_values:
        assert set(p_left) <= set(left_values)
        assert set(p_right) <= set(right_values)
    # Dedupe
    else:
        assert set(p_left) | set(p_right) <= set(left_values)

    assert probabilities["probability"].max() / 100 <= parameters["prob_range"][1]
    assert probabilities["probability"].min() / 100 >= parameters["prob_range"][0]

    assert len(probabilities) == total_rows

    edges = zip(p_left, p_right, strict=True)
    edges_set = {tuple(sorted(e)) for e in edges}
    assert len(edges_set) == total_rows

    self_references = [e for e in edges if e[0] == e[1]]
    assert len(self_references) == 0


def test_generate_dummy_probabilities_no_self_references() -> None:
    # Create input with repeated values
    left_values = tuple([1] * 4 + [2] * 4 + [3] * 4)

    try:
        probabilities = generate_dummy_probabilities(
            left_values=left_values,
            right_values=None,
            prob_range=(0.6, 0.8),
            num_components=3,
            total_rows=3,
        )
    except ValueError:
        return

    # If no ValueError was raised, continue with the rest of the checks
    p_left = probabilities["left_id"].to_list()
    p_right = probabilities["right_id"].to_list()

    # Check for self-references
    self_references = [
        (l_, r_) for l_, r_ in zip(p_left, p_right, strict=False) if l_ == r_
    ]
    assert len(self_references) == 0, f"Found self-references: {self_references}"


@pytest.mark.parametrize(
    ("parameters"),
    [
        {
            "left_range": (0, 10_000),
            "right_range": (10_000, 20_000),
            "num_components": 2,
            "total_rows": 1,
        },
        {
            "left_range": (0, 10),
            "right_range": (10, 20),
            "num_components": 2,
            "total_rows": 8_000,
        },
    ],
    ids=["lower_than_min", "higher_than_max"],
)
def test_generate_dummy_probabilities_errors(parameters: dict[str, Any]) -> None:
    left_values = tuple(range(*parameters["left_range"]))
    right_values = tuple(range(*parameters["right_range"]))

    with pytest.raises(ValueError):
        generate_dummy_probabilities(
            left_values=left_values,
            right_values=right_values,
            prob_range=(0.6, 0.8),
            num_components=parameters["num_components"],
            total_rows=parameters["total_rows"],
        )


@pytest.mark.parametrize(
    ("left_entities", "right_entities", "source_entities", "prob_range", "expected"),
    [
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["a2"]),
                ]
            ),
            None,  # Deduplication case
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.8, 1.0),
            {"edge_count": 1, "prob_range": (80, 100)},
            id="basic_dedupe",
        ),
        pytest.param(
            frozenset([make_cluster_entity(1, "left", ["a1"])]),
            frozenset([make_cluster_entity(2, "right", ["b1"])]),
            frozenset(
                [
                    make_source_entity("left", ["a1"], "a"),
                    make_source_entity("right", ["b1"], "b"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 0, "prob_range": (80, 100)},
            id="basic_link_no_match",
        ),
        pytest.param(
            frozenset([make_cluster_entity(1, "test", ["a1"])]),
            frozenset([make_cluster_entity(2, "test", ["a2"])]),
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.8, 1.0),
            {"edge_count": 1, "prob_range": (80, 100)},
            id="successful_link",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1", "a2"]),
                    make_cluster_entity(2, "test", ["a2", "a3"]),
                    make_cluster_entity(3, "test", ["a3", "a4"]),
                ]
            ),
            None,
            frozenset(
                [make_source_entity("test", ["a1", "a2", "a3", "a4"], "entity_a")]
            ),
            (0.8, 1.0),
            {"edge_count": 3, "prob_range": (80, 100)},
            id="overlapping_dedupe",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["b1"]),
                ]
            ),
            frozenset(
                [
                    make_cluster_entity(3, "test", ["a2"]),
                    make_cluster_entity(4, "test", ["b2"]),
                ]
            ),
            frozenset(
                [
                    make_source_entity("test", ["a1", "a2"], "a"),
                    make_source_entity("test", ["b1", "b2"], "b"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 2, "prob_range": (80, 100)},
            id="multi_component_link",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["a2"]),
                    make_cluster_entity(3, "test", ["x1"]),  # No source for this
                    make_cluster_entity(4, "test", ["y1"]),  # No source for this
                ]
            ),
            None,
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.8, 1.0),
            {"edge_count": 1, "prob_range": (80, 100)},
            id="partial_source_coverage",
        ),
        pytest.param(
            frozenset(),
            frozenset(),
            frozenset(),
            (0.8, 1.0),
            {"edge_count": 0, "prob_range": (80, 100)},
            id="empty_sets",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1"]),
                    make_cluster_entity(2, "test", ["a2"]),
                ]
            ),
            None,
            frozenset([make_source_entity("test", ["a1", "a2"], "a")]),
            (0.5, 0.7),
            {"edge_count": 1, "prob_range": (50, 70)},
            id="different_prob_range",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "test", ["a1", "a2"]),  # Merged
                    make_cluster_entity(2, "test", ["a3"]),  # Unmerged
                    make_cluster_entity(3, "test", ["b1"]),  # From different source
                ]
            ),
            None,
            frozenset(
                [
                    make_source_entity("test", ["a1", "a2", "a3"], "a"),
                    make_source_entity("test", ["b1"], "b"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 1, "prob_range": (80, 100)},
            id="mixed_merged_unmerged",
        ),
        pytest.param(
            frozenset(
                [
                    make_cluster_entity(1, "source1", ["1"]),
                    make_cluster_entity(2, "source2", ["A"]),
                    make_cluster_entity(3, "source3", ["X"]),
                ]
            ),
            None,
            frozenset(
                [
                    make_source_entity("source1", ["1"], "entity"),
                    make_source_entity("source2", ["A"], "entity"),
                    make_source_entity("source3", ["X"], "entity"),
                ]
            ),
            (0.8, 1.0),
            {"edge_count": 0, "prob_range": (80, 100)},
            id="multi_source_entity",
        ),
    ],
)
def test_generate_entity_probabilities_scenarios(
    left_entities: frozenset[ClusterEntity],
    right_entities: frozenset[ClusterEntity] | None,
    source_entities: frozenset[SourceEntity],
    prob_range: tuple[float, float],
    expected: dict,
) -> None:
    """Comprehensive test for generate_entity_probabilities with various scenarios."""
    # Run the function
    result = generate_entity_probabilities(
        left_entities, right_entities, source_entities, prob_range
    )

    # Check schema
    assert result.schema == pl.Schema(SCHEMA_RESULTS)

    # Get edges from result
    edges = list(
        zip(
            result["left_id"].to_list(),
            result["right_id"].to_list(),
            strict=True,
        )
    )

    # Check number of edges matches expected
    assert len(edges) == expected["edge_count"]

    # For non-empty results, validate probability ranges
    if edges:
        probs = result["probability"].to_numpy()
        prob_min, prob_max = expected["prob_range"]
        assert all(prob_min <= p <= prob_max for p in probs)


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal", "case"),
    [
        pytest.param(42, 42, True, "dedupe", id="same_seeds_dedupe"),
        pytest.param(1, 2, False, "dedupe", id="different_seeds_dedupe"),
        pytest.param(42, 42, True, "link", id="same_seeds_link"),
        pytest.param(1, 2, False, "link", id="different_seeds_link"),
    ],
)
def test_seed_determinism(
    seed1: int,
    seed2: int,
    should_be_equal: bool,
    case: str,
) -> None:
    """Test that seeds produce consistent/different results as expected."""
    # Create test entities
    source = make_source_entity("test", ["a1", "a2", "a3"], "entity_a")
    entities = frozenset(
        [
            make_cluster_entity(1, "test", ["a1"]),
            make_cluster_entity(2, "test", ["a2"]),
            make_cluster_entity(3, "test", ["a3"]),
        ]
    )

    if case == "dedupe":
        right_entities = None
    else:
        # For linking case, use second set of entities
        right_entities = frozenset(
            [
                make_cluster_entity(4, "test", ["a1"]),
                make_cluster_entity(5, "test", ["a2"]),
                make_cluster_entity(6, "test", ["a3"]),
            ]
        )

    # Generate results with the two seeds
    result1 = generate_entity_probabilities(
        left_entities=entities,
        right_entities=right_entities,
        source_entities=frozenset([source]),
        prob_range=(0.8, 1.0),
        seed=seed1,
    )

    result2 = generate_entity_probabilities(
        left_entities=entities,
        right_entities=right_entities,
        source_entities=frozenset([source]),
        prob_range=(0.8, 1.0),
        seed=seed2,
    )

    assert result1.shape[0] > 0
    assert result2.shape[0] > 0

    if should_be_equal:
        assert result1.equals(result2)
    else:
        assert not result1.equals(result2)


def test_disjoint_set_recovery() -> None:
    """Test that DisjointSet can recover the entity structure from probabilities."""
    # Create source entities
    source1 = make_source_entity("source1", ["1", "2", "3"], "entity1")
    source2 = make_source_entity("source1", ["4", "5", "6"], "entity2")

    # Create split cluster entities
    clusters = frozenset(
        [
            make_cluster_entity(1, "source1", ["1"]),
            make_cluster_entity(2, "source1", ["2"]),
            make_cluster_entity(3, "source1", ["3"]),
            make_cluster_entity(4, "source1", ["4"]),
            make_cluster_entity(5, "source1", ["5"]),
            make_cluster_entity(6, "source1", ["6"]),
        ]
    )

    # Generate probabilities
    table = generate_entity_probabilities(
        left_entities=clusters,
        right_entities=None,
        source_entities=frozenset([source1, source2]),
        prob_range=(0.9, 1.0),
    )

    # Use DisjointSet to cluster based on high probabilities
    ds = DisjointSet[int]()
    for row in table.to_dicts():
        if row["probability"] >= 90:  # High confidence matches
            ds.union(row["left_id"], row["right_id"])

    # Get resulting clusters
    clusters = ds.get_components()

    # Should recover original entities - exactly two clusters
    assert len(clusters) == 2

    # Each cluster should contain the right number of ClusterEntity objects
    cluster_sizes = sorted(len(cluster) for cluster in clusters)
    assert cluster_sizes == [3, 3]


@pytest.mark.parametrize(
    "prob_range",
    [
        pytest.param((-0.1, 0.5), id="negative_lower_bound"),  # Negative lower bound
        pytest.param((0.5, 1.1), id="upper_bound_too_high"),  # Upper bound > 1.0
        pytest.param((0.8, 0.7), id="decreasing_range"),  # Decreasing range
    ],
)
def test_invalid_probability_ranges(prob_range: tuple[float, float]) -> None:
    """Test that invalid probability ranges raise appropriate errors."""
    source = make_source_entity("test", ["a1", "a2"], "entity")
    entities = frozenset(
        [
            make_cluster_entity(1, "test", ["a1"]),
            make_cluster_entity(2, "test", ["a2"]),
        ]
    )

    with pytest.raises(ValueError, match="Probabilities must be"):
        generate_entity_probabilities(
            left_entities=entities,
            right_entities=None,
            source_entities=frozenset([source]),
            prob_range=prob_range,
        )


def test_complex_entity_recovery() -> None:
    """Test recovery of complex, multi-source entity structures."""
    # Create a source entity spanning multiple sources
    source = SourceEntity(
        base_values={"name": "Complex Entity"},
        keys=EntityReference(
            {
                "source1": frozenset(["1", "2"]),
                "source2": frozenset(["A", "B"]),
                "source3": frozenset(["X"]),
            }
        ),
    )

    # Create fragmented ClusterEntity objects
    clusters = frozenset(
        [
            ClusterEntity(keys=EntityReference({"source1": frozenset(["1"])})),
            ClusterEntity(keys=EntityReference({"source1": frozenset(["2"])})),
            ClusterEntity(keys=EntityReference({"source2": frozenset(["A"])})),
            ClusterEntity(keys=EntityReference({"source2": frozenset(["B"])})),
            ClusterEntity(keys=EntityReference({"source3": frozenset(["X"])})),
        ]
    )

    # Generate probabilities
    table = generate_entity_probabilities(
        left_entities=clusters,
        right_entities=None,
        source_entities=frozenset([source]),
        prob_range=(0.9, 1.0),
    )

    # There should be edges connecting all entities (n*(n-1))/2 = 10 edges
    assert len(table) == 10

    # Use DisjointSet to cluster
    ds = DisjointSet[int]()
    for row in table.to_dicts():
        if row["probability"] >= 90:
            ds.union(row["left_id"], row["right_id"])

    # Should recover as a single component
    clusters = ds.get_components()
    assert len(clusters) == 1
    assert len(clusters[0]) == 5  # All entities in one cluster
