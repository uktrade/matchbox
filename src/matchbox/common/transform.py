"""Functions to transform data between tabular and graph structures."""

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from typing import Generic, Self, TypeVar

import numpy as np
import polars as pl
import rustworkx as rx

from matchbox.common.hash import HASH_FUNC, IntMap, hash_values

T = TypeVar("T", bound=Hashable)


def to_clusters(
    results: pl.DataFrame,
    dtype: pl.DataType = pl.Int64,
    hash_func: Callable[[*tuple[T, ...]], T] = hash_values,
) -> pl.DataFrame:
    """Converts probabilities into connected components formed at each threshold.

    Args:
        results: Polars dataframe with columns ['left_id', 'right_id', 'probability']
        dtype: Polars data type for the parent and child columns
        hash_func: Function to hash the parent and child values

    Returns:
        Polars dataframe of parent, child, threshold, sorted by probability descending.
    """
    G = rx.PyGraph()
    added: dict[bytes, int] = {}
    components: dict[str, list] = {"parent": [], "child": [], "threshold": []}

    # Sort probabilities descending and select relevant columns
    results = results.sort(by="probability", descending=True)
    results = results.select(["left_id", "right_id", "probability"])

    # Get unique probability thresholds, sorted
    thresholds = results["probability"].unique().sort(descending=True)

    # Process edges grouped by probability threshold
    for prob in thresholds:
        threshold_edges = results.filter(pl.col("probability") == prob)

        # Get state before adding this batch of edges
        old_components = {frozenset(comp) for comp in rx.connected_components(G)}

        # Add all nodes and edges at this probability threshold
        edge_values = threshold_edges.select(["left_id", "right_id"]).rows()

        for left, right in edge_values:
            for hash_val in (left, right):
                if hash_val not in added:
                    idx = G.add_node(hash_val)
                    added[hash_val] = idx

            G.add_edge(added[left], added[right], None)

        new_components = {frozenset(comp) for comp in rx.connected_components(G)}
        changed_components = new_components - old_components

        # For each changed component, add ALL members at current threshold
        for comp in changed_components:
            children = [G.get_node_data(n) for n in comp]
            parent = hash_func(*children)

            components["parent"].extend([parent] * len(children))
            components["child"].extend(children)
            components["threshold"].extend([prob] * len(children))

    return pl.from_dict(
        components,
        schema=[("parent", dtype), ("child", dtype), ("threshold", pl.UInt8)],
    )


def graph_results(
    probabilities: pl.DataFrame, all_nodes: Iterable[int] | None = None
) -> tuple[rx.PyDiGraph, np.ndarray, np.ndarray]:
    """Convert probability table to graph representation.

    Args:
        probabilities: Polars dataframe with 'left_id', 'right_id' columns
        all_nodes: superset of node identities figuring in probabilities table.
            Used to optionally add isolated nodes to the graph.

    Returns:
        A tuple containing:
        - Rustwork directed graph
        - A list mapping the 'left_id' probabilities to node indices in the graph
        - A list mapping the 'right_id' probabilities to node indices in the graph
    """
    # Create index to use in graph
    unique = (
        pl.concat([probabilities["left_id"], probabilities["right_id"]], rechunk=True)
        .unique(maintain_order=True)
        .rename("unique_ids")
        .to_frame()
        .with_row_index()
    )

    left_indices = (
        probabilities.select("left_id")
        .join(unique, left_on="left_id", right_on="unique_ids", maintain_order="left")
        .to_series(1)
        .to_numpy()
    )
    right_indices = (
        probabilities.select("right_id")
        .join(unique, left_on="right_id", right_on="unique_ids", maintain_order="left")
        .to_series(1)
        .to_numpy()
    )

    # Create and process graph
    n_nodes = len(unique)
    n_edges = len(probabilities)

    graph = rx.PyGraph(node_count_hint=n_nodes, edge_count_hint=n_edges)
    graph.add_nodes_from(range(n_nodes))

    if all_nodes is not None:
        isolated_nodes = len(set(all_nodes) - set(unique["unique_ids"].to_list()))
        graph.add_nodes_from(range(isolated_nodes))

    edges = tuple(zip(left_indices, right_indices, strict=True))
    graph.add_edges_from_no_data(edges)

    return graph, left_indices, right_indices


class DisjointSet(Generic[T]):
    """Disjoint set forest with "path compression" and "union by rank" heuristics.

    This follows implementation from Cormen, Thomas H., et al. Introduction to
    algorithms. MIT press, 2022
    """

    def __init__(self) -> None:
        """Initialize the disjoint set."""
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}

    def _make_set(self, x: T) -> None:
        """Create a new set with a single element x."""
        self.parent[x] = x
        self.rank[x] = 0

    def add(self, x: T) -> None:
        """Add a new element to the disjoint set."""
        if x not in self.parent:
            self._make_set(x)

    def union(self, x: T, y: T) -> None:
        """Merge the sets containing elements x and y."""
        self._link(self._find(x), self._find(y))

    def _link(self, x: T, y: T) -> None:
        """Merge the sets containing elements x and y."""
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    def _find(self, x: T) -> T:
        """Return the representative element of the set containing x."""
        if x not in self.parent:
            self._make_set(x)
            return x

        if x != self.parent[x]:
            self.parent[x] = self._find(self.parent[x])

        return self.parent[x]

    def get_components(self) -> list[set[T]]:
        """Return the connected components of the disjoint set."""
        components = defaultdict(set)
        for x in self.parent:
            root = self._find(x)
            components[root].add(x)
        return list(components.values())


def hash_cluster_leaves(leaves: list[bytes]) -> bytes:
    """Canonical method to convert list of cluster IDs to their combined hash."""
    return HASH_FUNC(b"|".join(leaf for leaf in sorted(leaves))).digest()


class Cluster:
    """A cluster of connected components.

    Can be a source cluster (a single data point) or a model cluster (a cluster of
    clusters). The hash of a cluster is the hash of its source clusters -- its leaves.

    We generate negative integers for IDs, allowing us to generate a true ID with
    the database after we've calculated components using these objects.
    """

    id: int
    probability: int | None
    hash: bytes
    leaves: tuple["Cluster"] | None

    _intmap: IntMap  # Reference to the IntMap singleton

    def __init__(
        self,
        intmap: IntMap,
        probability: int | None = None,
        leaves: tuple["Cluster"] | list["Cluster"] | None = None,
        id: int | None = None,
        hash: bytes | None = None,
    ) -> None:
        """Initialise the Cluster.

        Args:
            intmap: An IntMap instance for generating unique IDs
            probability: probability of the cluster from its resolution, or None if
                source
            leaves: A list of Cluster objects that are the leaves of this cluster
            id: The ID of the cluster (only for leaf nodes)
            hash: The hash of the cluster (only for leaf nodes)
        """
        self._intmap = intmap
        self.probability = probability

        if leaves:
            self.leaves = tuple(sorted(leaves, key=lambda leaf: leaf.hash))
        else:
            self.leaves = None

        # Set hash - use provided hash or calculate
        if hash is not None:
            self.hash = hash
        elif self.leaves is None:
            raise ValueError("Leaf nodes must have hash specified")
        else:
            self.hash = hash_cluster_leaves([leaf.hash for leaf in self.leaves])

        # Set ID - use provided ID or calculate
        if id is not None:
            self.id = id
        elif self.leaves is None:
            raise ValueError("Leaf nodes must have id specified")
        else:
            self.id = intmap.index(leaf.id for leaf in self.leaves)

    def __hash__(self) -> int:
        """Return a hash of the cluster based on its hash bytes."""
        return hash(self.hash)

    def __eq__(self, other: object) -> bool:
        """Check equality of two Cluster objects based on their hash."""
        if not isinstance(other, Cluster):
            return False
        return self.hash == other.hash

    @classmethod
    def combine(
        cls: type[Self],
        clusters: Iterable["Cluster"],
        probability: int | None = None,
    ) -> "Cluster":
        """Efficiently combine multiple clusters at once.

        Args:
            clusters: An iterable of Cluster objects to combine
            probability: the probability of the cluster from its resolution

        Returns:
            A new Cluster containing all unique leaves from the input clusters
        """
        clusters = list(clusters)
        if len(clusters) == 1:
            new_probability = (
                clusters[0].probability if probability is None else probability
            )
            return cls(
                intmap=clusters[0]._intmap,
                probability=new_probability,
                leaves=clusters[0].leaves,
                id=clusters[0].id,
                hash=clusters[0].hash,
            )

        intmap = clusters[0]._intmap
        unique_dict: dict[int, Cluster] = {}

        for cluster in clusters:
            if cluster.leaves is None:
                unique_dict[id(cluster)] = cluster
            else:
                for leaf in cluster.leaves:
                    unique_dict[id(leaf)] = leaf

        return cls(
            intmap=intmap,
            probability=probability,
            leaves=list(unique_dict.values()),
        )


def threshold_float_to_int(threshold: float) -> int:
    """Convert user input float truth values to int."""
    if isinstance(threshold, float) and 0.0 <= threshold <= 1.0:
        return round(threshold * 100)
    else:
        raise ValueError(f"Truth value {threshold} not a valid probability")


def threshold_int_to_float(threshold: int) -> float:
    """Convert backend int truth values to float."""
    if isinstance(threshold, int) and 0 <= threshold <= 100:
        return float(threshold / 100)
    else:
        raise ValueError(f"Truth value {threshold} not valid")
