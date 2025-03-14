"""Functions to transform data between tabular and graph structures."""

import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Generic, Hashable, Iterable, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rustworkx as rx

from matchbox.common.hash import hash_values
from matchbox.common.logging import build_progress_bar, logger

T = TypeVar("T", bound=Hashable)


def to_clusters(
    results: pa.Table,
    dtype: pa.DataType = pa.binary,
    hash_func: Callable[[*tuple[T, ...]], T] = hash_values,
) -> pa.Table:
    """Converts probabilities into connected components formed at each threshold.

    Args:
        results: Arrow table with columns ['left_id', 'right_id', 'probability']
        dtype: Arrow data type for the parent and child columns
        hash_func: Function to hash the parent and child values

    Returns:
        Arrow table of parent, child, threshold, sorted by probability descending.
    """
    G = rx.PyGraph()
    added: dict[bytes, int] = {}
    components: dict[str, list] = {"parent": [], "child": [], "threshold": []}

    # Sort probabilities descending and select relevant columns
    results = results.sort_by([("probability", "descending")])
    results = results.select(["left_id", "right_id", "probability"])

    # Get unique probability thresholds, sorted
    thresholds = pc.unique(results.column("probability")).sort(order="descending")

    # Process edges grouped by probability threshold
    for prob in thresholds:
        threshold_edges = results.filter(pc.equal(results.column("probability"), prob))

        # Get state before adding this batch of edges
        old_components = {frozenset(comp) for comp in rx.connected_components(G)}

        # Add all nodes and edges at this probability threshold
        edge_values = list(
            zip(
                threshold_edges.column("left_id").to_pylist(),
                threshold_edges.column("right_id").to_pylist(),
                strict=True,
            )
        )
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

    return pa.Table.from_pydict(
        components,
        schema=pa.schema(
            [("parent", dtype()), ("child", dtype()), ("threshold", pa.uint8())]
        ),
    )


def graph_results(
    probabilities: pa.Table, all_nodes: Iterable[int] | None = None
) -> tuple[rx.PyDiGraph, np.ndarray, np.ndarray]:
    """Convert probability table to graph representation.

    Args:
        probabilities: PyArrow table with 'left_id', 'right_id' columns
        all_nodes: superset of node identities figuring in probabilities table.
            Used to optionally add isolated nodes to the graph.

    Returns:
        A tuple containing:
        - Rustwork directed graph
        - A list mapping the 'left_id' probabilities to node indices in the graph
        - A list mapping the 'right_id' probabilities to node indices in the graph
    """
    # Create index to use in graph
    unique = pc.unique(
        pa.concat_arrays(
            [
                probabilities["left_id"].combine_chunks(),
                probabilities["right_id"].combine_chunks(),
            ]
        )
    )

    left_indices = pc.index_in(probabilities["left_id"], unique).to_numpy()
    right_indices = pc.index_in(probabilities["right_id"], unique).to_numpy()

    # Create and process graph
    n_nodes = len(unique)
    n_edges = len(probabilities)

    graph = rx.PyGraph(node_count_hint=n_nodes, edge_count_hint=n_edges)
    graph.add_nodes_from(range(n_nodes))

    if all_nodes is not None:
        isolated_nodes = len(set(all_nodes) - set(unique.to_pylist()))
        graph.add_nodes_from(range(isolated_nodes))

    edges = tuple(zip(left_indices, right_indices, strict=True))
    graph.add_edges_from_no_data(edges)

    return graph, left_indices, right_indices


def attach_components_to_probabilities(probabilities: pa.Table) -> pa.Table:
    """Takes an Arrow table of probabilities and adds a component column.

    Expects an Arrow table of column left_id, right_id, probability.

    Returns a table with an additional column, component.
    """
    # Handle empty probabilities
    if len(probabilities) == 0:
        empty_components = pa.array([], type=pa.int64())
        return probabilities.append_column("component", empty_components)

    graph, left_indices, _ = graph_results(probabilities)
    components = rx.connected_components(graph)

    # Convert components to arrays, map back to input to join, and reattach
    component_indices = np.concatenate([np.array(list(c)) for c in components])
    component_labels = np.repeat(
        np.arange(len(components)), [len(c) for c in components]
    )

    node_to_component = np.zeros(graph.num_nodes(), dtype=np.int64)
    node_to_component[component_indices] = component_labels

    edge_components = pa.array(node_to_component[left_indices])

    return probabilities.append_column("component", edge_components).sort_by(
        [("component", "ascending"), ("probability", "descending")]
    )


class DisjointSet(Generic[T]):
    """Disjoint set forest with "path compression" and "union by rank" heuristics.

    This follows implementation from Cormen, Thomas H., et al. Introduction to
    algorithms. MIT press, 2022
    """

    def __init__(self):
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


def component_to_hierarchy(
    table: pa.Table,
    dtype: pa.DataType = pa.binary,
    hash_func: Callable[[*tuple[T, ...]], T] = hash_values,
) -> pa.Table:
    """Convert pairwise probabilities into a hierarchical representation.

    Assumes data is pre-sorted by probability descending.

    Args:
        table: Arrow Table with columns ['left', 'right', 'probability']
        dtype: Arrow data type for the parent and child columns
        hash_func: Function to hash the parent and child values

    Returns:
        Arrow Table with columns ['parent', 'child', 'probability']
    """
    probs = pc.unique(table["probability"]).sort(order="descending")

    djs = DisjointSet[int]()  # implements connected components
    current_roots: dict[int, set[int]] = defaultdict(set)  # tracks ultimate parents
    hierarchy: list[tuple[int, int, float]] = []  # the output of this function
    seen_components: set[frozenset[int]] = set()  # track previously seen component sets

    table_by_p = defaultdict(list)
    for batch in table.to_batches():
        d = batch.to_pydict()
        for left, right, p in zip(
            d["left_id"], d["right_id"], d["probability"], strict=False
        ):
            table_by_p[p].append([left, right])

    for threshold in probs.to_numpy():
        for left, right in table_by_p[threshold]:
            djs.union(left, right)
            parent = hash_func(left, right)
            hierarchy.extend([(parent, left, threshold), (parent, right, threshold)])
            current_roots[left].add(parent)
            current_roots[right].add(parent)

        for children in djs.get_components():
            if len(children) <= 2:
                continue  # Skip pairs already handled by pairwise probabilities

            # Skip if we've seen this exact component before
            frozen_children = frozenset(children)
            if frozen_children in seen_components:
                continue

            seen_components.add(frozen_children)

            parent = hash_func(*children)
            prev_roots: set[int] = set()
            for child in children:
                prev_roots.update(current_roots[child])
                current_roots[child] = {parent}

            for r in prev_roots:
                hierarchy.append((parent, r, threshold))

    parents, children, probs = zip(*hierarchy, strict=True)
    return pa.table(
        {
            "parent": pa.array(parents, type=dtype()),
            "child": pa.array(children, type=dtype()),
            "probability": pa.array(probs, type=pa.uint8()),
        }
    )


def to_hierarchical_clusters(
    probabilities: pa.Table,
    proc_func: Callable[[pa.Table, pa.DataType], pa.Table] = component_to_hierarchy,
    hash_func: Callable[[*tuple[T, ...]], T] = hash_values,
    dtype: pa.DataType = pa.binary,
    parallel: bool = True,
    timeout: int = 300,
    min_rows_per_worker: int = 1000,
) -> pa.Table:
    """Converts a table of pairwise probabilities into a table of hierarchical clusters.

    Args:
        probabilities: Arrow table with columns ['component', 'left', 'right',
            'probability']
        proc_func: Function to process each component
        hash_func: Function to hash the parent and child values
        dtype: Arrow data type for the parent and child columns
        parallel: Whether to work on separate connected components in parallel
        timeout: Maximum seconds to wait for each worker to finish processing.
            Ignored if parallel==False
        min_rows_per_worker: Minimum number of rows to pass to each worker.
            Ignored if parallel==False

    Returns:
        Arrow table with columns ['parent', 'child', 'probability']
    """
    # Handle empty probabilities
    if len(probabilities) == 0:
        return pa.table(
            {
                "parent": pa.array([], type=dtype()),
                "child": pa.array([], type=dtype()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    probabilities = probabilities.sort_by([("component", "ascending")])
    components = pc.unique(probabilities["component"])
    n_cores = multiprocessing.cpu_count()
    n_components = len(components)

    logger.info(f"Processing {n_components:,} components using {n_cores} workers")

    # Split table into separate component tables
    component_col = probabilities["component"]
    indices = []
    start_idx = 0

    with build_progress_bar() as progress:
        split_task = progress.add_task(
            "[cyan]Splitting tables...", total=len(component_col)
        )

        for i in range(1, len(component_col)):
            if (component_col[i] != component_col[i - 1]) and (
                (i - start_idx) >= min_rows_per_worker
            ):
                indices.append((start_idx, i))
                start_idx = i
            progress.update(split_task, advance=1)

        indices.append((start_idx, len(component_col)))
        progress.update(split_task, completed=len(component_col))

    component_tables = []
    for start, end in indices:
        idx_array = pa.array(range(start, end))
        component_tables.append(probabilities.take(idx_array))

    # Process components in parallel
    results = []
    with build_progress_bar() as progress:
        process_task = progress.add_task(
            "[green]Processing components...", total=len(component_tables)
        )
        if parallel:
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = [
                    executor.submit(
                        proc_func, component_table, hash_func=hash_func, dtype=dtype
                    )
                    for component_table in component_tables
                ]
                for future in futures:
                    try:
                        result = future.result(timeout=timeout)
                        results.append(result)
                        progress.update(process_task, advance=1)
                    except TimeoutError:
                        logger.error(
                            f"Component processing timed out after {timeout} seconds"
                        )
                        raise
                    except Exception as e:
                        logger.error(f"Error processing component: {str(e)}")
                        raise
        else:
            results = [
                proc_func(component_table, hash_func=hash_func, dtype=dtype)
                for component_table in component_tables
                if not progress.update(process_task, advance=1)
            ]

    logger.info(f"Completed processing {len(results):,} components successfully")

    # Create empty table if no results
    if not results:
        logger.warning("No results to concatenate")
        return pa.table(
            {
                "parent": pa.array([], type=dtype()),
                "child": pa.array([], type=dtype()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    return pa.concat_tables(results)
