import logging
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Generic, Hashable, Literal, TypeVar
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rustworkx as rx
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from matchbox.common.hash import hash_values

T = TypeVar("T", bound=Hashable)

logic_logger = logging.getLogger("mb_logic")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


def to_clusters(results: pa.Table) -> pa.Table:
    """
    Converts probabilities into a list of connected components formed at each threshold.

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
            children = sorted([G.get_node_data(n) for n in comp])
            parent = hash_values(*children)

            components["parent"].extend([parent] * len(children))
            components["child"].extend(children)
            components["threshold"].extend([prob] * len(children))

    return pa.Table.from_pydict(components)


def attach_components_to_probabilities(probabilities: pa.Table) -> pa.Table:
    """
    Takes an Arrow table of probabilities and adds a component column.

    Expects an Arrow table of column, left, right, probability.

    Returns a table with an additional column, component.
    """
    # Create index to use in graph
    unique = pc.unique(
        pa.concat_arrays(
            [
                probabilities["left"].combine_chunks(),
                probabilities["right"].combine_chunks(),
            ]
        )
    )
    left_indices = pc.index_in(probabilities["left"], unique)
    right_indices = pc.index_in(probabilities["right"], unique)

    # Create and process graph
    n_nodes = len(unique)
    n_edges = len(probabilities)

    graph = rx.PyGraph(node_count_hint=n_nodes, edge_count_hint=n_edges)
    graph.add_nodes_from(range(n_nodes))

    edges = tuple(zip(left_indices.to_numpy(), right_indices.to_numpy(), strict=True))
    graph.add_edges_from_no_data(edges)

    components = rx.connected_components(graph)

    # Convert components to arrays, map back to input to join, and reattach
    component_indices = np.concatenate([np.array(list(c)) for c in components])
    component_labels = np.repeat(
        np.arange(len(components)), [len(c) for c in components]
    )

    node_to_component = np.zeros(len(unique), dtype=np.int64)
    node_to_component[component_indices] = component_labels

    edge_components = pa.array(node_to_component[left_indices.to_numpy()])

    return probabilities.append_column("component", edge_components).sort_by(
        [("component", "ascending"), ("probability", "descending")]
    )


class DisjointSet(Generic[T]):
    """
    Disjoint set forest with "path compression" and "union by rank" heuristics.

    This follows implementation from Cormen, Thomas H., et al. Introduction to
    algorithms. MIT press, 2022
    """

    def __init__(self):
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}

    def _make_set(self, x: T) -> None:
        self.parent[x] = x
        self.rank[x] = 0

    def union(self, x: T, y: T) -> None:
        self._link(self._find(x), self._find(y))

    def _link(self, x: T, y: T) -> None:
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    def _find(self, x: T) -> T:
        if x not in self.parent:
            self._make_set(x)
            return x

        if x != self.parent[x]:
            self.parent[x] = self._find(self.parent[x])

        return self.parent[x]

    def get_components(self) -> list[set[T]]:
        components = defaultdict(set)
        for x in self.parent:
            root = self._find(x)
            components[root].add(x)
        return list(components.values())


def component_to_hierarchy(
    table: pa.Table,
    dtype: pa.DataType = pa.large_binary,
    hash_func: Callable[[*tuple[T, ...]], T] = hash_values,
) -> pa.Table:
    """
    Convert pairwise probabilities into a hierarchical representation.

    Assumes data is pre-sorted by probability descending.

    Args:
        table: Arrow Table with columns ['left', 'right', 'probability']

    Returns:
        Arrow Table with columns ['parent', 'child', 'probability']
    """
    probs = np.sort(pc.unique(table["probability"]).to_numpy())[::-1]

    djs = DisjointSet[int]()  # implements connected components
    current_roots: dict[int, set[int]] = defaultdict(set)  # tracks ultimate parents
    hierarchy: list[tuple[int, int, float]] = []  # the output of this function

    for threshold in probs:
        # Get current probability rows
        mask = pc.equal(table["probability"], threshold)
        current_probs = table.filter(mask)

        # Add new pairwise relationships at this threshold
        for left, right in zip(
            current_probs["left"].to_numpy(),
            current_probs["right"].to_numpy(),
            strict=True,
        ):
            djs.union(left, right)
            parent = hash_func(left, right)
            hierarchy.extend([(parent, left, threshold), (parent, right, threshold)])
            current_roots[left].add(parent)
            current_roots[right].add(parent)

        for children in djs.get_components():
            if len(children) <= 2:
                continue  # Skip pairs already handled by pairwise probabilities

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
    dtype: pa.DataType = pa.large_binary,
    timeout: int = 300,
) -> pa.Table:
    """
    Converts a table of pairwise probabilities into a table of hierarchical clusters.

    Args:
        probabilities: Arrow table with columns ['component', 'left', 'right',
            'probability']
        proc_func: Function to process each component
        dtype: Arrow data type for parent/child columns
        timeout: Maximum seconds to wait for each component to process

    Returns:
        Arrow table with columns ['parent', 'child', 'probability']
    """
    console = Console()
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    probabilities = probabilities.sort_by([("component", "ascending")])
    components = pc.unique(probabilities["component"])
    n_cores = multiprocessing.cpu_count()
    n_components = len(components)

    logic_logger.info(f"Processing {n_components:,} components using {n_cores} workers")

    # Split table into separate component tables
    component_col = probabilities["component"]
    indices = []
    start_idx = 0

    with Progress(*progress_columns, console=console) as progress:
        split_task = progress.add_task(
            "[cyan]Splitting tables...", total=len(component_col)
        )

        for i in range(1, len(component_col)):
            if component_col[i] != component_col[i - 1]:
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
    with Progress(*progress_columns, console=console) as progress:
        process_task = progress.add_task(
            "[green]Processing components...", total=len(component_tables)
        )

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(proc_func, component_table, dtype, hash_func)
                for component_table in component_tables
            ]

            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                    progress.update(process_task, advance=1)
                except TimeoutError:
                    logic_logger.error(
                        f"Component processing timed out after {timeout} seconds"
                    )
                    raise
                except Exception as e:
                    logic_logger.error(f"Error processing component: {str(e)}")
                    raise

    logic_logger.info(f"Completed processing {len(results):,} components successfully")

    # Create empty table if no results
    if not results:
        logic_logger.warning("No results to concatenate")
        return pa.table(
            {
                "parent": pa.array([], type=dtype()),
                "child": pa.array([], type=dtype()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    return pa.concat_tables(results)


def drop_duplicates(
    table: pa.Table,
    on: list[str] | None = None,
    keep: Literal["first", "last"] = "first",
) -> pa.Table:
    """
    Remove duplicate rows from a PyArrow table based on specified columns.

    This function efficiently removes duplicate rows from a PyArrow table,
    keeping either the first or last occurrence of each unique combination
    of values in the specified columns.

    Lifted with love from this gist:
    https://gist.github.com/nmehran/57f264bd951b2f77af08f760eafea40e

    An alternative:
    https://github.com/TomScheffers/pyarrow_ops/
    """
    if not isinstance(table, pa.Table):
        raise TypeError("Parameter 'table' must be a PyArrow Table")

    if keep not in ["first", "last"]:
        raise ValueError("Parameter 'keep' must be either 'first' or 'last'")

    if not on:
        on = table.column_names

    # Generate a unique column name for row index
    index_column = f"index_{uuid4().hex}"
    index_aggregate_column = f"{index_column}_{keep}"

    # Create row numbers
    num_rows = table.num_rows
    row_numbers = pa.array(np.arange(num_rows, dtype=np.int64))

    # Append row numbers, group by specified columns, and aggregate
    unique_indices = (
        table.append_column(index_column, row_numbers)
        .group_by(on, use_threads=False)
        .aggregate([(index_column, keep)])
    ).column(index_aggregate_column)

    return pc.take(table, unique_indices, boundscheck=False)
