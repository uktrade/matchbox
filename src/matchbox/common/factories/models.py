"""Factory functions for generating model testkits and data for testing."""

import warnings
from collections import Counter
from collections.abc import Hashable
from functools import cache
from textwrap import dedent
from typing import Any, Literal, TypeVar

import numpy as np
import polars as pl
import pyarrow as pa
import rustworkx as rx
from faker import Faker
from pyarrow import compute as pc
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import create_engine

from matchbox.client.dags import DAG
from matchbox.client.models import add_model_class
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.models.models import Model
from matchbox.client.queries import Query
from matchbox.common.arrow import SCHEMA_RESULTS
from matchbox.common.dtos import (
    ModelType,
)
from matchbox.common.factories.entities import (
    ClusterEntity,
    FeatureConfig,
    SourceEntity,
    SuffixRule,
    probabilities_to_results_entities,
    query_to_cluster_entities,
)
from matchbox.common.factories.sources import (
    SourceTestkit,
    SourceTestkitParameters,
    linked_sources_factory,
)
from matchbox.common.graph import (
    ModelResolutionName,
    SourceResolutionName,
)
from matchbox.common.transform import DisjointSet, graph_results

T = TypeVar("T", bound=Hashable)


class MockDeduper(Deduper):
    """Mock deduper that does nothing."""

    def prepare(self, left: pl.DataFrame) -> None:
        """Mock prepare method."""
        return self

    def dedupe(self, left: pl.DataFrame) -> pl.DataFrame:
        """Mock dedupe method."""
        return pl.from_arrow(pa.Table.from_pylist([], schema=SCHEMA_RESULTS))


class MockLinker(Linker):
    """Mock linker that does nothing."""

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Mock prepare method."""
        return self

    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """Mock link method."""
        return pl.from_arrow(pa.Table.from_pylist([], schema=SCHEMA_RESULTS))


add_model_class(MockDeduper)
add_model_class(MockLinker)


def component_report(all_nodes: list[Any], table: pa.Table) -> dict:
    """Fast reporting on connected components using rustworkx.

    Args:
        all_nodes: list of identities of inputs being matched
        table: PyArrow table with 'left', 'right' columns

    Returns:
        dictionary containing basic component statistics
    """
    graph, _, _ = graph_results(table, all_nodes)
    components = rx.connected_components(graph)
    component_sizes = Counter(len(component) for component in components)

    return {
        "num_components": len(components),
        "total_nodes": graph.num_nodes(),
        "total_edges": graph.num_edges(),
        "component_sizes": component_sizes,
        "min_component_size": min(component_sizes.keys()),
        "max_component_size": max(component_sizes.keys()),
    }


def validate_components(
    edges: list[tuple[int, int]],
    entities: set[ClusterEntity],
    source_entities: set[SourceEntity],
) -> bool:
    """Validate that probability edges create valid components.

    Each component should be a subset of exactly one source entity.

    Args:
        edges: List of (left_id, right_id) edges
        entities: Set of ClusterEntity objects
        source_entities: Set of SourceEntities
    """
    # Create disjoint set of ClusterEntity objects
    ds = DisjointSet[ClusterEntity]()

    # Map IDs to ClusterEntity objects for lookup
    id_to_entity = {entity.id: entity for entity in entities}

    # Union entities based on probability edges
    for left_id, right_id in edges:
        left = id_to_entity[left_id]
        right = id_to_entity[right_id]
        ds.union(left, right)

    # Get components
    components = ds.get_components()

    # For each component, check if it's a subset of a single source entity
    for component in components:
        if len(component) <= 1:
            continue

        # Merge all ClusterEntity objects in component
        merged = None
        for entity in component:
            if merged is None:
                merged = entity
            else:
                merged += entity

        # Find which source entity this component belongs to
        found_source = None
        for source in source_entities:
            if merged.is_subset_of_source_entity(source):
                if found_source is not None:
                    # Component matches multiple source entities - invalid!
                    return False
                found_source = source

        if found_source is None:
            # Component doesn't match any source entity
            return False

    return True


def _min_edges_component(left: int, right: int, deduplicate: bool) -> int:
    """Calculate min edges for component to be connected.

    Does so by assuming a spanning tree.

    Args:
        left: number of nodes of component on the left
        right: number of nodes of component on the right (for linking)
        deduplicate: whether edges are for deduplication

    Returns:
        Minimum number of edges
    """
    if not deduplicate:
        return left + right - 1

    return left - 1


def _max_edges_component(left: int, right: int, deduplicate: bool) -> int:
    """Calculate max edges for component to be avoid duplication.

    Considers complete graph for deduping, and complete bipartite graph for linking.

    Args:
        left: number of nodes of component on the left
        right: number of nodes of component on the right (for linking)
        deduplicate: whether edges are for deduplication

    Returns:
        Maximum number of edges
    """
    if not deduplicate:
        return left * right
    # n*(n-1) is always divisible by 2
    return left * (left - 1) // 2


def calculate_min_max_edges(
    left_nodes: int, right_nodes: int, num_components: int, deduplicate: bool
) -> tuple[int, int]:
    """Calculate min and max edges for a graph.

    Args:
        left_nodes: number of nodes in left source
        right_nodes: number of nodes in right source
        num_components: number of requested components
        deduplicate: whether edges are for deduplication

    Returns:
        Two-tuple representing min and max edges
    """
    left_mod, right_mod = left_nodes % num_components, right_nodes % num_components
    left_div, right_div = left_nodes // num_components, right_nodes // num_components

    min_mod, max_mod = sorted([left_mod, right_mod])

    min_edges, max_edges = 0, 0
    # components where both sides have maximum nodes
    min_edges += (
        _min_edges_component(left_div + 1, right_div + 1, deduplicate) * min_mod
    )
    max_edges += (
        _max_edges_component(left_div + 1, right_div + 1, deduplicate) * min_mod
    )
    # components where one side has maximum nodes
    left_after_min_mod, right_after_min_mod = left_div + 1, right_div
    if left_mod == min_mod:
        left_after_min_mod, right_after_min_mod = left_div, right_div + 1
    min_edges += _min_edges_component(
        left_after_min_mod, right_after_min_mod, deduplicate
    ) * (max_mod - min_mod)
    max_edges += _max_edges_component(
        left_after_min_mod, right_after_min_mod, deduplicate
    ) * (max_mod - min_mod)
    # components where both side have minimum nodes
    min_edges += _min_edges_component(left_div, right_div, deduplicate) * (
        num_components - max_mod
    )
    max_edges += _max_edges_component(left_div, right_div, deduplicate) * (
        num_components - max_mod
    )

    return min_edges, max_edges


@cache
def generate_dummy_probabilities(
    left_values: tuple[int],
    right_values: tuple[int] | None,
    prob_range: tuple[float, float],
    num_components: int,
    total_rows: int | None = None,
    seed: int = 42,
) -> pa.Table:
    """Generate dummy Arrow probabilities data with guaranteed isolated components.

    While much of the factory system uses generate_entity_probabilities, this function
    is still in use in PostgreSQL benchmarking, and has been designed to be performant
    at scale.

    Args:
        left_values: Tuple of integers to use for left column
        right_values: Tuple of integers to use for right column. If None, assume we
            are generating probabilities for deduplication
        prob_range: Tuple of (min_prob, max_prob) to constrain probabilities
        num_components: Number of distinct connected components to generate
        total_rows: Total number of rows to generate
        seed: Random seed for reproducibility

    Returns:
        PyArrow Table with 'left_id', 'right_id', and 'probability' columns
    """
    # Validate inputs
    deduplicate = False
    if right_values is None:
        left_values = tuple(set(left_values))  # Remove duplicates
        right_values = left_values
        deduplicate = True

    if len(left_values) < 2 or len(right_values) < 2:
        raise ValueError("Need at least 2 possible values for both left and right")
    if num_components > min(len(left_values), len(right_values)):
        raise ValueError(
            "Cannot have more components than minimum of left/right values"
        )

    left_nodes, right_nodes = len(left_values), len(right_values)
    min_possible_edges, max_possible_edges = calculate_min_max_edges(
        left_nodes, right_nodes, num_components, deduplicate
    )

    mode = "dedupe" if deduplicate else "link"

    if total_rows is None:
        total_rows = min_possible_edges

    if total_rows == 0:
        raise ValueError("At least one edge must be generated")
    elif total_rows < min_possible_edges:
        raise ValueError(
            dedent(f"""
            Cannot generate {total_rows:,} {mode} edges with {num_components:,}
            components.
            Min edges is {min_possible_edges:,} for nodes given.
            Either decrease the number of nodes, increase the number of components, 
            or increase the total edges requested.
            """)
        )
    elif total_rows > max_possible_edges:
        raise ValueError(
            dedent(f"""
            Cannot generate {total_rows:,} {mode} edges with {num_components:,}
            components. 
            Max edges is {max_possible_edges:,} for nodes given.
            Either increase the number of nodes, decrease the number of components, 
            or decrease the total edges requested.
            """)
        )

    n_extra_edges = total_rows - min_possible_edges

    # Create seeded random number generator
    rng = np.random.default_rng(seed=seed)

    # Convert probability range to integers (60-80 for 0.60-0.80)
    prob_min = int(prob_range[0] * 100)
    prob_max = int(prob_range[1] * 100)

    # Split values into completely separate groups for each component
    left_components = np.array_split(np.array(left_values), num_components)
    right_components = np.array_split(np.array(right_values), num_components)
    # For each left-right component pair, the right equals the left rotated by one
    right_components = [np.roll(c, -1) for c in right_components]

    all_edges = []

    # Generate edges for each component
    for comp_idx in range(num_components):
        comp_left_values = left_components[comp_idx]
        comp_right_values = right_components[comp_idx]

        min_comp_nodes, max_comp_nodes = sorted(
            [len(comp_left_values), len(comp_right_values)]
        )

        # Ensure basic connectivity within the component by creating a spanning-tree
        base_edges = set()
        # For deduping (A B C) you just need (A - B) (B - C) (C - A)
        # which just needs matching pairwise the data and its rotated version.
        # For deduping, `min_comp_nodes` == `max_comp_nodes`
        if deduplicate:
            for i in range(min_comp_nodes - 1):
                small_n, large_n = sorted([comp_left_values[i], comp_right_values[i]])
                base_edges.add((small_n, large_n))
        else:
            # For linking (A B) and (C D E), we begin by adding (A - C) and (B - D)
            for i in range(min_comp_nodes):
                base_edges.add((comp_left_values[i], comp_right_values[i]))
            # we now add (C - B)
            for i in range(min_comp_nodes - 1):
                base_edges.add((comp_left_values[i + 1], comp_right_values[i]))
            # we now add (A - D)
            left_right_diff = max_comp_nodes - min_comp_nodes
            for i in range(left_right_diff):
                left_i, right_i = 0, min_comp_nodes + i
                if len(comp_right_values) < len(comp_left_values):
                    left_i, right_i = min_comp_nodes + i, 0

                base_edges.add((comp_left_values[left_i], comp_right_values[right_i]))

        component_edges = list(base_edges)

        if n_extra_edges > 0:
            # Generate remaining random edges strictly within this component
            # Note: this can certainly be optimised, but doesn't matter much for now
            if deduplicate:
                all_possible_edges = list(
                    {
                        tuple(sorted([x, y]))
                        for x in comp_left_values
                        for y in comp_right_values
                        if x != y and tuple(sorted([x, y])) not in base_edges
                    }
                )
            else:
                all_possible_edges = list(
                    {
                        (x, y)
                        for x in comp_left_values
                        for y in comp_right_values
                        if x != y and (x, y) not in base_edges
                    }
                )
            max_new_edges = len(all_possible_edges)
            if max_new_edges >= n_extra_edges:
                edges_required = n_extra_edges
                n_extra_edges = 0
            else:
                edges_required = max_new_edges
                n_extra_edges -= max_new_edges

            extra_edges_idx = rng.choice(
                len(all_possible_edges), size=edges_required, replace=False
            )
            extra_edges = [
                e for i, e in enumerate(all_possible_edges) if i in extra_edges_idx
            ]
            component_edges += extra_edges
        random_probs = rng.integers(prob_min, prob_max + 1, size=len(component_edges))

        component_edges = [
            (le, ri, pr)
            for (le, ri), pr in zip(component_edges, random_probs, strict=True)
        ]

        all_edges.extend(component_edges)

    # Convert to arrays
    lefts, rights, probs = zip(*all_edges, strict=True)

    # Create PyArrow arrays
    left_array = pa.array(lefts, type=pa.uint64())
    right_array = pa.array(rights, type=pa.uint64())
    prob_array = pa.array(probs, type=pa.uint8())

    return pa.table(
        [left_array, right_array, prob_array],
        names=["left_id", "right_id", "probability"],
    )


def generate_entity_probabilities(
    left_entities: frozenset[ClusterEntity],
    right_entities: frozenset[ClusterEntity] | None,
    source_entities: frozenset[SourceEntity],
    prob_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> pa.Table:
    """Generate probabilities that will recover entity relationships.

    Compares ClusterEntity objects against ground truth SourceEntities by checking
    whether their EntityReferences are subsets of the source entities. Initially
    focused on generating fully connected, correct probabilities only.

    Args:
        left_entities: Set of ClusterEntity objects from left input
        right_entities: Set of ClusterEntity objects from right input. If None, assume
            we are deduplicating left_entities.
        source_entities: Ground truth set of SourceEntities
        prob_range: Range of probabilities to assign to matches. All matches will
            be assigned a random probability in this range.
        seed: Random seed for reproducibility

    Returns:
        PyArrow Table with 'left_id', 'right_id', and 'probability' columns
    """
    # Validate inputs
    if not (0 <= prob_range[0] <= prob_range[1] <= 1):
        raise ValueError("Probabilities must be increasing values between 0 and 1")

    # Handle deduplication case
    if right_entities is None:
        right_entities = left_entities

    # Create mapping of ClusterEntity -> SourceEntity
    entity_mapping: dict[ClusterEntity, SourceEntity] = {}

    def _map_entity(entity: ClusterEntity) -> None:
        matching_sources = [
            source
            for source in source_entities
            if entity.is_subset_of_source_entity(source)
        ]
        if len(matching_sources) > 1:
            raise ValueError(
                f"ClusterEntity with ID {entity.id} is a subset of multiple "
                f"SourceEntities. This violates the uniqueness constraint."
            )
        if matching_sources:
            entity_mapping[entity] = matching_sources[0]

    # Map all entities, checking constraints
    for entity in left_entities:
        _map_entity(entity)
    if right_entities is not left_entities:
        for entity in right_entities:
            _map_entity(entity)

    # Group by SourceEntity
    source_groups: dict[
        SourceEntity, tuple[set[ClusterEntity], set[ClusterEntity]]
    ] = {}
    for entity, source in entity_mapping.items():
        if source not in source_groups:
            source_groups[source] = (set(), set())
        # Add to left or right group based on which input set it came from
        if entity in left_entities:
            source_groups[source][0].add(entity)
        if entity in right_entities:  # Note: could be in both for deduplication
            source_groups[source][1].add(entity)

    # Generate probability edges for each group
    edges = []
    rng = np.random.default_rng(seed=seed)

    # Convert probability range to integers (80-100 for 0.80-1.00)
    prob_min = int(prob_range[0] * 100)
    prob_max = int(prob_range[1] * 100)

    for left_group, right_group in source_groups.values():
        # Skip empty groups
        if not left_group or not right_group:
            continue

        # Generate all pairs within this group
        for left_entity in left_group:
            for right_entity in right_group:
                # For deduplication, only include each pair once
                # and ensure left_id < right_id
                if (
                    right_entities == left_entities
                    and left_entity.id >= right_entity.id
                ):
                    continue

                # Generate random probability in range
                prob = rng.integers(prob_min, prob_max + 1)
                edges.append((left_entity.id, right_entity.id, prob))

    # If no edges were generated, return empty table with correct schema
    if not edges:
        return pa.table(
            [
                pa.array([], type=pa.uint64()),
                pa.array([], type=pa.uint64()),
                pa.array([], type=pa.uint8()),
            ],
            schema=SCHEMA_RESULTS,
        )

    # Convert to arrays
    lefts, rights, probs = zip(*edges, strict=False)

    # Create PyArrow arrays
    left_array = pa.array(lefts, type=pa.uint64())
    right_array = pa.array(rights, type=pa.uint64())
    prob_array = pa.array(probs, type=pa.uint8())

    return pa.table(
        [left_array, right_array, prob_array],
        schema=SCHEMA_RESULTS,
    )


class ModelTestkit(BaseModel):
    """A testkit of data and metadata for a Model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    left_data: pa.Table
    left_query: Query
    left_clusters: dict[int, ClusterEntity]
    right_data: pa.Table | None
    right_query: Query | None
    right_clusters: dict[int, ClusterEntity] | None
    probabilities: pa.Table

    _entities: tuple[ClusterEntity, ...]
    _threshold: int
    _query_lookup: pa.Table

    @property
    def name(self) -> str:
        """Return the full name of the Model."""
        return self.model.name

    @property
    def data(self) -> pa.Table:
        """Return a PyArrow table in the same format as matchbox queries."""
        if self.model.config.type == ModelType.DEDUPER:
            data = self.left_data
        else:
            data = pa.concat_tables(
                [self.left_data, self.right_data], promote_options="default"
            )

        indices = pc.index_in(data["id"], self._query_lookup["id"])
        indices = indices.combine_chunks()
        replacements = pc.take(self._query_lookup["new_id"], indices).combine_chunks()

        new_ids = pc.replace_with_mask(
            data["id"],
            pc.is_valid(indices),
            replacements,
        )

        id_index = data.schema.get_field_index("id")
        return data.set_column(id_index, "id", new_ids)

    @property
    def entities(self) -> tuple[ClusterEntity, ...]:
        """ClusterEntities that were generated by the model."""
        return self._entities

    @entities.setter
    def entities(self, value: tuple[ClusterEntity, ...]):
        self._entities = value

    @property
    def threshold(self) -> int:
        """Threshold for the model."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: int):
        """Set the threshold for the model."""
        right_clusters = self.right_clusters.values() if self.right_clusters else []
        input_results = set(self.left_clusters.values()) | set(right_clusters)

        entities: tuple[ClusterEntity] = probabilities_to_results_entities(
            probabilities=self.probabilities,
            left_clusters=tuple(self.left_clusters.values()),
            right_clusters=tuple(right_clusters)
            if self.right_clusters is not None
            else None,
            threshold=value,
        )

        id_mapping = {
            original.id: entity.id
            for original in input_results
            for entity in set(entities)
            if original in entity
        }

        ids, new_ids = zip(*id_mapping.items(), strict=True) if id_mapping else ([], [])

        self._query_lookup = pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "new_id": pa.array(new_ids, type=pa.int64()),
            }
        )
        self._entities = entities
        self._threshold = value

    @model_validator(mode="after")
    def init_query_lookup(self) -> "ModelTestkit":
        """Initialize the query lookup table."""
        self.threshold = 0
        return self


def _testkit_to_query(testkit: SourceTestkit | ModelTestkit) -> Query:
    if isinstance(testkit, SourceTestkit):
        return Query(testkit.source, dag=testkit.source.dag)
    else:
        all_sources = list(testkit.model.left_query.sources)
        if testkit.model.right_query is not None:
            all_sources += list(testkit.model.right_query.sources)
        return Query(
            *all_sources,
            model=testkit.model,
            dag=testkit.model.dag,
        )


def model_factory(
    name: ModelResolutionName | None = None,
    dag: DAG | None = None,
    description: str | None = None,
    left_testkit: SourceTestkit | ModelTestkit | None = None,
    right_testkit: SourceTestkit | ModelTestkit | None = None,
    true_entities: tuple[SourceEntity, ...] | None = None,
    model_type: Literal["deduper", "linker"] | None = None,
    n_true_entities: int | None = None,
    prob_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> ModelTestkit:
    """Generate a complete model testkit.

    Allows autoconfiguration with minimal settings, or more nuanced control.

    Can either be used to generate a model in a pipeline, interconnected with existing
    SourceTestkit or ModelTestkit objects, or generate a standalone model with
    random data.

    Args:
        name: Name of the model
        dag: DAG containing this model.
            Overridden by dag of left testkit if present.
        description: Description of the model
        left_testkit: A SourceTestkit or ModelTestkit for the left source
        right_testkit: If creating a linker, a SourceTestkit or ModelTestkit for the
            right source
        true_entities: Ground truth SourceEntity objects to use for
            generating probabilities. Must be supplied if sources are given
        model_type: Type of the model, one of 'deduper' or 'linker'
            Defaults to deduper. Ignored if left_testkit or right_testkit are provided.
        n_true_entities: Base number of entities to generate when using default configs.
            Defaults to 10. Ignored if left_testkit or right_testkit are provided.
        prob_range: Range of probabilities to generate
        seed: Random seed for reproducibility

    Returns:
        ModelTestkit: A model testkit with generated data

    Raises:
        ValueError:
            * If probabilities are not in increasing order and between 0 and 1
            * If sources are provided without true entities
        UserWarning: If some arguments are ignored due to sources or true entities
    """
    # ==== Input validation ====
    if not (0 <= prob_range[0] <= prob_range[1] <= 1):
        raise ValueError("Probabilities must be increasing values between 0 and 1")

    if left_testkit is not None and true_entities is None:
        raise ValueError("Must provide true entities when sources are given")

    if any([left_testkit, true_entities]) and any(
        [model_type is not None, n_true_entities is not None]
    ):
        warnings.warn(
            "Some arguments will be ignored as sources or true entities are provided",
            UserWarning,
            stacklevel=2,
        )

    # ==== Setup ====
    generator = Faker()
    generator.seed_instance(seed)
    n_true_entities = n_true_entities or 10
    dummy_true_entities = None

    # ==== SourceConfig configuration ====
    if left_testkit is not None:
        # Using provided sources
        left_data = left_testkit.data
        left_query = _testkit_to_query(left_testkit)
        left_entities = left_testkit.entities
        if isinstance(left_testkit, SourceTestkit):
            dag = left_testkit.source.dag
        else:
            dag = left_testkit.model.dag

        right_data = None
        right_query = None
        right_entities = None

        if right_testkit is not None:
            model_type = ModelType.LINKER
            right_data = right_testkit.data
            right_query = _testkit_to_query(right_testkit)
            right_entities = right_testkit.entities
        else:
            model_type = ModelType.DEDUPER
            right_entities = None
    else:
        if dag is None:
            dag = DAG("collection")
        # Create default sources
        engine = create_engine("sqlite:///:memory:")
        resolved_model_type = ModelType(model_type.lower() if model_type else "deduper")

        # Define common features
        features = {
            "company_name": FeatureConfig(
                name="company_name", base_generator="company"
            ),
            "crn": FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
            "cdms": FeatureConfig(
                name="cdms",
                base_generator="numerify",
                parameters=(("text", "ORG-########"),),
            ),
        }

        # Configure left source
        left_parameters = SourceTestkitParameters(
            name="crn",
            engine=engine,
            features=(
                features["company_name"].add_variations(
                    SuffixRule(suffix=" Limited"),
                    SuffixRule(suffix=" UK"),
                    SuffixRule(suffix=" Company"),
                ),
                features["crn"],
            ),
            drop_base=True,
            repetition=0,
        )

        # Configure sources based on model type
        source_parameters = [left_parameters]
        if resolved_model_type == ModelType.LINKER:
            source_parameters.append(
                SourceTestkitParameters(
                    name="cdms",
                    features=(features["crn"], features["cdms"]),
                    repetition=1,
                )
            )

        # Generate linked sources
        linked = linked_sources_factory(
            dag=dag,
            source_parameters=tuple(source_parameters),
            n_true_entities=n_true_entities,
            seed=seed,
        )

        # Extract source data
        left_data = linked.sources["crn"].data
        left_query = Query(linked.sources["crn"], dag=dag)
        left_entities = linked.sources["crn"].entities

        right_data = None
        right_query = None
        right_entities = None
        if resolved_model_type == ModelType.LINKER:
            right_data = linked.sources["cdms"].data
            right_query = Query(linked.sources["cdms"], dag=dag)
            right_entities = linked.sources["cdms"].entities

        dummy_true_entities = tuple(linked.true_entities)
        model_type = resolved_model_type

    # ==== Model creation ====
    model_class = MockLinker if model_type == ModelType.LINKER else MockDeduper
    model_settings = (
        LinkerSettings() if model_type == ModelType.LINKER else DeduperSettings()
    )
    model = Model(
        dag=dag,
        name=name or generator.unique.word(),
        description=description or generator.sentence(),
        model_class=model_class,
        model_settings=model_settings,
        left_query=left_query,
        right_query=right_query,
    )

    # ==== Entity and probability generation ====
    # We need to generate true entities when either:
    # * No true entities are provided (true_entities is None)
    # * We're using default sources (left_testkit is None)
    if true_entities is None or left_testkit is None:
        final_true_entities = generator.random_elements(
            elements=dummy_true_entities,
            unique=True,
            length=n_true_entities,
        )
    else:
        final_true_entities = true_entities

    probabilities = generate_entity_probabilities(
        left_entities=frozenset(left_entities),
        right_entities=frozenset(right_entities) if right_entities else None,
        source_entities=frozenset(final_true_entities),
        prob_range=prob_range,
        seed=seed,
    )

    # ==== Final model creation ====
    return ModelTestkit(
        model=model,
        left_data=left_data,
        left_query=left_query,
        left_clusters={entity.id: entity for entity in left_entities},
        right_data=right_data,
        right_query=right_query,
        right_clusters={entity.id: entity for entity in right_entities}
        if right_entities
        else None,
        probabilities=probabilities,
    )


def query_to_model_factory(
    left_query: Query,
    left_data: pa.Table,
    left_keys: dict[SourceResolutionName, str],
    true_entities: tuple[SourceEntity, ...],
    name: ModelResolutionName | None = None,
    description: str | None = None,
    right_query: Query | None = None,
    right_data: pa.Table | None = None,
    right_keys: dict[SourceResolutionName, str] | None = None,
    prob_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> ModelTestkit:
    """Turns raw queries from Matchbox into ModelTestkits.

    Args:
        left_query: Query generating left data
        left_data: PyArrow table with left query data
        left_keys: Dictionary mapping source resolution names to key field names
            in left query
        true_entities: Ground truth SourceEntity objects to use for generating
            probabilities
        name: Name of the model
        description: Description of the model
        right_query: Query generating right data
        right_data: PyArrow table with right query data, if creating a linker
        right_keys: Dictionary mapping source resolution names to key field names
            in right query
        prob_range: Range of probabilities to generate
        seed: Random seed for reproducibility

    Returns:
        ModelTestkit with the processed data
    """
    # Validate inputs
    if not (0 <= prob_range[0] <= prob_range[1] <= 1):
        raise ValueError("Probabilities must be increasing values between 0 and 1")

    dag = left_query.dag
    if right_query and right_query.dag != dag:
        raise ValueError("DAGs for left and right query must match.")

    # Check if right-side arguments are consistently provided
    right_args = [right_query, right_data, right_keys]
    if any(arg is not None for arg in right_args) and not all(
        arg is not None for arg in right_args
    ):
        raise ValueError(
            "When providing right-side arguments, all of right_query, "
            "right_data, and right_keys must be provided"
        )

    # Create a Faker instance with the seed
    generator = Faker()
    generator.seed_instance(seed)

    # Create left and right ClusterEntity sets
    left_clusters = query_to_cluster_entities(left_data, left_keys)

    if right_data is not None and right_keys is not None:
        right_clusters = query_to_cluster_entities(right_data, right_keys)
    else:
        right_clusters = None

    # Create model metadata
    model_class = MockLinker if right_data is not None else MockDeduper
    model_settings = LinkerSettings() if right_data is not None else DeduperSettings()

    model = Model(
        dag=dag,
        name=name or generator.unique.word(),
        description=description or generator.sentence(),
        model_class=model_class,
        model_settings=model_settings,
        left_query=left_query,
        right_query=right_query,
    )

    # Generate probabilities
    probabilities = generate_entity_probabilities(
        left_entities=frozenset(left_clusters),
        right_entities=frozenset(right_clusters) if right_clusters else None,
        source_entities=frozenset(true_entities) if true_entities else frozenset(),
        prob_range=prob_range,
        seed=seed,
    )

    # Create ModelTestkit
    return ModelTestkit(
        model=model,
        left_data=left_data,
        left_query=left_query,
        left_clusters={entity.id: entity for entity in left_clusters},
        right_data=right_data,
        right_query=right_query,
        right_clusters={entity.id: entity for entity in right_clusters}
        if right_clusters
        else None,
        probabilities=probabilities,
    )
