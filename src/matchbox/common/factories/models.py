"""Factory functions for generating dummy models and data for testing."""

import warnings
from collections import Counter
from functools import cache
from textwrap import dedent
from typing import Any, Hashable, Literal, TypeVar
from unittest.mock import Mock, PropertyMock, create_autospec

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rustworkx as rx
from faker import Faker
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import create_engine

from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.models.models import Model
from matchbox.client.results import Results
from matchbox.common.dtos import ModelMetadata, ModelType
from matchbox.common.factories.entities import (
    FeatureConfig,
    ResultsEntity,
    SourceEntity,
    SuffixRule,
    probabilities_to_results_entities,
)
from matchbox.common.factories.sources import (
    SourceConfig,
    SourceDummy,
    linked_sources_factory,
    source_factory,
)
from matchbox.common.transform import DisjointSet, graph_results

T = TypeVar("T", bound=Hashable)


def verify_components(all_nodes: list[Any], table: pa.Table) -> dict:
    """Fast verification of connected components using rustworkx.

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
    entities: set[ResultsEntity],
    source_entities: set[SourceEntity],
) -> bool:
    """Validate that probability edges create valid components.

    Each component should be a subset of exactly one source entity.

    Args:
        edges: List of (left_id, right_id) edges
        entities: Set of ResultsEntities
        source_entities: Set of SourceEntities
    """
    # Create disjoint set of ResultsEntities
    ds = DisjointSet[ResultsEntity]()

    # Map IDs to ResultsEntities for lookup
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

        # Merge all ResultsEntities in component
        merged = None
        for entity in component:
            if merged is None:
                merged = entity
            else:
                merged = merged + entity

        # Find which source entity this component belongs to
        found_source = None
        for source in source_entities:
            if merged.is_subset_of(source):
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

    Args:
        left_values: Tuple of integers to use for left column
        right_values: Tuple of integers to use for right column. If None, assume we
            are generating probabilities for deduplication
        prob_range: Tuple of (min_prob, max_prob) to constrain probabilities
        num_components: Number of distinct connected components to generate
        total_rows: Total number of rows to generate

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
            # TODO: this can certainly be optimised
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


@cache
def generate_entity_probabilities(
    left_entities: frozenset[ResultsEntity],
    right_entities: frozenset[ResultsEntity] | None,
    source_entities: frozenset[SourceEntity],
    prob_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> pa.Table:
    """Generate probabilities that will recover entity relationships.

    Compares ResultsEntities against ground truth SourceEntities by checking whether
    their EntityReferences are subsets of the source entities. Initially focused on
    generating fully connected, correct probabilities only.

    Args:
        left_entities: Set of ResultsEntities from left input
        right_entities: Set of ResultsEntities from right input. If None, assume
            we are deduplicating left_entities.
        source_entities: Ground truth set of SourceEntities
        prob_range: Range of probabilities to assign to matches. All matches will
            be assigned a random probability in this range.

    Returns:
        PyArrow Table with 'left_id', 'right_id', and 'probability' columns
    """
    # Handle deduplication case
    if right_entities is None:
        right_entities = left_entities

    # Create mapping of ResultsEntity -> SourceEntity
    entity_mapping: dict[ResultsEntity, SourceEntity] = {}

    def _map_entity(entity: ResultsEntity) -> None:
        matching_sources = [
            source for source in source_entities if entity.is_subset_of(source)
        ]
        if len(matching_sources) > 1:
            raise ValueError(
                f"ResultsEntity with ID {entity.id} is a subset of multiple "
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
        SourceEntity, tuple[set[ResultsEntity], set[ResultsEntity]]
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
                if right_entities is left_entities:
                    if left_entity.id >= right_entity.id:
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
            names=["left_id", "right_id", "probability"],
        )

    # Convert to arrays
    lefts, rights, probs = zip(*edges, strict=False)

    # Create PyArrow arrays
    left_array = pa.array(lefts, type=pa.uint64())
    right_array = pa.array(rights, type=pa.uint64())
    prob_array = pa.array(probs, type=pa.uint8())

    return pa.table(
        [left_array, right_array, prob_array],
        names=["left_id", "right_id", "probability"],
    )


class ModelDummy(BaseModel):
    """Complete representation of a generated dummy Model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    left_query: pa.Table
    left_results: dict[int, ResultsEntity]
    right_query: pa.Table | None
    right_results: dict[int, ResultsEntity] | None
    probabilities: pa.Table

    _entities: tuple[ResultsEntity, ...]
    _threshold: int
    _query_lookup: pa.Table

    @property
    def name(self) -> str:
        """Name of the model."""
        return self.model.metadata.name

    @property
    def entities(self) -> tuple[ResultsEntity, ...]:
        """Entities that were generated by the model."""
        return self._entities

    @entities.setter
    def entities(self, value: tuple[ResultsEntity, ...]):
        self._entities = value

    @property
    def threshold(self) -> int:
        """Threshold for the model."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: int):
        """Set the threshold for the model."""
        right_results = self.right_results.values() if self.right_results else []
        input_results = set(self.left_results.values()) | set(right_results)

        entities: tuple[ResultsEntity] = probabilities_to_results_entities(
            probabilities=self.probabilities,
            left_results=tuple(self.left_results.values()),
            right_results=tuple(right_results)
            if self.right_results is not None
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
    def init_query_lookup(self) -> "ModelDummy":
        """Initialize the query lookup table."""
        self.threshold = 0
        return self

    def to_mock(self) -> Mock:
        """Create a mock Model object that mimics this dummy model's behavior."""
        mock_model = create_autospec(Model)

        # Set basic attributes
        mock_model.metadata = self.model
        mock_model.left_data = DataFrame()  # Default empty DataFrame
        mock_model.right_data = (
            DataFrame() if self.model.type == ModelType.LINKER else None
        )

        # Mock results property
        mock_results = Results(probabilities=self.data, metadata=self.model)
        type(mock_model).results = PropertyMock(return_value=mock_results)

        # Mock run method
        mock_model.run.return_value = mock_results

        # Mock the model instance based on type
        if self.model.type == ModelType.LINKER:
            mock_model.model_instance = create_autospec(Linker)
            mock_model.model_instance.link.return_value = self.data
        else:
            mock_model.model_instance = create_autospec(Deduper)
            mock_model.model_instance.dedupe.return_value = self.data

        return mock_model

    def query(self) -> pa.Table:
        """Return the query table for the model."""
        if self.model.metadata.type == ModelType.DEDUPER:
            query = self.left_query
        else:
            query = pa.concat_tables(
                [self.left_query, self.right_query], promote_options="default"
            )

        indices = pc.index_in(query["id"], self._query_lookup["id"])
        indices = indices.combine_chunks()
        replacements = pc.take(self._query_lookup["new_id"], indices).combine_chunks()

        new_ids = pc.replace_with_mask(
            query["id"],
            pc.is_valid(indices),
            replacements,
        )

        id_index = query.schema.get_field_index("id")
        return query.set_column(id_index, "id", new_ids)


def model_factory(
    name: str | None = None,
    description: str | None = None,
    left_source: SourceDummy | ModelDummy | None = None,
    right_source: SourceDummy | ModelDummy | None = None,
    true_entities: tuple[SourceEntity, ...] | None = None,
    model_type: Literal["deduper", "linker"] | None = None,
    n_true_entities: int = 10,
    prob_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> ModelDummy:
    """Generate a complete dummy model.

    Allows autoconfiguration with minimal settings, or more nuanced control.

    Can either be used to generate a model in a pipeline, intercorrected with existing
    SourceDummy or ModelDummy objects, or generate a standalone model with random data.

    Args:
        name: Name of the model
        description: Description of the model
        left_source: A SourceDummy or ModelDummy for the left source
        right_source: If creating a linker, a SourceDummy or ModelDummy for the
            right source
        true_entities: Ground truth SourceEntity objects to use for
            generating probabilities. If none are supplied, random entities will
            be generated
        model_type: Type of the model, one of 'deduper' or 'linker'
            Ignored if left_source or right_source are provided.
        n_true_entities: Base number of entities to generate when using default configs.
            Ignored if left_source or right_source are provided.
        prob_range: Range of probabilities to generate
        seed: Random seed for reproducibility

    Returns:
        SourceModel: A dummy model with generated data
    """
    # Validate inputs
    if not (0 <= prob_range[0] <= prob_range[1] <= 1):
        raise ValueError("Probabilities must be increasing values between 0 and 1")
    if any([left_source, true_entities]) and any(
        [model_type is not None, n_true_entities != 10]
    ):
        warnings.warn(
            "Some arguments will be ignored as sources or true entities are provided",
            UserWarning,
            stacklevel=2,
        )

    generator = Faker()
    generator.seed_instance(seed)

    # Process provided sources or create defaults
    if left_source is not None:
        left_resolution = left_source.name
        left_query = left_source.query()
        left_entities = left_source.entities
        if right_source is not None:
            model_type = ModelType.LINKER
            right_resolution = right_source.name
            right_query = right_source.query()
            right_entities = right_source.entities
        else:
            model_type = ModelType.DEDUPER
            right_resolution, right_query, right_entities = None, None, None
    else:
        # Create default sources
        engine = create_engine("sqlite:///:memory:")
        model_type = ModelType(model_type.lower() if model_type else "deduper")
        features = {
            "company_name": FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            "crn": FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
            "duns": FeatureConfig(
                name="duns",
                base_generator="numerify",
                parameters=(("text", "########"),),
            ),
            "cdms": FeatureConfig(
                name="cdms",
                base_generator="numerify",
                parameters=(("text", "ORG-########"),),
            ),
            "address": FeatureConfig(
                name="address",
                base_generator="address",
            ),
        }

        left_resolution = generator.unique.word()
        left_config = SourceConfig(
            full_name="crn",
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
            n_true_entities=n_true_entities,
            repetition=0,
        )

        if model_type == ModelType.LINKER:
            right_resolution = generator.unique.word()
            right_config = SourceConfig(
                full_name="cdms",
                features=(
                    features["crn"],
                    features["cdms"],
                ),
                n_true_entities=n_true_entities,
                repetition=1,
            )

            linked = linked_sources_factory(
                source_configs=(left_config, right_config),
                n_true_entities=n_true_entities,
                seed=seed,
            )

            left_query = linked.sources["crn"].query()
            left_entities = linked.sources["crn"].entities
            right_query = linked.sources["cdms"].query()
            right_entities = linked.sources["cdms"].entities

            dummy_true_entities: tuple[SourceEntity, ...] = tuple(
                linked.true_entities.values()
            )
        else:
            right_resolution = None
            right_query = None
            right_entities = None

            source = source_factory(
                full_name=left_config.full_name,
                engine=left_config.engine,
                features=left_config.features,
                n_true_entities=left_config.n_true_entities,
                repetition=left_config.repetition,
                seed=seed,
            )

            left_query = source.query()
            left_entities = source.entities

            dummy_true_entities: tuple[SourceEntity, ...] = source.true_entities

    # Create model metadata, Model, and dummy probabilities

    metadata = ModelMetadata(
        name=name or generator.unique.word(),
        description=description or generator.sentence(),
        type=model_type,
        left_resolution=left_resolution,
        right_resolution=right_resolution,
    )

    model = Model(
        metadata=metadata,
        model_instance=Mock(),
        left_data=left_query,
        right_data=right_query,
    )

    # We need the to generate true entities when either:
    # * Sources provided, but no true entities given
    # * No sources provided, so need random n_true_entities
    if true_entities is None or left_source is None:
        true_entities = generator.random_elements(
            elements=dummy_true_entities,
            unique=True,
            length=n_true_entities,
        )

    probabilities = generate_entity_probabilities(
        left_entities=frozenset(left_entities),
        right_entities=frozenset(right_entities) if right_entities else None,
        source_entities=frozenset(true_entities),
        prob_range=prob_range,
        seed=seed,
    )

    return ModelDummy(
        model=model,
        left_query=left_query,
        left_results={entity.id: entity for entity in left_entities},
        right_query=right_query,
        right_results={entity.id: entity for entity in right_entities}
        if right_entities
        else None,
        probabilities=probabilities,
    )
