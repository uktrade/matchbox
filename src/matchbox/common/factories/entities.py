"""Classes and functions for generating and comparing entities.

These underpin the entity resolution process, which is the core of the
source and model testkit factory system.
"""

from abc import ABC, abstractmethod
from collections import Counter
from functools import cache
from random import getrandbits
from typing import TYPE_CHECKING, Any, Self

import pandas as pd
import polars as pl
import pyarrow as pa
from faker import Faker
from frozendict import frozendict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from matchbox.common.dtos import DataTypes
from matchbox.common.graph import SourceResolutionName
from matchbox.common.transform import DisjointSet

if TYPE_CHECKING:
    from matchbox.common.factories.sources import SourceTestkit
else:
    SourceTestkit = Any


class VariationRule(BaseModel, ABC):
    """Abstract base class for variation rules."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def apply(self, value: str) -> str:
        """Apply the variation to a value."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the type of variation."""
        pass


class SuffixRule(VariationRule):
    """Add a suffix to a value."""

    suffix: str

    def apply(self, value: str) -> str:
        """Apply the suffix to the value."""
        return f"{value}{self.suffix}"

    @property
    def type(self) -> str:
        """Return the type of variation."""
        return "suffix"


class PrefixRule(VariationRule):
    """Add a prefix to a value."""

    prefix: str

    def apply(self, value: str) -> str:
        """Apply the prefix to the value."""
        return f"{self.prefix}{value}"

    @property
    def type(self) -> str:
        """Return the type of variation."""
        return "prefix"


class ReplaceRule(VariationRule):
    """Replace occurrences of a string with another."""

    old: str
    new: str

    def apply(self, value: str) -> str:
        """Apply the replacement to the value."""
        return value.replace(self.old, self.new)

    @property
    def type(self) -> str:
        """Return the type of variation."""
        return "replace"


def infer_data_type(base: str, parameters: tuple | None) -> DataTypes:
    """Infer an appropriate Matchbox type from a Faker configuration.

    Args:
        base: Faker generator type
        parameters: Parameters for the generator

    Returns:
        A Matchbox DataType
    """
    generator = Faker()
    value_generator = getattr(generator, base)
    parameters = {} if not parameters else dict(parameters)
    examples = [value_generator(**dict(parameters)) for _ in range(5)]
    series = pl.Series(examples)
    return DataTypes.from_dtype(series.dtype)


class FeatureConfig(BaseModel):
    """Configuration for generating a feature with variations."""

    model_config = ConfigDict(frozen=True)

    name: str
    base_generator: str
    parameters: tuple | None = Field(
        default=None,
        description=(
            "Parameters for the generator. A tuple of tuples passed to the generator."
        ),
    )
    unique: bool = Field(
        default=True,
        description=(
            "Whether the generator enforces uniqueness in the generated data. "
            "For example, using unique=True with the 'boolean' generator will error "
            "if more the two values are generated."
        ),
    )
    drop_base: bool = Field(
        default=False, description="Whether the base case is dropped."
    )
    variations: tuple[VariationRule, ...] = Field(default_factory=tuple)
    datatype: DataTypes = Field(
        default_factory=lambda data: infer_data_type(
            data["base_generator"], data["parameters"]
        )
    )

    def add_variations(self, *rule: VariationRule) -> "FeatureConfig":
        """Add a variation rule to the feature."""
        return FeatureConfig(
            name=self.name,
            base_generator=self.base_generator,
            parameters=self.parameters,
            unique=self.unique,
            drop_base=self.drop_base,
            variations=self.variations + tuple(rule),
        )

    @field_validator("name", mode="after")
    @classmethod
    def protected_names(cls: type[Self], value: str) -> str:
        """Ensure name is not a reserved keyword."""
        if value in {"id", "key"}:
            raise ValueError("Feature name cannot be 'id' or 'key'.")
        return value

    @field_validator("datatype", mode="before")
    @classmethod
    def string_to_strenum(cls: type[Self], value: str) -> DataTypes:
        """Convert string to DataTypes enum."""
        if isinstance(value, str):
            return DataTypes(value)
        return value


class EntityReference(frozendict):
    """Reference to an entity's presence in specific sources.

    Maps source resolution names to sets of primary keys.
    """

    def __init__(
        self, mapping: dict[SourceResolutionName, frozenset[str]] | None = None
    ) -> None:
        """Initialise the EntityReference."""
        super().__init__({} if mapping is None else mapping)

    def __add__(self, other: "EntityReference") -> "EntityReference":
        """Merge two EntityReferences by unioning keys for each source."""
        if not isinstance(other, EntityReference):
            return NotImplemented

        return EntityReference(
            {
                k: self.get(k, frozenset()) | other.get(k, frozenset())
                for k in self.keys() | other.keys()
            }
        )

    def __le__(self, other: "EntityReference") -> bool:
        """Test if self is a subset of other."""
        if not isinstance(other, EntityReference):
            return NotImplemented

        return all(name in other and self[name] <= other[name] for name in self)


class EntityIDMixin:
    """Mixin providing common ID-based functionality for entity classes.

    Implements integer conversion and comparison operators for sorting
    based on the entity's ID.
    """

    id: int

    def __int__(self) -> int:
        """Allow converting an entity to an integer by returning its ID."""
        return self.id

    def __lt__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id < other.id
        if isinstance(other, int):
            return self.id < other
        return NotImplemented

    def __gt__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id > other.id
        if isinstance(other, int):
            return self.id > other
        return NotImplemented

    def __le__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id <= other.id
        if isinstance(other, int):
            return self.id <= other
        return NotImplemented

    def __ge__(self, other: int | Any) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id >= other.id
        if isinstance(other, int):
            return self.id >= other
        return NotImplemented


class SourceKeyMixin:
    """Mixin providing common source key functionality for entity classes.

    Implements methods for accessing and retrieving source keys.
    """

    keys: EntityReference

    def get_keys(self, name: SourceResolutionName) -> set[str]:
        """Get keys for a specific source.

        Args:
            name: Name of the source

        Returns:
            Set of keys, empty if source not found
        """
        return set(self.keys.get(name, frozenset()))

    def get_values(
        self, sources: dict[SourceResolutionName, "SourceTestkit"]
    ) -> dict[SourceResolutionName, dict[str, list[str]]]:
        """Get all unique values for this entity across sources.

        Each source may have its own variations/transformations of the base data,
        so we maintain separation between sources.

        Args:
            sources: Dictionary of source resolution name to source data

        Returns:
            Dictionary mapping:
                source_name -> {
                    feature_name -> [unique values for that feature in that source]
                }
        """
        values: dict[str, dict[str, list[str]]] = {}

        # For each source we have keys for
        for source_name, keys in self.keys.items():
            source = sources.get(source_name)

            if source is None:
                raise ValueError(f"SourceConfig not found: {source_name}")

            # Get rows for this entity in this source
            df = source.data.to_pandas()
            entity_rows = df[df["key"].isin(keys)]

            # Get unique values for each feature in this source
            values[source_name] = {
                feature.name: sorted(entity_rows[feature.name].unique())
                for feature in source.features
            }

        return values


class ClusterEntity(BaseModel, EntityIDMixin, SourceKeyMixin):
    """Represents a merged entity mid-pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: int = Field(default_factory=lambda: getrandbits(63))  # 64 gives OverflowError
    keys: EntityReference

    def __add__(self, other: "ClusterEntity") -> "ClusterEntity":
        """Combine two ClusterEntity objects by combining the keys."""
        if other is None:
            return self
        if not isinstance(other, ClusterEntity):
            return NotImplemented
        return ClusterEntity(keys=self.keys + other.keys)

    def __radd__(self, other: Any) -> "ClusterEntity":
        """Handle sum() by treating 0 as an empty ClusterEntity."""
        if other == 0:  # sum() starts with 0
            return self
        return NotImplemented

    def __sub__(self, other: "ClusterEntity") -> dict[str, frozenset[str]]:
        """Return keys in self that aren't in other, by source.

        Used to diff two ClusterEntity objects.
        """
        if not isinstance(other, ClusterEntity):
            return NotImplemented

        diff = {}
        for name, our_keys in self.keys.items():
            their_keys = other.keys.get(name, frozenset())
            if remaining := our_keys - their_keys:
                diff[name] = remaining

        return diff

    def __rsub__(self, other: "ClusterEntity") -> dict[str, frozenset[str]]:
        """Support reverse subtraction."""
        if not isinstance(other, ClusterEntity):
            return NotImplemented
        return other - self

    def __eq__(self, other: Any) -> bool:
        """Compare based on keys."""
        if not isinstance(other, ClusterEntity):
            return NotImplemented
        return self.keys == other.keys

    def __contains__(self, other: "ClusterEntity") -> bool:
        """Check if this entity contains all keys from other entity."""
        return other.keys <= self.keys

    def __hash__(self) -> int:
        """Hash based on EntityReference which is itself hashable."""
        return hash(self.keys)

    def is_subset_of_source_entity(self, source_entity: "SourceEntity") -> bool:
        """Check if this ClusterEntity's references are a subset of a SourceEntity's."""
        return self.keys <= source_entity.keys

    def similarity_ratio(self, other: "ClusterEntity") -> float:
        """Return ratio of shared keys to total keys across all sources."""
        total_keys = 0
        shared_keys = 0

        # Get all source names
        all_sources = set(self.keys.keys()) | set(other.keys.keys())

        for name in all_sources:
            our_keys = self.keys.get(name, frozenset())
            their_keys = other.keys.get(name, frozenset())

            total_keys += len(our_keys | their_keys)
            shared_keys += len(our_keys & their_keys)

        return shared_keys / total_keys if total_keys > 0 else 0.0


class SourceEntity(BaseModel, EntityIDMixin, SourceKeyMixin):
    """Represents a single entity across all sources."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(default_factory=lambda: getrandbits(63))  # 64 gives OverflowError
    base_values: dict[str, Any] = Field(description="Feature name -> base value")
    keys: EntityReference = Field(
        description="Source to keys mapping",
        default=EntityReference(mapping=frozenset()),
    )
    total_unique_variations: int = Field(default=0)

    def __eq__(self, other: object) -> bool:
        """Equal if base values are shared, or integer ID matches."""
        if isinstance(other, SourceEntity):
            return self.base_values == other.base_values
        if isinstance(other, int):
            return self.id == other
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on sorted base values."""
        return hash(tuple(sorted(self.base_values.items())))

    def add_source_reference(self, name: SourceResolutionName, keys: list[str]) -> None:
        """Add or update a source reference.

        Args:
            name: Source name
            keys: List of primary keys for this source
        """
        mapping = dict(self.keys)
        mapping[name] = frozenset(keys)
        self.keys = EntityReference(mapping)

    def to_cluster_entity(self, *names: SourceResolutionName) -> ClusterEntity | None:
        """Convert this SourceEntity to a ClusterEntity with the specified sources.

        This method makes diffing really easy. Testing whether ClusterEntity objects
        are subsets of SourceEntity objects is a weaker, logically more fragile test
        than directly comparing equality of sets of ClusterEntity objects. It enables
        a really simple syntactical expression of the test.

        ```python
        actual: set[ClusterEntity] = ...
        expected: set[ClusterEntity] = {
            s.to_cluster_entity("source1", "source2")
            for s in source_entities
        }

        is_identical = expected) == actual
        missing = expected - actual
        extra = actual - expected
        ```

        Args:
            *names: Names of sources to include in the ClusterEntity

        Returns:
            ClusterEntity containing only the specified sources' keys, or None
            if none of the specified sources are present in this entity.
        """
        filtered = {
            name: self.keys.get(name)
            for name in names
            if self.keys.get(name) is not None
        }

        if len(filtered) == 0:
            return None

        return ClusterEntity(keys=EntityReference(filtered))


def query_to_cluster_entities(
    data: pa.Table | pd.DataFrame | pl.DataFrame, keys: dict[SourceResolutionName, str]
) -> set[ClusterEntity]:
    """Convert a query result to a set of ClusterEntities.

    Useful for turning a real query from a real model resolution in Matchbox into
    a set of ClusterEntities that can be used in `LinkedSourcesTestkit.diff_results()`.

    Args:
        data: A PyArrow table or DataFrame representing a query result
        keys: Mapping of source resolution names to key field names

    Returns:
        A set of ClusterEntity objects
    """
    # Convert polars to pandas for compatibility with existing logic
    if isinstance(data, pl.DataFrame | pa.Table):
        data = data.to_pandas()

    must_have_fields = set(["id"] + list(keys.values()))
    if not must_have_fields.issubset(data.columns):
        raise ValueError(
            f"Fields {must_have_fields.difference(data.columns)} must be included "
            "in the data and are missing."
        )

    def _create_cluster_entity(group: pd.DataFrame) -> ClusterEntity:
        entity_refs = {
            source: frozenset(group[key_field].dropna().values)
            for source, key_field in keys.items()
            if not group[key_field].dropna().empty
        }

        return ClusterEntity(
            id=group.name,
            keys=EntityReference(entity_refs),
        )

    result = data.groupby("id").apply(_create_cluster_entity, include_groups=False)
    return set(result.tolist())


@cache
def generate_entities(
    generator: Faker,
    features: tuple[FeatureConfig, ...],
    n: int,
) -> tuple[SourceEntity]:
    """Generate base entities with their ground truth values from generator."""
    entities = []
    for _ in range(n):
        base_values = {}
        for feature in features:
            generator_func = generator.unique if feature.unique else generator
            value_generator = getattr(generator_func, feature.base_generator)
            parameters = {} if not feature.parameters else dict(feature.parameters)
            base_values[feature.name] = value_generator(**parameters)

        entities.append(SourceEntity(base_values=base_values, keys=EntityReference()))
    return tuple(entities)


def probabilities_to_results_entities(
    probabilities: pa.Table,
    left_clusters: tuple[ClusterEntity, ...],
    right_clusters: tuple[ClusterEntity, ...] | None = None,
    threshold: float | int = 0,
) -> tuple[ClusterEntity, ...]:
    """Convert probabilities to ClusterEntity objects based on a threshold."""
    left_lookup = {entity.id: entity for entity in left_clusters}
    if right_clusters is not None:
        right_lookup = {entity.id: entity for entity in right_clusters}
    else:
        right_lookup = left_lookup

    djs = DisjointSet[ClusterEntity]()

    # Validate threshold
    if isinstance(threshold, float):
        threshold = int(threshold * 100)

    # Add ALL entities to the disjoint set
    for entity in left_clusters:
        djs.add(entity)
    if right_clusters is not None:
        for entity in right_clusters:
            djs.add(entity)

    # Add edges to the disjoint set
    for record in probabilities.to_pylist():
        if record["probability"] >= threshold:
            djs.union(
                left_lookup[record["left_id"]],
                right_lookup[record["right_id"]],
            )

    components: set[set[ClusterEntity]] = djs.get_components()

    entities: list[ClusterEntity] = []
    for component in components:
        merged: ClusterEntity = sum(component)
        entities.append(merged)

    return tuple(entities)


def diff_results(
    expected: list[ClusterEntity], actual: list[ClusterEntity]
) -> tuple[bool, dict]:
    """Compare two lists of ClusterEntity with detailed diff information.

    Args:
        expected: Expected ClusterEntity list
        actual: Actual ClusterEntity list

    Returns:
        A tuple containing:
        - Boolean: True if lists are identical, False otherwise
        - Dictionary that counts the number of actual entities that fall into the
            following criteria:
            - 'perfect': Match an expected entity exactly
            - 'subset': Are a subset of an expected entity
            - 'superset': Are a superset of an expected entity
            - 'wrong': Don't match any expected entity
            - 'invalid': Contain keys not present in any expected entity
    """
    expected_set, actual_set = set(expected), set(actual)
    if expected_set == actual_set:
        return True, {}

    all_expected = sum(expected_set)
    perfect_matches = expected_set & actual_set
    remaining_actual = actual_set - perfect_matches

    counter = Counter(
        {
            "perfect": len(perfect_matches),
            "subset": 0,
            "superset": 0,
            "wrong": 0,
            "invalid": 0,
        }
    )

    for a in remaining_actual:
        if any(a in e for e in expected_set):
            counter["subset"] += 1
        elif a not in all_expected:
            counter["invalid"] += 1
        elif any(e in a for e in expected_set):
            counter["superset"] += 1
        else:
            counter["wrong"] += 1

    return False, dict(counter)
