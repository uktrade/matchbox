from abc import ABC, abstractmethod
from functools import cache, wraps
from typing import Any, Callable, ParamSpec, TypeVar
from unittest.mock import Mock, create_autospec
from uuid import UUID, uuid4

import pandas as pd
import pyarrow as pa
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, create_engine

from matchbox.common.arrow import SCHEMA_INDEX
from matchbox.common.sources import Source, SourceAddress, SourceColumn

P = ParamSpec("P")
R = TypeVar("R")


def make_features_hashable(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Handle features in first positional arg
        if args and args[0] is not None:
            if isinstance(args[0][0], dict):
                args = (tuple(FeatureConfig(**d) for d in args[0]),) + args[1:]
            else:
                args = (tuple(args[0]),) + args[1:]

        # Handle features in kwargs
        if "features" in kwargs and kwargs["features"] is not None:
            if isinstance(kwargs["features"][0], dict):
                kwargs["features"] = tuple(
                    FeatureConfig(**d) for d in kwargs["features"]
                )
            else:
                kwargs["features"] = tuple(kwargs["features"])

        return func(*args, **kwargs)

    return wrapper


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
        return f"{value}{self.suffix}"

    @property
    def type(self) -> str:
        return "suffix"


class PrefixRule(VariationRule):
    """Add a prefix to a value."""

    prefix: str

    def apply(self, value: str) -> str:
        return f"{self.prefix}{value}"

    @property
    def type(self) -> str:
        return "prefix"


class ReplaceRule(VariationRule):
    """Replace occurrences of a string with another."""

    old: str
    new: str

    def apply(self, value: str) -> str:
        return value.replace(self.old, self.new)

    @property
    def type(self) -> str:
        return "replace"


class DropBaseRule(VariationRule):
    """Drop the base value."""

    drop: bool = Field(default=True)

    def apply(self, value: str) -> str:
        return value

    @property
    def type(self) -> str:
        return "drop_base"


class FeatureConfig(BaseModel):
    """Configuration for generating a feature with variations."""

    model_config = ConfigDict(frozen=True)

    name: str
    base_generator: str
    parameters: tuple = Field(default_factory=tuple)
    unique: bool = Field(default=True)
    variations: tuple[VariationRule, ...] = Field(default_factory=tuple)

    def add_variations(self, *rule: VariationRule) -> "FeatureConfig":
        """Add a variation rule to the feature."""
        return FeatureConfig(
            name=self.name,
            base_generator=self.base_generator,
            parameters=self.parameters,
            variations=self.variations + tuple(rule),
        )


class SourceConfig(BaseModel):
    """Configuration for generating a source."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    features: tuple[FeatureConfig, ...] = Field(default_factory=tuple)
    full_name: str
    engine: Engine = Field(default=create_engine("sqlite:///:memory:"))
    n_entities: int = Field(default=10)
    repetition: int = Field(default=0)


class SourceEntityReference(BaseModel):
    """Reference to an entity's presence in a specific source."""

    name: str
    source_pks: tuple[str, ...]

    model_config = ConfigDict(frozen=True)


class SourceEntity(BaseModel):
    """Represents a single entity across all sources."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    base_values: dict[str, Any] = Field(description="Feature name -> base value")
    source_pks: tuple[SourceEntityReference, ...] = Field(
        default_factory=tuple, description="Source references containing PKs"
    )
    total_unique_variations: int = Field(default=0)

    def __eq__(self, other: "SourceEntity") -> bool:
        """Entities are equal if they share base values."""
        return self.base_values == other.base_values

    def __hash__(self) -> int:
        """Hash based on sorted base values."""
        return hash(tuple(sorted(self.base_values.items())))

    def add_source_reference(self, name: str, pks: list[str]) -> None:
        """Add or update a source reference."""
        new_ref = SourceEntityReference(name=name, source_pks=tuple(pks))
        existing_refs = [ref for ref in self.source_pks if ref.name != name]
        self.source_pks = tuple(existing_refs + [new_ref])

    def get_source_pks(self, source_name: str) -> list[str]:
        """Get PKs for a specific source."""
        for ref in self.source_pks:
            if ref.name == source_name:
                return list(ref.source_pks)
        return []

    def variations(
        self, sources: dict[str, "SourceDummy"]
    ) -> dict[str, dict[str, dict[str, str]]]:
        """Get all variations of this entity across sources with their rules."""
        variations = {}
        for ref in self.source_pks:
            source = sources.get(ref.name)

            if source is None:
                continue

            # Initialize source variations
            variations[ref.name] = {}

            # For each feature, track variations and their origins
            for feature in source.features:
                feature_variations = {}
                base_value = self.base_values[feature.name]

                # Add base value
                feature_variations[base_value] = "Base value"

                # Add variations from rules
                for i, rule in enumerate(feature.variations):
                    varied_value = rule.apply(base_value)
                    feature_variations[varied_value] = (
                        f"Variation {i + 1}: {rule.model_dump()}"
                    )

                variations[ref.name][feature.name] = feature_variations

        return variations

    def get_values(
        self, sources: dict[str, "SourceDummy"]
    ) -> dict[str, dict[str, list[str]]]:
        """Get all unique values for this entity across sources."""
        values = {}
        for ref in self.source_pks:
            source = sources[ref.name]
            df = source.data.to_pandas()

            # Get rows for this entity
            entity_rows = df[df["pk"].isin(ref.source_pks)]

            # Get unique values for each feature
            values[ref.name] = {
                feature.name: sorted(entity_rows[feature.name].unique())
                for feature in source.features
            }

        return values


class SourceDummy(BaseModel):
    """Complete representation of a generated dummy Source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Source
    features: tuple[FeatureConfig, ...]
    data: pa.Table
    data_hashes: pa.Table
    entities: tuple[SourceEntity, ...] | None = Field(
        default=None,
        description=(
            "Generated entities. Optional: when the SourceDummy comes from a "
            "source_factory they're stored here, but from linked_source_factory "
            "they're stored as part of the shared LinkedSourcesDummy object."
        ),
    )

    def to_mock(self) -> Mock:
        """Create a mock Source object that mimics this dummy source's behavior."""
        mock_source = create_autospec(self.source)

        mock_source.set_engine.return_value = mock_source
        mock_source.default_columns.return_value = mock_source
        mock_source.hash_data.return_value = self.data_hashes
        mock_source.to_table = self.data

        mock_source.model_dump.side_effect = self.source.model_dump
        mock_source.model_dump_json.side_effect = self.source.model_dump_json

        return mock_source


class LinkedSourcesDummy(BaseModel):
    """Container for multiple related sources with entity tracking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entities: dict[UUID, SourceEntity] = Field(default_factory=dict)
    sources: dict[str, SourceDummy]

    def find_entities(
        self,
        min_appearances: dict[str, int] | None = None,
        max_appearances: dict[str, int] | None = None,
    ) -> list[SourceEntity]:
        """Find entities matching appearance criteria."""
        result = list(self.entities.values())

        if min_appearances:
            result = [
                e
                for e in result
                if all(
                    len(e.get_source_pks(src)) >= count
                    for src, count in min_appearances.items()
                )
            ]

        if max_appearances:
            result = [
                e
                for e in result
                if all(
                    len(e.get_source_pks(src)) <= count
                    for src, count in max_appearances.items()
                )
            ]

        return result


@cache
def generate_entities(
    generator: Faker,
    features: tuple[FeatureConfig, ...],
    n_entities: int,
) -> tuple[SourceEntity]:
    """Generate base entities with their ground truth values."""
    entities = []
    for _ in range(n_entities):
        base_values = {
            f.name: getattr(
                generator.unique if f.unique else generator, f.base_generator
            )(**dict(f.parameters))
            for f in features
        }
        entities.append(SourceEntity(base_values=base_values))
    return tuple(entities)


@cache
def generate_source(
    generator: Faker,
    n_true_entities: int,
    features: tuple[FeatureConfig, ...],
    repetition: int,
    seed_entities: tuple[SourceEntity, ...] | None = None,
) -> tuple[pa.Table, pa.Table, dict[UUID, list[str]]]:
    """Generate raw data as PyArrow tables with entity tracking."""
    # Generate or select entities
    if seed_entities is None:
        selected_entities = generate_entities(generator, features, n_true_entities)
    else:
        selected_entities = generator.random.sample(
            seed_entities, min(n_true_entities, len(seed_entities))
        )

    # Calculate variations (excluding DropBaseRule)
    max_variations = max(
        sum(1 for rule in f.variations if not isinstance(rule, DropBaseRule))
        for f in features
    )

    raw_data = {"pk": []}
    for feature in features:
        raw_data[feature.name] = []

    # Track PKs for each entity
    entity_pks: dict[UUID, list[str]] = {entity.id: [] for entity in selected_entities}

    # Generate data for each selected entity
    for entity in selected_entities:
        # Check which features should include base values
        features_with_base = [
            f
            for f in features
            if not any(isinstance(rule, DropBaseRule) for rule in f.variations)
        ]

        # Add base values for all features at once
        if features_with_base:
            pk = str(uuid4())
            raw_data["pk"].append(pk)
            entity_pks[entity.id].append(pk)

            for feature in features:
                raw_data[feature.name].append(entity.base_values[feature.name])

        # Add variations
        for variation_idx in range(max_variations):
            pk = str(uuid4())
            raw_data["pk"].append(pk)
            entity_pks[entity.id].append(pk)

            for feature in features:
                non_drop_variations = [
                    rule
                    for rule in feature.variations
                    if not isinstance(rule, DropBaseRule)
                ]
                if variation_idx < len(non_drop_variations):
                    value = non_drop_variations[variation_idx].apply(
                        entity.base_values[feature.name]
                    )
                else:
                    value = entity.base_values[feature.name]
                raw_data[feature.name].append(value)

    # Create DataFrame
    df = pd.DataFrame(raw_data)

    # Remove rows with base values for features that have DropBaseRule
    drop_base_features = [
        feature.name
        for feature in features
        if any(isinstance(rule, DropBaseRule) for rule in feature.variations)
    ]

    if drop_base_features:
        # Track which rows we're removing to update entity_pks
        initial_pks = set(df["pk"])

        for entity in selected_entities:
            for feature_name in drop_base_features:
                base_value = entity.base_values[feature_name]
                df = df[df[feature_name] != base_value]

        # Update entity_pks to remove dropped PKs
        remaining_pks = set(df["pk"])
        dropped_pks = initial_pks - remaining_pks

        for entity_id in entity_pks:
            entity_pks[entity_id] = [
                pk for pk in entity_pks[entity_id] if pk not in dropped_pks
            ]

    # Apply repetition
    if repetition:
        df = pd.concat([df] * repetition, ignore_index=True)

    # Update entity PKs for repetition + 1 (repetitions + original)
    for entity_id in entity_pks:
        entity_pks[entity_id] = entity_pks[entity_id] * (repetition + 1)

    # Create hash groups and data_hashes table
    feature_names = [f.name for f in features]
    hash_groups = df.groupby(feature_names, sort=False)["pk"].agg(list).reset_index()
    hash_groups["hash"] = [str(uuid4()).encode() for _ in range(len(hash_groups))]

    data_hashes = pa.Table.from_pydict(
        {
            "source_pk": hash_groups["pk"].tolist(),
            "hash": hash_groups["hash"].tolist(),
        },
        schema=SCHEMA_INDEX,
    )

    # Update entities with variations count
    feature_cols = [col for col in df.columns if col != "pk"]
    for entity in selected_entities:
        entity_rows = df[df["pk"].isin(entity_pks[entity.id])]
        entity.total_unique_variations = (
            entity_rows[feature_cols].drop_duplicates().shape[0]
        )

    return pa.Table.from_pandas(df), data_hashes, entity_pks


@make_features_hashable
@cache
def source_factory(
    features: list[FeatureConfig] | list[dict] | None = None,
    full_name: str | None = None,
    engine: Engine | None = None,
    n_true_entities: int = 10,
    repetition: int = 0,
    seed: int = 42,
) -> SourceDummy:
    """Generate a complete dummy source.

    Args:
        features: list of FeatureConfigs, used to generate features with variations
        full_name: Full name of the source, like "dbt.companies_house".
        engine: SQLAlchemy engine to use for the source.
        n_true_entities: Number of true entities to generate.
        repetition: Number of times to repeat the data.
        seed: Random seed for data generation.

    Returns:
        SourceDummy: Complete dummy source with generated data, including entities.
    """
    generator = Faker()
    generator.seed_instance(seed)

    if features is None:
        features = (
            FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
        )

    if full_name is None:
        full_name = generator.unique.word()

    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    # Generate base entities
    base_entities = generate_entities(
        generator=generator,
        features=features,
        n_entities=n_true_entities,
    )

    # Generate data using the base entities
    data, data_hashes, entity_pks = generate_source(
        generator=generator,
        n_true_entities=n_true_entities,
        features=features,
        repetition=repetition,
        seed_entities=base_entities,
    )

    # Create source entities with references
    source_entities = []
    for entity in base_entities:
        # Get PKs for this entity if they exist
        pks = entity_pks.get(entity.id, [])
        if pks:
            # Add source reference
            entity.add_source_reference(full_name, pks)
            source_entities.append(entity)

    source = Source(
        address=SourceAddress.compose(full_name=full_name, engine=engine),
        db_pk="pk",
        columns=[SourceColumn(name=feature.name) for feature in features],
    )

    return SourceDummy(
        source=source,
        features=features,
        data=data,
        data_hashes=data_hashes,
        entities=tuple(source_entities),
    )


@cache
def linked_sources_factory(
    source_configs: tuple[SourceConfig, ...] | None = None,
    n_entities: int = 10,
    seed: int = 42,
) -> LinkedSourcesDummy:
    """Generate a set of linked sources with tracked entities.

    Args:
        source_configs: Configurations for generating sources. If None, a default
            set of configurations will be used.
        n_entities: Base number of entities to generate when using default configs.
            Ignored if source_configs is provided.
        seed: Random seed for data generation.

    Returns:
        LinkedSourcesDummy: Container for generated sources and entities.
    """
    generator = Faker()
    generator.seed_instance(seed)

    if source_configs is None:
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

        engine = create_engine("sqlite:///:memory:")

        source_configs = (
            SourceConfig(
                full_name="crn",
                engine=engine,
                features=(
                    features["company_name"].add_variations(
                        DropBaseRule(),
                        SuffixRule(suffix=" Limited"),
                        SuffixRule(suffix=" UK"),
                        SuffixRule(suffix=" Company"),
                    ),
                    features["crn"],
                ),
                n_entities=n_entities,
                repetition=0,
            ),
            SourceConfig(
                full_name="duns",
                features=(
                    features["company_name"],
                    features["duns"],
                ),
                n_entities=n_entities // 2,
                repetition=0,
            ),
            SourceConfig(
                full_name="cdms",
                features=(
                    features["crn"],
                    features["cdms"],
                ),
                n_entities=n_entities,
                repetition=1,
            ),
        )

    # Collect all unique features
    all_features = set()
    for config in source_configs:
        all_features.update(config.features)
    all_features = tuple(sorted(all_features, key=lambda f: f.name))

    # Find maximum number of entities needed
    max_entities = max(config.n_entities for config in source_configs)

    # Generate all possible entities
    all_entities = generate_entities(
        generator=generator, features=all_features, n_entities=max_entities
    )

    # Initialize LinkedSourcesDummy
    linked = LinkedSourcesDummy(
        entities={entity.id: entity for entity in all_entities},
        sources={},
    )

    # Generate sources
    for config in source_configs:
        # Generate source data using seed entities
        data, data_hashes, entity_pks = generate_source(
            generator=generator,
            features=tuple(config.features),
            n_true_entities=config.n_entities,
            repetition=config.repetition,
            seed_entities=all_entities,
        )

        # Create source
        source = Source(
            address=SourceAddress.compose(
                full_name=config.full_name, engine=config.engine
            ),
            db_pk="pk",
            columns=[SourceColumn(name=feature.name) for feature in config.features],
        )

        # Add source directly to linked.sources
        linked.sources[config.full_name] = SourceDummy(
            source=source,
            features=tuple(config.features),
            data=data,
            data_hashes=data_hashes,
        )

        # Update entities with source references
        for entity_id, pks in entity_pks.items():
            entity = linked.entities[entity_id]
            entity.add_source_reference(config.full_name, pks)

    return linked


if __name__ == "__main__":
    pass
