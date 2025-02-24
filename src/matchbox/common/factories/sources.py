"""Factories for generating dummy sources and linked sources for testing."""

import warnings
from functools import cache, wraps
from itertools import product
from typing import Callable, ParamSpec, TypeVar
from unittest.mock import Mock, create_autospec
from uuid import uuid4

import pandas as pd
import pyarrow as pa
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, create_engine

from matchbox.common.arrow import SCHEMA_INDEX
from matchbox.common.factories.entities import (
    EntityReference,
    FeatureConfig,
    ResultsEntity,
    SourceEntity,
    SuffixRule,
    diff_results,
    generate_entities,
    probabilities_to_results_entities,
)
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


class SourceConfig(BaseModel):
    """Configuration for generating a source."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    features: tuple[FeatureConfig, ...] = Field(default_factory=tuple)
    full_name: str
    engine: Engine = Field(default=create_engine("sqlite:///:memory:"))
    n_true_entities: int | None = Field(default=None)
    repetition: int = Field(default=0)


class SourceDummy(BaseModel):
    """Complete representation of a generated dummy Source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Source
    features: tuple[FeatureConfig, ...]
    data: pa.Table
    data_hashes: pa.Table
    true_entities: tuple[SourceEntity, ...] | None = Field(
        default=None,
        description=(
            "Generated true entities. Optional: when the SourceDummy comes from a "
            "source_factory they're stored here, but from linked_source_factory "
            "they're stored as part of the shared LinkedSourcesDummy object."
        ),
    )
    entities: tuple[ResultsEntity, ...] = Field(
        description="Entities that were generated from the source."
    )

    @property
    def name(self) -> str:
        """Return the full name of the source."""
        return self.source.address.full_name

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

    def query(self) -> pa.Table:
        """Return a PyArrow table in the same format at matchbox.query()."""
        return self.data


class LinkedSourcesDummy(BaseModel):
    """Container for multiple related sources with entity tracking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    true_entities: dict[int, SourceEntity] = Field(default_factory=dict)
    sources: dict[str, SourceDummy]

    def find_entities(
        self,
        min_appearances: dict[str, int] | None = None,
        max_appearances: dict[str, int] | None = None,
    ) -> list[SourceEntity]:
        """Find entities matching appearance criteria."""
        result = list(self.true_entities.values())

        def _meets_criteria(
            entity: SourceEntity, criteria: dict[str, int], compare: Callable
        ) -> bool:
            return all(
                compare(len(entity.get_source_pks(src)), count)
                for src, count in criteria.items()
            )

        if min_appearances:
            result = [
                entity
                for entity in result
                if _meets_criteria(entity, min_appearances, lambda x, y: x >= y)
            ]

        if max_appearances:
            result = [
                entity
                for entity in result
                if _meets_criteria(entity, max_appearances, lambda x, y: x <= y)
            ]

        return result

    def diff_results(
        self,
        probabilities: pa.Table,
        sources: list[str],
        left_results: tuple[ResultsEntity, ...],
        right_results: tuple[ResultsEntity, ...] | None = None,
        threshold: int | float = 0,
        verbose: bool = False,
    ) -> tuple[bool, str]:
        """Diff a results of probabilities with the true SourceEntities.

        Returns a tuple of:
            * Whether the results match the true entities
            * A message describing the result
        """
        return diff_results(
            expected=[
                entity.to_results_entity(*sources)
                for entity in self.true_entities.values()
            ],
            actual=probabilities_to_results_entities(
                probabilities=probabilities,
                left_results=left_results,
                right_results=right_results,
                threshold=threshold,
            ),
            verbose=verbose,
        )


def generate_rows(
    generator: Faker,
    selected_entities: tuple[SourceEntity, ...],
    features: tuple[FeatureConfig, ...],
) -> tuple[dict[str, list], dict[int, list[str]], dict[int, list[str]]]:
    """Generate raw data rows. Adds an ID shared by unique rows, and a PK for every row.

    Returns a tuple of:
    * raw_data: Dictionary of column arrays for DataFrame creation
    * entity_pks: Maps SourceEntity.id to the set of PKs where that entity appears
    * id_pks: Maps each ID to the set of PKs where that row appears

    For example, if this is the raw data:

    | id | pk | company_name |
    |----|----|--------------|
    | 1  | 1  | alpha co     |
    | 2  | 2  | alpha ltd    |
    | 1  | 3  | alpha co     |
    | 2  | 4  | alpha ltd    |
    | 3  | 5  | beta co      |
    | 4  | 6  | beta ltd     |
    | 3  | 7  | beta co      |
    | 4  | 8  | beta ltd     |


    Entity PKs would be this, because there are two true SourceEntities:

    {
        1: [1, 2, 3, 4],
        2: [5, 6, 7, 8],
    }

    And ID PKs would be this, because there are four unique rows:

    {
        1: [1, 3],
        2: [2, 4],
        3: [5, 7],
        4: [6, 8],
    }
    """
    raw_data = {"pk": [], "id": []}
    for feature in features:
        raw_data[feature.name] = []

    # Track entity locations and row identities
    entity_pks = {entity.id: [] for entity in selected_entities}
    id_pks = {}
    value_to_id = {}

    def add_row(entity_id: int, values: tuple) -> None:
        """Add a row of data, handling IDs and PKs."""
        pk = str(generator.uuid4())
        entity_pks[entity_id].append(pk)

        if values not in value_to_id:
            mb_id = generator.random_number(digits=16)
            value_to_id[values] = mb_id
            id_pks[mb_id] = []

        row_id = value_to_id[values]
        id_pks[row_id].append(pk)

        raw_data["pk"].append(pk)
        raw_data["id"].append(row_id)
        for feature, value in zip(features, values, strict=True):
            raw_data[feature.name].append(value)

    for entity in selected_entities:
        # For each feature, collect all possible values
        possible_values = []
        for feature in features:
            base = entity.base_values[feature.name]

            variations = []
            # Apply all variations as long as they change the value
            for v in (rule.apply(base) for rule in feature.variations):
                if v != base:
                    variations.append(v)

            values = variations if feature.drop_base else variations + [base]
            possible_values.append(values or [base])

        # Create a row for each combination
        for values in product(*possible_values):
            add_row(entity.id, values)

    return raw_data, entity_pks, id_pks


@cache
def generate_source(
    generator: Faker,
    n_true_entities: int,
    features: tuple[FeatureConfig, ...],
    repetition: int,
    seed_entities: tuple[SourceEntity, ...] | None = None,
) -> tuple[pa.Table, pa.Table, dict[int, set[str]], dict[int, set[str]]]:
    """Generate raw data as PyArrow tables with entity tracking.

    Returns:
        - data: PyArrow table with generated data
        - data_hashes: PyArrow table with hash groups
        - entity_pks: SourceEntity ID -> list of PKs mapping
        - row_pks: Results row ID -> list of PKs mapping for identical rows
    """
    # Select or generate entities
    if seed_entities is None:
        selected_entities = generate_entities(generator, features, n_true_entities)
    else:
        selected_entities = generator.random_elements(
            elements=seed_entities,
            unique=True,
            length=min(n_true_entities, len(seed_entities)),
        )

    # Generate initial data
    raw_data, entity_pks, row_pks = generate_rows(
        generator, selected_entities, features
    )

    # Create DataFrame
    df = pd.DataFrame(raw_data)

    # Handle repetition
    df = pd.concat([df] * (repetition + 1), ignore_index=True)
    entity_pks = {eid: pks * (repetition + 1) for eid, pks in entity_pks.items()}
    row_pks = {rid: pks * (repetition + 1) for rid, pks in row_pks.items()}

    # Create hash groups
    source_pks = []
    hashes = []
    for group_pks in row_pks.values():
        source_pks.append(list(group_pks))
        hashes.append(str(uuid4()).encode())

    data_hashes = pa.Table.from_pydict(
        {
            "source_pk": source_pks,
            "hash": hashes,
        },
        schema=SCHEMA_INDEX,
    )

    # Update variation counts
    for entity in selected_entities:
        if entity.id in entity_pks:
            # Count unique row IDs this entity appears in
            entity_rows = df[df["pk"].isin(entity_pks[entity.id])]
            entity.total_unique_variations = len(set(entity_rows["id"]))

    return (
        pa.Table.from_pandas(df, preserve_index=False),
        data_hashes,
        entity_pks,
        row_pks,
    )


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
    """Generate a complete dummy source."""
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
        n=n_true_entities,
    )

    # Generate data using the base entities
    data, data_hashes, entity_pks, row_pks = generate_source(
        generator=generator,
        n_true_entities=n_true_entities,
        features=features,
        repetition=repetition,
        seed_entities=base_entities,
    )

    # Create source entities with references
    source_entities = []
    for entity in base_entities:
        pks = entity_pks.get(entity.id, [])
        if pks:
            entity.add_source_reference(full_name, pks)
            source_entities.append(entity)

    # Create ResultsEntity objects from row_pks
    results_entities = [
        ResultsEntity(
            id=row_id,
            source_pks=EntityReference(
                mapping=frozenset([(full_name, frozenset(pks))])
            ),
        )
        for row_id, pks in row_pks.items()
    ]

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
        true_entities=tuple(source_entities),
        entities=tuple(sorted(results_entities)),
    )


@cache
def linked_sources_factory(
    source_configs: tuple[SourceConfig, ...] | None = None,
    n_true_entities: int | None = None,
    seed: int = 42,
) -> LinkedSourcesDummy:
    """Generate a set of linked sources with tracked entities.

    Args:
        source_configs: Optional tuple of source configurations
        n_true_entities: Optional number of true entities to generate. If provided,
            overrides any n_true_entities in source configs. If not provided, each
            SourceConfig must specify its own n_true_entities.
        seed: Random seed for reproducibility
    """
    generator = Faker()
    generator.seed_instance(seed)

    if source_configs is None:
        # Use factory parameter or default for default configs
        n_true_entities = n_true_entities if n_true_entities is not None else 10

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
                        SuffixRule(suffix=" Limited"),
                        SuffixRule(suffix=" UK"),
                        SuffixRule(suffix=" Company"),
                    ),
                    features["crn"],
                ),
                drop_base=True,
                n_true_entities=n_true_entities,
                repetition=0,
            ),
            SourceConfig(
                full_name="duns",
                features=(
                    features["company_name"],
                    features["duns"],
                ),
                n_true_entities=n_true_entities // 2,
                repetition=0,
            ),
            SourceConfig(
                full_name="cdms",
                features=(
                    features["crn"],
                    features["cdms"],
                ),
                n_true_entities=n_true_entities,
                repetition=1,
            ),
        )
    else:
        if n_true_entities is not None:
            # Factory parameter provided - warn if configs have values set
            config_entities = [config.n_true_entities for config in source_configs]
            if any(n is not None for n in config_entities):
                warnings.warn(
                    "Both source configs and linked_sources_factory specify "
                    "n_true_entities. The factory parameter will be used.",
                    UserWarning,
                    stacklevel=2,
                )
            # Override all configs with factory parameter
            source_configs = tuple(
                SourceConfig(
                    full_name=config.full_name,
                    engine=config.engine,
                    features=config.features,
                    repetition=config.repetition,
                    n_true_entities=n_true_entities,
                )
                for config in source_configs
            )
        else:
            # No factory parameter - check all configs have n_true_entities set
            missing_counts = [
                config.full_name
                for config in source_configs
                if config.n_true_entities is None
            ]
            if missing_counts:
                raise ValueError(
                    "n_true_entities not set for sources: "
                    f"{', '.join(missing_counts)}. When factory n_true_entities is "
                    "not provided, all configs must specify it."
                )

    # Collect all unique features
    all_features = set()
    for config in source_configs:
        all_features.update(config.features)
    all_features = tuple(sorted(all_features, key=lambda f: f.name))

    # Find maximum number of entities needed across all sources
    max_entities = max(config.n_true_entities for config in source_configs)

    # Generate all possible entities
    all_entities = generate_entities(
        generator=generator, features=all_features, n=max_entities
    )

    # Initialize LinkedSourcesDummy
    linked = LinkedSourcesDummy(
        true_entities={entity.id: entity for entity in all_entities},
        sources={},
    )

    # Generate sources
    for config in source_configs:
        # Generate source data using seed entities
        data, data_hashes, entity_pks, row_pks = generate_source(
            generator=generator,
            features=tuple(config.features),
            n_true_entities=config.n_true_entities,
            repetition=config.repetition,
            seed_entities=all_entities,
        )

        # Create ResultsEntity objects from row_pks
        results_entities = [
            ResultsEntity(
                id=row_id,
                source_pks=EntityReference(
                    mapping=frozenset([(config.full_name, frozenset(pks))])
                ),
            )
            for row_id, pks in row_pks.items()
        ]

        # Create source
        source = Source(
            address=SourceAddress.compose(
                full_name=config.full_name, engine=config.engine
            ),
            db_pk="pk",
            columns=[SourceColumn(name=feature.name) for feature in config.features],
        )

        # Add source to linked.sources
        linked.sources[config.full_name] = SourceDummy(
            source=source,
            features=tuple(config.features),
            data=data,
            data_hashes=data_hashes,
            entities=tuple(sorted(results_entities)),
        )

        # Update entities with source references
        for entity_id, pks in entity_pks.items():
            entity = linked.true_entities[entity_id]
            entity.add_source_reference(config.full_name, pks)

    return linked


if __name__ == "__main__":
    pass
