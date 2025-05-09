"""Factories for generating sources and linked source testkits for testing."""

import warnings
from functools import cache, wraps
from itertools import product
from typing import Any, Callable, ParamSpec, TypeVar
from unittest.mock import Mock, create_autospec

import pandas as pd
import pyarrow as pa
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field

from matchbox.common.arrow import SCHEMA_INDEX
from matchbox.common.dtos import DataTypes, SourceResolutionName
from matchbox.common.factories.entities import (
    ClusterEntity,
    EntityReference,
    FeatureConfig,
    SourceEntity,
    SuffixRule,
    convert_data_type,
    diff_results,
    generate_entities,
    probabilities_to_results_entities,
)
from matchbox.common.factories.locations import (
    LocationConfig,
    RelationalDBConfig,
    RelationalDBConfigOptions,
)
from matchbox.common.hash import hash_values
from matchbox.common.sources import (
    LocationType,
    RelationalDBLocation,  # noqa: F401
    SourceConfig,
    SourceField,
)

P = ParamSpec("P")
R = TypeVar("R")


def make_features_hashable(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to allow configuring source_factory with dicts.

    This retains the hashability of FeatureConfig while still making it simple
    to use the factory without special objects.
    """

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


class SourceTestkitConfig(BaseModel):
    """Configuration for generating a source."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: SourceResolutionName
    features: tuple[FeatureConfig, ...] = Field(default_factory=tuple)
    location_config: LocationConfig | None = Field(
        default=RelationalDBConfig(),
        description=(
            "Location configuration for the source. If not provided, "
            "a default in-memory SQLite database will be used."
        ),
    )
    n_true_entities: int | None = Field(default=None)
    repetition: int = Field(default=0)


class SourceTestkit(BaseModel):
    """A testkit of data and metadata for a SourceConfig."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SourceConfig = Field(description="The real generated SourceConfig object.")
    features: tuple[FeatureConfig, ...] | None = Field(
        description=(
            "The features used to generate the data. "
            "If None, the source data was not generated, but set manually."
        ),
        default=None,
    )
    data: pa.Table = Field(description="The PyArrow table of generated data.")
    data_hashes: pa.Table = Field(description="A PyArrow table of hashes for the data.")
    entities: tuple[ClusterEntity, ...] = Field(
        description="ClusterEntities that were generated from the source."
    )
    location_writer: Callable[[pa.Table, LocationType], None] = Field(
        description=(
            "A function that takes the data and location and writes the data to "
            "the location."
        )
    )

    @property
    def name(self) -> str:
        """Return the resolution name of the SourceConfig."""
        return self.config.name

    @property
    def mock(self) -> Mock:
        """Create a mock SourceConfig object with this testkit's configuration."""
        mock_source_config = create_autospec(self.config)

        mock_source_config.set_credentials.return_value = mock_source_config
        mock_source_config.from_location.return_value = mock_source_config
        mock_source_config.hash_data.return_value = self.data_hashes

        mock_source_config.model_dump.side_effect = self.config.model_dump
        mock_source_config.model_dump_json.side_effect = self.config.model_dump_json

        return mock_source_config

    @property
    def query(self) -> pa.Table:
        """Return a PyArrow table in the same format as matchbox.query()."""
        return self.data

    @property
    def query_backend(self) -> pa.Table:
        """Return a PyArrow table in the same format as the SCHEMA_MB_IDS DTO."""
        return pa.Table.from_arrays(
            [self.data["id"], self.data["identifier"]],
            names=["id", "source_identifier"],
        )

    def set_credentials(self, credentials: Any) -> None:
        """Set the credentials for the SourceConfig."""
        self.config.set_credentials(credentials)

    def write_to_location(
        self, credentials: Any, set_credentials: bool = False
    ) -> None:
        """Write the data to the SourceConfig's location.

        Credentials aren't set in testkits, so they must be provided here.

        Args:
            credentials: Credentials to use for the location.
            set_credentials: Whether to set the credentials on the SourceConfig.
                Offered here for convenience as it's often the next step.
        """
        self.location_writer(self.data, self.config.location, credentials)
        if set_credentials:
            self.set_credentials(credentials)


class LinkedSourcesTestkit(BaseModel):
    """Container for multiple related SourceConfig testkits with entity tracking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    true_entities: set[SourceEntity] = Field(default_factory=set)
    sources: dict[str, SourceTestkit]

    def find_entities(
        self,
        min_appearances: dict[str, int] | None = None,
        max_appearances: dict[str, int] | None = None,
    ) -> list[SourceEntity]:
        """Find entities matching appearance criteria."""
        result = list(self.true_entities)

        def _meets_criteria(
            entity: SourceEntity, criteria: dict[str, int], compare: Callable
        ) -> bool:
            return all(
                compare(len(entity.get_source_identifiers(src)), count)
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

    def true_entity_subset(self, *sources: str) -> list[ClusterEntity]:
        """Return a subset of true entities that appear in the given sources."""
        cluster_entities = [
            entity.to_cluster_entity(*sources) for entity in self.true_entities
        ]
        return [entity for entity in cluster_entities if entity is not None]

    def diff_results(
        self,
        probabilities: pa.Table,
        sources: list[str],
        left_clusters: tuple[ClusterEntity, ...],
        right_clusters: tuple[ClusterEntity, ...] | None = None,
        threshold: int | float = 0,
    ) -> tuple[bool, dict]:
        """Diff a results of probabilities with the true SourceEntities.

        Args:
            probabilities: Probabilities table to diff
            sources: Subset of the LinkedSourcesTestkit.sources that represents
                the true sources to compare against
            left_clusters: ClusterEntity objects from the object used as an input
                to the process that produced the probabilities table. Should
                be a SourceTestkit.entities or ModelTestkit.entities.
            right_clusters: ClusterEntity objects from the object used as an input
                to the process that produced the probabilities table. Should
                be a SourceTestkit.entities or ModelTestkit.entities.
            threshold: Threshold for considering a match true

        Returns:
            A tuple of whether the results are identical, and a report dictionary.
                See [`diff_results()`][matchbox.common.factories.entities.diff_results]
                for the report format.
        """
        return diff_results(
            expected=self.true_entity_subset(*sources),
            actual=probabilities_to_results_entities(
                probabilities=probabilities,
                left_clusters=left_clusters,
                right_clusters=right_clusters,
                threshold=threshold,
            ),
        )


def generate_rows(
    generator: Faker,
    selected_entities: tuple[SourceEntity, ...],
    features: tuple[FeatureConfig, ...],
) -> tuple[
    dict[str, list], dict[int, list[str]], dict[int, list[str]], dict[int, bytes]
]:
    """Generate raw data rows. Adds an ID shared by unique rows, and a PK for every row.

    Returns a tuple of:

    * raw_data: Dictionary of column arrays for DataFrame creation
    * entity_identifiers: Maps SourceEntity.id to the set of identifiers where that
        entity appears
    * id_identifiers: Maps each ID to the set of identifiers where that row appears
    * id_hashes: Maps each ID to its hash value

    For example, if this is the raw data:

    | id | identifier | company_name |
    |----|------------|--------------|
    | 1  | 1          | alpha co     |
    | 2  | 2          | alpha ltd    |
    | 1  | 3          | alpha co     |
    | 2  | 4          | alpha ltd    |
    | 3  | 5          | beta co      |
    | 4  | 6          | beta ltd     |
    | 3  | 7          | beta co      |
    | 4  | 8          | beta ltd     |


    Entity identifiers would be this, because there are two true SourceEntities:

    {
        1: [1, 2, 3, 4],
        2: [5, 6, 7, 8],
    }

    And ID identifiers would be this, because there are four unique rows:

    {
        1: [1, 3],
        2: [2, 4],
        3: [5, 7],
        4: [6, 8],
    }
    """
    raw_data = {"identifier": [], "id": []}
    for feature in features:
        raw_data[feature.name] = []

    # Track entity locations and row identities
    entity_identifiers = {entity.id: [] for entity in selected_entities}
    id_identifiers = {}
    id_hashes = {}
    value_to_id = {}

    def add_row(entity_id: int, values: tuple) -> None:
        """Add a row of data, handling IDs and identifiers."""
        identifier = str(generator.uuid4())
        entity_identifiers[entity_id].append(identifier)
        row_hash = hash_values(*(str(v) for v in values))

        if values not in value_to_id:
            mb_id = generator.random_number(digits=16)
            value_to_id[values] = mb_id
            id_identifiers[mb_id] = []
            id_hashes[mb_id] = row_hash

        row_id = value_to_id[values]
        id_identifiers[row_id].append(identifier)

        raw_data["identifier"].append(identifier)
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

    return raw_data, entity_identifiers, id_identifiers, id_hashes


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
        - entity_identifiers: SourceEntity ID -> list of PKs mapping
        - row_identifiers: Results row ID -> list of PKs mapping for identical rows
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
    raw_data, entity_identifiers, row_identifiers, id_hashes = generate_rows(
        generator, selected_entities, features
    )

    # Create DataFrame
    df = pd.DataFrame(raw_data)

    # Handle repetition
    df = pd.concat([df] * (repetition + 1), ignore_index=True)
    entity_identifiers = {
        eid: identifiers * (repetition + 1)
        for eid, identifiers in entity_identifiers.items()
    }
    row_identifiers = {
        rid: identifiers * (repetition + 1)
        for rid, identifiers in row_identifiers.items()
    }

    # Create hash groups
    source_identifiers = []
    hashes = []
    for row_id, group_identifiers in row_identifiers.items():
        source_identifiers.append(list(group_identifiers))
        hashes.append(id_hashes[row_id])

    data_hashes = pa.Table.from_pydict(
        {
            "source_identifier": source_identifiers,
            "hash": hashes,
        },
        schema=SCHEMA_INDEX,
    )

    # Update variation counts
    for entity in selected_entities:
        if entity.id in entity_identifiers:
            # Count unique row IDs this entity appears in
            entity_rows = df[df["identifier"].isin(entity_identifiers[entity.id])]
            entity.total_unique_variations = len(set(entity_rows["id"]))

    return (
        pa.Table.from_pandas(df, preserve_index=False),
        data_hashes,
        entity_identifiers,
        row_identifiers,
    )


@make_features_hashable
@cache
def source_factory(
    features: list[FeatureConfig] | list[dict] | None = None,
    name: SourceResolutionName | None = None,
    location_config: LocationConfig | None = None,
    n_true_entities: int = 10,
    repetition: int = 0,
    seed: int = 42,
) -> SourceTestkit:
    """Generate a complete source testkit from configured features.

    Args:
        features: Optional list of feature configurations. If not provided,
            defaults to a set of common features.
        name: Optional resolution name for the source. If not provided,
            a unique name will be generated.
        location_config: Optional location configuration. If not provided,
            defaults to an in-memory SQLite database.
        n_true_entities: Number of true entities to generate.
        repetition: Number of times to repeat the data.
        seed: Random seed for reproducibility.

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

    if name is None:
        name = generator.unique.word()

    if location_config is None:
        location_config = RelationalDBConfig(
            location_options=RelationalDBConfigOptions(
                table_strategy="single",
                table_mapping=None,
            ),
            uri="sqlite:///:memory:",
        )

    # Generate base entities
    base_entities = generate_entities(
        generator=generator,
        features=features,
        n=n_true_entities,
    )

    # Generate data using the base entities
    data, data_hashes, entity_identifiers, row_identifiers = generate_source(
        generator=generator,
        n_true_entities=n_true_entities,
        features=features,
        repetition=repetition,
        seed_entities=base_entities,
    )

    # Create source entities with references
    source_entities = []
    for entity in base_entities:
        identifiers = entity_identifiers.get(entity.id, [])
        if identifiers:
            entity.add_source_reference(name, identifiers)
            source_entities.append(entity)

    # Create ClusterEntity objects from row_identifiers
    results_entities = [
        ClusterEntity(
            id=row_id,
            source_identifiers=EntityReference({name: frozenset(identifiers)}),
        )
        for row_id, identifiers in row_identifiers.items()
    ]

    # Create identifier and fields
    identifier: SourceField = SourceField(
        name="identifier",
        type=DataTypes.STRING,
    )
    fields: tuple[SourceField, ...] = tuple(
        SourceField(name=feature.name, type=feature.datatype) for feature in features
    )

    # Create the extract/transform string and location writer
    extract_transform, location_writer = location_config.to_et_and_location_writer(
        identifier=identifier,
        fields=fields,
        generator=generator,
    )

    source = SourceConfig(
        location=location_config.to_location(),
        name=name,
        extract_transform=extract_transform,
        identifier=identifier,
        fields=fields,
    )

    return SourceTestkit(
        config=source,
        features=features,
        data=data,
        data_hashes=data_hashes,
        entities=tuple(sorted(results_entities)),
        location_writer=location_writer,
    )


def source_from_tuple(
    data_tuple: tuple[dict[str, Any], ...],
    data_identifiers: tuple[Any],
    name: SourceResolutionName | None = None,
    location_config: LocationConfig | None = None,
    seed: int = 42,
) -> SourceTestkit:
    """Generate a complete source testkit from dummy data.

    Args:
        data_tuple: Tuple of dictionaries representing the data rows.
        data_identifiers: Tuple of primary keys for the data rows.
        name: Optional resolution name for the source. If not provided,
            a unique name will be generated.
        location_config: Optional location configuration. If not provided,
            defaults to an in-memory SQLite database.
        seed: Random seed for reproducibility.
    """
    generator = Faker()
    generator.seed_instance(seed)

    if name is None:
        name = generator.unique.word()

    if location_config is None:
        location_config = RelationalDBConfig(
            location_options=RelationalDBConfigOptions(
                table_strategy="single",
                table_mapping=None,
            ),
            uri="sqlite:///:memory:",
        )

    base_entities = tuple(SourceEntity(base_values=row) for row in data_tuple)

    # Create source entities with references
    source_entities: list[SourceEntity] = []
    for entity, identifier in zip(base_entities, data_identifiers, strict=True):
        entity.add_source_reference(name, [identifier])
        source_entities.append(entity)
    entity_ids = {entity.id for entity in source_entities}

    # Create ClusterEntity objects from row_identifiers
    results_entities = [
        ClusterEntity(
            id=entity_id,
            source_identifiers=EntityReference({name: frozenset([identifier])}),
        )
        for identifier, entity_id in zip(data_identifiers, entity_ids, strict=True)
    ]

    # Create identifier and fields
    identifier: SourceField = SourceField(
        name="identifier",
        type=DataTypes.STRING,
    )
    fields: tuple[SourceField, ...] = tuple(
        SourceField(name=k, type=convert_data_type(type(v)))
        for k, v in data_tuple[0].items()
    )

    # Create the extract/transform string and location writer
    extract_transform, location_writer = location_config.to_et_and_location_writer(
        identifier=identifier,
        fields=fields,
        generator=generator,
    )

    source_config = SourceConfig(
        location=location_config.to_location(),
        name=name,
        extract_transform=extract_transform,
        identifier=identifier,
        fields=fields,
    )

    hashes = [hash_values(*row) for row in data_tuple]

    data_hashes = pa.Table.from_pydict(
        {
            # Assumes that string conversion will be the same as the SQL warehouse's
            "source_identifier": [str(dpk) for dpk in data_identifiers],
            "hash": hashes,
        },
        schema=SCHEMA_INDEX,
    )

    raw_data = pa.Table.from_pylist(list(data_tuple))
    raw_identifiers = pa.array(data_identifiers)

    data = raw_data.append_column("id", [entity_ids]).append_column(
        "identifier", raw_identifiers
    )

    return SourceTestkit(
        config=source_config,
        data=data,
        data_hashes=data_hashes,
        entities=tuple(sorted(results_entities)),
        location_writer=location_writer,
    )


@cache
def linked_sources_factory(
    source_testkit_configs: tuple[SourceTestkitConfig, ...] | None = None,
    n_true_entities: int | None = None,
    location_config: LocationConfig | None = None,
    seed: int = 42,
) -> LinkedSourcesTestkit:
    """Generate a set of linked sources with tracked entities.

    Args:
        source_testkit_configs: Optional tuple of source configurations
        n_true_entities: Optional number of true entities to generate. If provided,
            overrides any n_true_entities in source configs. If not provided, each
            SourceTestkitConfig must specify its own n_true_entities.
        location_config: Optional location to use for all sources. If provided,
            overrides any location in source configs.
        seed: Random seed for reproducibility
    """
    generator = Faker()
    generator.seed_instance(seed)

    default_location = RelationalDBConfig(
        location_options={"table_strategy": "single"},
        uri="sqlite:///:memory:",
    )

    if source_testkit_configs is None:
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

        source_testkit_configs = (
            SourceTestkitConfig(
                name="crn",
                location_config=location_config or default_location,
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
            SourceTestkitConfig(
                name="duns",
                location_config=location_config or default_location,
                features=(
                    features["company_name"],
                    features["duns"],
                ),
                n_true_entities=n_true_entities // 2,
                repetition=0,
            ),
            SourceTestkitConfig(
                name="cdms",
                location_config=location_config or default_location,
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
            config_entities = [
                config.n_true_entities for config in source_testkit_configs
            ]
            if any(n is not None for n in config_entities):
                warnings.warn(
                    "Both source configs and linked_sources_factory specify "
                    "n_true_entities. The factory parameter will be used.",
                    UserWarning,
                    stacklevel=2,
                )
            # Override all configs with factory parameter
            source_testkit_configs = tuple(
                SourceTestkitConfig(
                    name=config.name,
                    location_config=location_config or default_location,
                    features=config.features,
                    repetition=config.repetition,
                    n_true_entities=n_true_entities,
                )
                for config in source_testkit_configs
            )
        else:
            # No factory parameter - check all configs have n_true_entities set
            missing_counts = [
                config.name
                for config in source_testkit_configs
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
    for config in source_testkit_configs:
        all_features.update(config.features)
    all_features = tuple(sorted(all_features, key=lambda f: f.name))

    # Find maximum number of entities needed across all sources
    max_entities = max(config.n_true_entities for config in source_testkit_configs)

    # Generate all possible entities
    all_entities = generate_entities(
        generator=generator, features=all_features, n=max_entities
    )

    # Initialize LinkedSourcesTestkit
    true_entity_lookup = {entity.id: entity for entity in all_entities}
    linked = LinkedSourcesTestkit(
        true_entities=all_entities,
        sources={},
    )

    # Generate sources
    for config in source_testkit_configs:
        # Generate source data using seed entities
        data, data_hashes, entity_identifiers, row_identifiers = generate_source(
            generator=generator,
            features=tuple(config.features),
            n_true_entities=config.n_true_entities,
            repetition=config.repetition,
            seed_entities=all_entities,
        )

        # Create ClusterEntity objects from row_identifiers
        results_entities = [
            ClusterEntity(
                id=row_id,
                source_identifiers=EntityReference(
                    {config.name: frozenset(identifiers)}
                ),
            )
            for row_id, identifiers in row_identifiers.items()
        ]

        # Create identifier and fields
        identifier: SourceField = SourceField(
            name="identifier",
            type=DataTypes.STRING,
        )
        fields: tuple[SourceField, ...] = tuple(
            SourceField(name=f.name, type=f.datatype) for f in config.features
        )

        # Create the extract/transform string and location writer
        extract_transform, location_writer = (
            config.location_config.to_et_and_location_writer(
                identifier=identifier,
                fields=fields,
                generator=generator,
            )
        )

        # Create source
        source_config = SourceConfig(
            location=config.location_config.to_location(),
            name=config.name,
            extract_transform=extract_transform,
            identifier=identifier,
            fields=fields,
        )

        # Add source to linked.sources
        linked.sources[config.name] = SourceTestkit(
            config=source_config,
            features=tuple(config.features),
            data=data,
            data_hashes=data_hashes,
            entities=tuple(sorted(results_entities)),
            location_writer=location_writer,
        )

        # Update entities with source references
        for entity_id, identifiers in entity_identifiers.items():
            entity = true_entity_lookup[entity_id]
            entity.add_source_reference(config.name, identifiers)

    return linked
