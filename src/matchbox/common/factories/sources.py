from abc import ABC, abstractmethod
from functools import cache, wraps
from math import comb
from typing import Callable, ParamSpec, TypeVar
from unittest.mock import Mock, create_autospec
from uuid import uuid4

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


class FeatureConfig(BaseModel):
    """Configuration for generating a feature with variations."""

    model_config = ConfigDict(frozen=True)

    name: str
    base_generator: str
    parameters: tuple = Field(default_factory=tuple)
    variations: tuple[VariationRule, ...] = Field(default_factory=tuple)


class SourceMetrics(BaseModel):
    """Metrics about the generated data.

    The metrics represent:

    * `n_true_entities`: The number of true entities generated.
    * `n_unique_rows`: The number of unique rows generated. Can also be
        thought of as the drop count after dropping all duplicates.
    * `n_potential_pairs`: The number of potential pairs that can be compared
        for deduplication.

    For example, in the following table:

    ```markdown
    | id | company_name |
    |----|--------------|
    | 1  | alpha        |
    | 2  | alpha ltd    |
    | 3  | alpha        |
    | 4  | alpha ltd    |
    | 5  | beta         |
    | 6  | beta ltd     |
    | 7  | beta         |
    | 8  | beta ltd     |
    ```

    * `n_true_entities` = 2
    * `n_unique_rows` = 4
    * `n_potential_pairs` = 12

    The potential pairs formula multiplies (`n_unique_rows` choose 2) by
    `n_true_entities` because we need to compare each unique variation with every
    other variation for each true entity in the dataset. This accounts for the fact
    that each unique row appears `n_true_entities` times in the full dataset.
    """

    n_true_entities: int
    n_unique_rows: int
    n_potential_pairs: int

    @classmethod
    def calculate(
        cls, n_true_entities: int, max_variations_per_entity: int
    ) -> "SourceMetrics":
        """Calculate metrics based on entity count and variations."""
        n_unique_rows = 1 + max_variations_per_entity
        n_potential_pairs = comb(n_unique_rows, 2) * n_true_entities

        return cls(
            n_true_entities=n_true_entities,
            n_unique_rows=n_unique_rows,
            n_potential_pairs=n_potential_pairs,
        )


class SourceDummy(BaseModel):
    """Complete representation of a generated dummy Source."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Source
    features: tuple[FeatureConfig, ...]
    data: pa.Table
    data_hashes: pa.Table
    metrics: SourceMetrics

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


class SourceDataGenerator:
    """Generates dummy data for a Source."""

    def __init__(self, seed: int = 42):
        self.faker = Faker()
        self.faker.seed_instance(seed)

    def generate_data(
        self, n_true_entities: int, features: tuple[FeatureConfig], repetition: int
    ) -> tuple[pa.Table, pa.Table, SourceMetrics]:
        """Generate raw data as PyArrow tables.

        Returns:
            A tuple of:
            * The raw data
            * The data hashes
            * The metrics that go with this data
        """
        max_variations = max(len(f.variations) for f in features)

        raw_data = {"pk": []}
        for feature in features:
            raw_data[feature.name] = []

        # Generate data entity by entity
        for _ in range(n_true_entities):
            # Generate base values -- the raw row
            base_values = {
                f.name: getattr(self.faker, f.base_generator)(**dict(f.parameters))
                for f in features
            }

            raw_data["pk"].append(str(uuid4()))
            for name, value in base_values.items():
                raw_data[name].append(value)

            # Add variations
            for variation_idx in range(max_variations):
                raw_data["pk"].append(str(uuid4()))
                for feature in features:
                    if variation_idx < len(feature.variations):
                        # Apply variation
                        value = feature.variations[variation_idx].apply(
                            base_values[feature.name]
                        )
                    else:
                        # Use base value for padding
                        value = base_values[feature.name]
                    raw_data[feature.name].append(value)

        # Create DataFrame and apply repetition
        df = pd.DataFrame(raw_data)
        df = pd.concat([df] * repetition, ignore_index=True)

        # Group by all features except pk to get hash groups
        feature_names = [f.name for f in features]
        hash_groups = (
            df.groupby(feature_names, sort=False)["pk"].agg(list).reset_index()
        )

        # Create data_hashes table
        hash_groups["hash"] = [str(uuid4()).encode() for _ in range(len(hash_groups))]
        data_hashes = pa.Table.from_pydict(
            {
                "source_pk": hash_groups["pk"].tolist(),
                "hash": hash_groups["hash"].tolist(),
            },
            schema=SCHEMA_INDEX,
        )

        metrics = SourceMetrics.calculate(
            n_true_entities=n_true_entities, max_variations_per_entity=max_variations
        )

        return pa.Table.from_pandas(df), data_hashes, metrics


@make_features_hashable
@cache
def source_factory(
    features: list[FeatureConfig] | list[dict] | None = None,
    full_name: str | None = None,
    engine: Engine | None = None,
    n_true_entities: int = 10,
    repetition: int = 1,
    seed: int = 42,
) -> SourceDummy:
    """Generate a complete dummy source.

    Args:
        features: List of FeatureConfigs, used to generate features with variations
        full_name: Full name of the source, like "dbt.companies_house".
        engine: SQLAlchemy engine to use for the source.
        n_true_entities: Number of true entities to generate.
        repetition: Number of times to repeat the data.
        seed: Random seed for data generation.

    Returns:
        SourceDummy: Complete dummy source with generated data.
    """
    generator = SourceDataGenerator(seed)

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
        full_name = generator.faker.word()

    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    data, data_hashes, metrics = generator.generate_data(
        n_true_entities=n_true_entities, features=features, repetition=repetition
    )

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
        metrics=metrics,
    )
