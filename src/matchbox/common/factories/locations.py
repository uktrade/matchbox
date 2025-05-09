"""Factories for generating locations for testing."""

import textwrap
from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, Literal, Union

import pandas as pd
import pyarrow as pa
from faker import Faker
from frozendict import frozendict
from pydantic import AnyUrl, BaseModel, BeforeValidator, ConfigDict, Field
from sqlalchemy.engine import Engine

from matchbox.common.factories.entities import (
    FeatureConfig,
)
from matchbox.common.sources import (
    LocationType,
    LocationTypeStr,
    RelationalDBLocation,
    SourceField,
)

LocationConfigType = Union["RelationalDBConfig"]


class LocationConfig(ABC, BaseModel):
    """Configuration for a location type."""

    model_config = ConfigDict(frozen=True)

    location_type: LocationTypeStr
    location_options: LocationConfigType
    uri: AnyUrl | None = Field(default=None)

    @abstractmethod
    def to_location(self) -> LocationType:
        """Convert the configuration to a location type."""
        ...

    @abstractmethod
    def to_et_and_location_writer(
        self,
        fields: list[FeatureConfig],
        generator: Faker,
    ) -> tuple[str, Callable[[pa.Table, LocationType, Any], None]]:
        """Convert the configuration to an e/t string and location writer.

        Args:
            fields: List of FeatureConfig objects to use for the location writer.
            generator: Faker instance to use for generating metadata.

        Returns:
            - The extract transform string to retrieve the data from the location
            - A function that takes SourceTestkit.data and
                SourceTestkit.config.location and writes the data to the location.
                The function should take the following arguments:

                    - data: The data to write to the location.
                    - location: The location to write the data to.
                    - credentials: The credentials to use for the location.
        """
        ...


RelationalDBConfigTableMapping = Annotated[
    frozendict[str, tuple[FeatureConfig]],
    BeforeValidator(lambda v: frozendict(v) if isinstance(v, dict) else None),
]


class RelationalDBConfigOptions(BaseModel):
    """Configuration options for a relational database location."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    table_strategy: Literal["single", "spread"] = Field(
        default="single",
        description="A high-level strategy for how to split the data into tables.",
    )
    table_mapping: RelationalDBConfigTableMapping | None = Field(
        default=None,
        description=(
            "A low-level mapping of features to tables. "
            "If supplied, table_strategy will be ignored.",
        ),
    )


class RelationalDBConfig(LocationConfig):
    """Configuration for a relational database location."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    location_type: Literal["rdbms"] = Field(default="rdbms")
    location_options: RelationalDBConfigOptions = Field(
        default=RelationalDBConfigOptions()
    )
    uri: AnyUrl | None = Field(default="sqlite:///:memory:")

    def to_location(self) -> RelationalDBLocation:  # noqa: D102
        return RelationalDBLocation(uri=self.uri)

    def to_et_and_location_writer(  # noqa: D102
        self,
        identifier: SourceField,
        fields: list[SourceField],
        generator: Faker,
    ) -> tuple[str, Callable[[pa.Table, RelationalDBLocation, Engine], None]]:
        _to_location: Callable[[pa.Table, RelationalDBLocation, Engine], None]
        sql: str

        # Validate that identifier name is "pk" for now
        if identifier.name != "pk":
            raise ValueError(
                f"Identifier name must be 'pk' for now, but got '{identifier.name}'"
            )

        def _make_select_clause(field_names: list[str], table_map: dict[str, str]):
            """Create SELECT clause with proper table prefixes."""
            return ",\n    ".join(
                f'"{table_map[name]}"."{name}"' for name in field_names
            )

        def _write_table(
            df: pd.DataFrame,
            table_name: str,
            field_names: list[str],
            location: RelationalDBLocation,
            credentials: Engine,
        ):
            """Write a subset of columns to a database table."""
            if str(credentials.url) != str(location.uri):
                raise ValueError(
                    "The credentials provided do not match the location URI."
                )

            if not field_names:
                # Spread strategy 6NF central table
                df[[identifier.name]].to_sql(
                    name=table_name,
                    con=credentials,
                    index=False,
                    if_exists="replace",
                )
            else:
                # Write pk plus specified fields
                columns = [identifier.name] + field_names
                df[columns].to_sql(
                    name=table_name,
                    con=credentials,
                    index=False,
                    if_exists="replace",
                )

        # This handles three strategies:
        # 1. Custom table mapping (table_mapping is provided)
        # 2. Spread strategy (each field in its own table)
        # 3. Single table strategy (default)

        tables_to_fields: dict[str, list[str]] = {}
        field_to_table: dict[str, str] = {}

        # 1. Custom table mapping
        if self.location_options.table_mapping:
            # Use the explicit mapping provided
            for table, table_fields in self.location_options.table_mapping.items():
                tables_to_fields[table] = [f.name for f in table_fields]
                for field in table_fields:
                    field_to_table[field.name] = table

            # Validate the mapping is complete
            mapped_fields: list[FeatureConfig] = [
                field
                for fields in self.location_options.table_mapping.values()
                for field in fields
            ]

            if len(mapped_fields) != len(fields):
                raise ValueError(
                    "Table mapping does not match the number of features. "
                    "Perhaps the same feature is in multiple tables? "
                    "Please ensure they are set correctly."
                )

            if set(mapped_fields) != set(fields):
                fields_set = set(fields)
                mapped_set = set(mapped_fields)

                raise ValueError(
                    "Table mapping does not match the features. "
                    f"Missing fields: {fields_set - mapped_set}. "
                    f"Extra fields: {mapped_set - fields_set}. "
                    "Please ensure they are set correctly."
                )

        # 2. Spread strategy
        elif self.location_options.table_strategy == "spread":
            # Create a table for each field
            main_table = generator.unique.word()
            tables_to_fields[main_table] = []  # Main table just holds PKs, 6NF

            for field in fields:
                table_name = f"{main_table}_{field.name}"
                tables_to_fields[table_name] = [field.name]
                field_to_table[field.name] = table_name

        # 3. Single table strategy
        else:
            # Put all fields in one table
            table_name = generator.unique.word()
            tables_to_fields[table_name] = [field.name for field in fields]
            field_to_table.update({field.name: table_name for field in fields})

        # The main table is always the first one
        tables: list[str] = list(tables_to_fields.keys())
        main_table: str = tables[0]

        # Add identifier to the field_to_table mapping, using the main table
        field_to_table[identifier.name] = main_table

        # Define the location writer function
        def _to_location(
            data: pa.Table, location: RelationalDBLocation, credentials: Engine
        ) -> None:
            """Write the data to tables in the database according to the strategy."""
            df = data.to_pandas()

            for table, table_fields in tables_to_fields.items():
                _write_table(df, table, table_fields, location, credentials)

        # Build SQL query with joins
        # Include identifier in the field names for SELECT clause
        all_field_names = [identifier.name] + [field.name for field in fields]
        select_string = _make_select_clause(all_field_names, field_to_table)

        # Build JOIN clauses for all tables except the main one
        join_clauses = [
            textwrap.dedent(f"""
                LEFT JOIN "{table}" ON
                    "{main_table}"."{identifier.name}" = "{table}"."{identifier.name}"
            """)
            for table in tables[1:]
        ]
        join_string = "\n    ".join(join_clauses)

        sql = textwrap.dedent(
            f"""
            SELECT
                {select_string}
            FROM 
                "{main_table}"
            {join_string if join_clauses else ""};
            """
        )

        return sql, _to_location


def location_factory(
    location_type: LocationTypeStr = "rdbms",
    location_options: dict[str, Any] | None = None,
    uri: AnyUrl | None = None,
) -> LocationConfig:
    """Generate a location configuration for a source.

    Provides default options, but also ensures all options agree with each other.

    Args:
        location_type: Option type of location to create. Currently only "rdbms" is
            supported.
        location_options: Optional options for the location type. If not provided,
            defaults to a single table strategy for a relational database.
        uri: Optional URI for the source. If not provided, defaults to an in-memory
            SQLite database.

    Returns:
        A LocationConfig object with the specified options.
    """
    if location_type == "rdbms":
        if location_options is None:
            location_options = RelationalDBConfigOptions(
                table_strategy="single",
                table_mapping=None,
            )

        location_options = RelationalDBConfigOptions.model_validate(location_options)

        if uri is None:
            uri = "sqlite:///:memory:"

        return RelationalDBConfig(
            location_type=location_type,
            location_options=location_options,
            uri=uri,
        )

    raise ValueError(f"Unsupported location type: {location_type}")
