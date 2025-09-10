"""Interface to locations where source data is stored."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import (
    Any,
    ParamSpec,
    Self,
    TypeVar,
    overload,
)

import polars as pl
import sqlglot
from pyarrow import Table as ArrowTable
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from matchbox.common.db import (
    QueryReturnClass,
    QueryReturnType,
    sql_to_df,
)
from matchbox.common.dtos import (
    DataTypes,
    LocationConfig,
    LocationType,
    Resolution,
    SourceConfig,
    SourceField,
)
from matchbox.common.exceptions import (
    MatchboxSourceClientError,
    MatchboxSourceExtractTransformError,
)
from matchbox.common.graph import ResolutionType
from matchbox.common.hash import HashMethod, hash_rows
from matchbox.common.logging import logger

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F")


def requires_client(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator that checks if client is set before executing a method.

    A helper method for Location subclasses.

    Raises:
        MatchboxSourceClientError: If the client is not set.
    """

    @wraps(method)
    def wrapper(self: "Location", *args, **kwargs) -> T:
        if self.client is None:
            raise MatchboxSourceClientError
        return method(self, *args, **kwargs)

    return wrapper


class Location(ABC):
    """A location for a data source."""

    def __init__(self, name: str, client: Any):
        """Initialise location."""
        self.config = LocationConfig(type=self.location_type, name=name)
        self.client = client

    def __deepcopy__(self, memo=None):
        """Create a deep copy of the Location object."""
        if memo is None:
            memo = {}

        # Both objects should share the same client
        obj_copy = type(self)(name=deepcopy(self.config.name, memo), client=self.client)

        return obj_copy

    @property
    @abstractmethod
    def location_type(self) -> LocationType:
        """Output location type string."""
        ...

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data location.

        Raises:
            AttributeError: If the client is not set.
        """
        ...

    @abstractmethod
    def validate_extract_transform(self, extract_transform: str) -> bool:
        """Validate ET logic against this location's query language.

        Raises:
            MatchboxSourceExtractTransformError: If the ET logic is invalid.
        """
        ...

    @abstractmethod
    def infer_types(self, extract_transform: str) -> dict[str, DataTypes]:
        """Extract all data types from the ET logic."""
        ...

    @abstractmethod
    def execute(
        self,
        extract_transform: str,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        keys: tuple[str, list[str]] | None = None,
    ) -> Iterator[QueryReturnClass]:
        """Execute ET logic against location and return batches.

        Args:
            extract_transform: The ET logic to execute.
            batch_size: The size of the batches to return.
            rename: Renaming to apply after the ET logic is executed.

                * If a dictionary is provided, it will be used to rename the columns.
                * If a callable is provided, it will take the old name as input and
                    return the new name.
            return_type: The type of data to return. Defaults to "polars".
            keys: Rule to only retrieve rows by specific keys.
                The key of the dictionary is a field name on which to filter.
                Filters source entries where the key field is in the dict values.

        Raises:
            AttributeError: If the cliet is not set.
        """
        ...

    def from_config(config: LocationConfig, client: Any) -> Self:
        """Initialise location from a location config and an appropriate client."""
        LocClass = location_type_to_class(config.type)
        return LocClass(name=config.name, client=client)


class RelationalDBLocation(Location):
    """A location for a relational database."""

    client: Engine
    location_type: LocationType = LocationType.RDBMS

    @contextmanager
    def _get_connection(self):
        """Context manager for getting database connections with proper cleanup."""
        connection = self.client.connect()
        try:
            yield connection
        finally:
            connection.close()

    @requires_client
    def connect(self) -> bool:  # noqa: D102
        try:
            with self._get_connection() as conn:
                _ = sql_to_df(stmt="select 1", connection=conn)
                return True
        except OperationalError:
            return False

    def validate_extract_transform(self, extract_transform: str) -> bool:  # noqa: D102
        """Check that the SQL statement only contains a single data-extracting command.

        We are NOT attempting a full sanitisation of the SQL statement
        # Validation is done purely to stop accidental mistakes, not malicious actors
        # Users should only run indexing using SourceConfigs they trust and have read,
        # using least privilege credentials

        Args:
            extract_transform: The SQL statement to validate

        Returns:
            bool: True if the SQL statement is valid

        Raises:
            ParseError: If the SQL statement cannot be parsed
            MatchboxSourceExtractTransformError: If validation requirements are not met
        """
        if not extract_transform.strip():
            raise MatchboxSourceExtractTransformError(
                "SQL statement is empty or only contains whitespace."
            )

        expressions = sqlglot.parse(extract_transform)

        if len(expressions) > 1:
            raise MatchboxSourceExtractTransformError(
                "SQL statement contains multiple commands."
            )

        if not expressions:
            raise MatchboxSourceExtractTransformError(
                "SQL statement does not contain any valid expressions."
            )

        expr = expressions[0]

        if not isinstance(expr, sqlglot.expressions.Select):
            raise MatchboxSourceExtractTransformError(
                "SQL statement must start with a SELECT or WITH command."
            )

        forbidden = (
            sqlglot.expressions.DDL,
            sqlglot.expressions.DML,
            sqlglot.expressions.Into,
        )

        if len(list(expr.find_all(forbidden))) > 0:
            raise MatchboxSourceExtractTransformError(
                "SQL statement must not contain DDL or DML commands."
            )

        return True

    @requires_client
    def infer_types(self, extract_transform: str) -> dict[str, DataTypes]:  # noqa: D102
        extract_transform = extract_transform.rstrip(" \t\n;")
        one_row_query = f"select * from ({extract_transform}) as sub limit 1;"
        one_row: pl.DataFrame = list(self.execute(one_row_query))[0]
        column_names = one_row.columns

        inferred_types = {}
        for col in column_names:
            # This expression uses cross-dialect SQL standards;
            # though this is hard to prove as the standard is behind a paywall
            sample_query = (
                f"select {col} from ({extract_transform}) as sub "
                f"where {col} is not null limit 1;"
            )

            sample_row: pl.DataFrame = list(self.execute(sample_query))[0]

            if len(sample_row):
                sample_dtype = sample_row[col].dtype
                inferred_types[col] = DataTypes.from_dtype(sample_dtype)
            else:
                inferred_types[col] = DataTypes.NULL

        return inferred_types

    @requires_client
    def execute(  # noqa: D102
        self,
        extract_transform: str,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        keys: tuple[str, list[str]] | None = None,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[QueryReturnClass, None, None]:
        batch_size = batch_size or 10_000
        with self._get_connection() as conn:
            if keys:
                key_field, filter_values = keys
                quoted_key_values = [f"'{key_val}'" for key_val in filter_values]
                # Only filter original SQL if keys are provided
                if quoted_key_values:
                    comma_separated_values = ", ".join(quoted_key_values)
                    # Deal with end-of-statement semi-colons that could be supplied
                    extract_transform = extract_transform.replace(";", "")
                    # This "IN" expression is a SQL standard;
                    # though this is hard to prove as the standard is behind a paywall
                    extract_transform = (
                        f"select * from ({extract_transform}) as sub "
                        f"where {key_field} in ({comma_separated_values})"
                    )
            yield from sql_to_df(
                stmt=extract_transform,
                schema_overrides=schema_overrides,
                connection=conn,
                rename=rename,
                batch_size=batch_size,
                return_batches=True,
                return_type=return_type,
            )


def location_type_to_class(location_type: LocationType) -> type[Location]:
    """Map location type string to the corresponding class."""
    match location_type:
        case LocationType.RDBMS:
            return RelationalDBLocation
        case _:
            raise ValueError("Location type not recognised.")


class Source:
    """Client-side wrapper for source configs."""

    @overload
    def __init__(
        self,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: str,
        index_fields: list[str],
        description: str | None = None,
        infer_types: bool = True,
    ) -> None: ...

    @overload
    def __init__(
        self,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: SourceField,
        index_fields: list[SourceField],
        description: str | None = None,
        infer_types: bool = False,
    ) -> None: ...

    def __init__(
        self,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: str | SourceField,
        index_fields: list[str] | list[SourceField],
        description: str | None = None,
        infer_types: bool = False,
    ):
        """Initialise source.

        Args:
            location: The location where the source data is stored.
            name: The name of the source.
            description: An optional description of the source.
            extract_transform: The extract/transform logic to apply to the source data.
            key_field: The name of the field to use as the key, or a SourceField
                instance defining the key field. This is the unique identifier we'll
                use to refer to matched data in the source.
            index_fields: The names of the fields to use as index fields, or a list
                of SourceField instances defining the index fields. These are the
                fields you plan to match on.
            infer_types: Whether to infer data types for the fields from the source.
                If False, you must provide SourceField instances for key_field and
                index_fields.
        """
        if not location.validate_extract_transform(extract_transform):
            raise MatchboxSourceExtractTransformError

        self.location = location
        self.name = name
        self.description = description

        if infer_types:
            self._validate_fields(key_field, index_fields, str)

            # Assumes client has been set on location
            inferred_types = location.infer_types(extract_transform)
            remote_fields = {
                field_name: SourceField(name=field_name, type=dtype)
                for field_name, dtype in inferred_types.items()
            }
            typed_key_field = SourceField(name=key_field, type=DataTypes.STRING)
            typed_index_fields = tuple(remote_fields[field] for field in index_fields)
        else:
            typed_key_field, typed_index_fields = self._validate_fields(
                key_field, index_fields, SourceField
            )

        self.config = SourceConfig(
            location_config=location.config,
            extract_transform=extract_transform,
            key_field=typed_key_field,
            index_fields=typed_index_fields,
        )

    def _validate_fields(
        self,
        key_field: Any,
        index_fields: list[Any],
        type_check: type[str] | type[SourceField],
    ) -> tuple[F, tuple[F, ...]]:
        """Validate that fields match the expected type (str or SourceField)."""
        if not isinstance(key_field, type_check):
            raise ValueError(
                f"Expected {type_check.__name__}, got {type(key_field).__name__}"
            )

        if not all(isinstance(f, type_check) for f in index_fields):
            raise ValueError(
                f"All index_fields must be {type_check.__name__} instances"
            )

        return key_field, tuple(index_fields)

    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        return Resolution(
            name=self.name,
            description=self.description,
            truth=None,
            resolution_type=ResolutionType.SOURCE,
            config=self.config,
        )

    @classmethod
    def from_resolution(cls, resolution: Resolution, location: Location) -> "Source":
        """Reconstruct from Resolution."""
        assert resolution.resolution_type == ResolutionType.SOURCE, (
            "Resolution must be of type 'source'"
        )
        assert isinstance(resolution.config, SourceConfig), (
            "Config must be SourceConfig"
        )

        # Create a new Source instance
        source = cls.__new__(cls)  # Create without calling __init__
        source.location = location
        source.name = resolution.name
        source.description = resolution.description
        source.config = resolution.config

        return source

    def __hash__(self) -> int:
        """Return a hash of the Source based on its config."""
        return hash(self.config)

    def __eq__(self, other: Any) -> bool:
        """Check equality of two Source objects based on their config."""
        if not isinstance(other, Source):
            return False
        return self.config == other.config

    def query(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        keys: list[str] | None = None,
    ) -> Generator[QueryReturnClass, None, None]:
        """Applies the extract/transform logic to the source and returns the results.

        Args:
            qualify_names: If True, qualify the names of the columns with the
                source name.
            batch_size: Indicate the size of each batch when fetching data in batches.
            return_type: The type of data to return. Defaults to "polars".
            keys: List of keys to select a subset of all source entries.

        Returns:
            The requested data in the specified format, as an iterator of tables.
        """
        _rename: Callable | None = None
        if qualify_names:

            def _rename(c: str) -> str:
                return self.name + "_" + c

        all_fields = self.config.index_fields + tuple([self.config.key_field])
        schema_overrides = {field.name: field.type.to_dtype() for field in all_fields}

        if keys:
            yield from self.location.execute(
                extract_transform=self.config.extract_transform,
                schema_overrides=schema_overrides,
                rename=_rename,
                batch_size=batch_size,
                return_type=return_type,
                keys=(self.config.key_field.name, keys),
            )
        else:
            yield from self.location.execute(
                extract_transform=self.config.extract_transform,
                schema_overrides=schema_overrides,
                rename=_rename,
                batch_size=batch_size,
                return_type=return_type,
            )

    def hash_data(self, batch_size: int | None = None) -> ArrowTable:
        """Retrieve and hash a dataset from its warehouse, ready to be inserted.

        Hashes the index fields defined in the source based on the
        extract/transform logic.

        Does not hash the key field.

        Args:
            batch_size: If set, process data in batches internally. Indicates the
                size of each batch.

        Returns:
            A PyArrow Table containing source keys and their hashes.
        """
        log_prefix = f"Hash {self.name}"
        batch_info = (
            f"with batch size {batch_size:,}" if batch_size else "without batching"
        )
        logger.debug(f"Retrieving and hashing {batch_info}", prefix=log_prefix)

        key_field: str = self.config.key_field.name
        index_fields: list[str] = [field.name for field in self.config.index_fields]

        all_results: list[pl.DataFrame] = []
        for batch in self.query(
            batch_size=batch_size,
            return_type="polars",
        ):
            batch: pl.DataFrame

            if batch[key_field].is_null().any():
                raise ValueError("keys column contains null values")

            row_hashes: pl.Series = hash_rows(
                df=batch,
                columns=list(sorted(index_fields)),
                method=HashMethod.SHA256,
            )

            result = (
                batch.rename({key_field: "keys"})
                .with_columns(row_hashes.alias("hash"))
                .select(["hash", "keys"])
            )
            all_results.append(result)

        processed_df = pl.concat(all_results)

        return processed_df.group_by("hash").agg(pl.col("keys")).to_arrow()

    # Note: name, description, truth are now instance variables, not properties

    @property
    def prefix(self) -> str:
        """Get the prefix for the source."""
        return self.config.prefix(self.name)

    @property
    def qualified_key(self) -> str:
        """Get the qualified key for the source."""
        return self.config.qualified_key(self.name)

    @property
    def qualified_index_fields(self) -> list[str]:
        """Get the qualified index fields for the source."""
        return self.config.qualified_index_fields(self.name)

    def qualify_field(self, field: str) -> str:
        """Qualify field names with the source name.

        Args:
            field: The field name to qualify.

        Returns:
            A single qualified field.

        """
        return self.config.qualify_field(self.name, field)

    def f(self, fields: str | Iterable[str]) -> str | list[str]:
        """Qualify one or more field names with the source name.

        Args:
            fields: The field name to qualify, or a list of field names.

        Returns:
            A single qualified field, or a list of qualified field names.

        """
        return self.config.f(self.name, fields)
