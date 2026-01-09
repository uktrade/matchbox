"""Interface to locations where source data is stored."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matchbox.common.datatypes import DataTypes

if TYPE_CHECKING:
    from adbc_driver_manager.dbapi import Connection as AdbcConnection

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from copy import deepcopy
from enum import StrEnum
from functools import wraps
from typing import Any, ParamSpec, Self, TypeVar

import polars as pl
import sqlglot
from sqlalchemy import Connection, Engine
from sqlalchemy.exc import OperationalError
from sqlglot.errors import ParseError

from matchbox.common.db import (
    QueryReturnClass,
    QueryReturnType,
    sql_to_df,
)
from matchbox.common.dtos import LocationConfig, LocationType
from matchbox.common.exceptions import (
    MatchboxSourceClientError,
    MatchboxSourceExtractTransformError,
)
from matchbox.common.logging import logger

T = TypeVar("T")
P = ParamSpec("P")


class ClientType(StrEnum):
    """Enumeration of valid location clients."""

    SQLALCHEMY = "sqlalchemy"
    ADBC = "adbc"


CLIENT_CLASSES: dict[ClientType, type | tuple[type, ...]] = {
    ClientType.SQLALCHEMY: Engine
}

try:
    from adbc_driver_manager.dbapi import Connection as AdbcConnection

    CLIENT_CLASSES[ClientType.ADBC] = AdbcConnection
except ImportError:
    pass


def requires_client(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator that checks if client is set before executing a method.

    A helper method for Location subclasses.

    Raises:
        MatchboxSourceClientError: If the client is not set.
    """

    @wraps(method)
    def wrapper(self: Location, *args: Any, **kwargs: Any) -> T:
        if self.client is None:
            raise MatchboxSourceClientError
        return method(self, *args, **kwargs)

    return wrapper


class Location(ABC):
    """A location for a data source."""

    def __init__(self, name: str) -> None:
        """Initialise location."""
        self.config = LocationConfig(type=self.location_type, name=name)
        self._client = None

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Create a deep copy of the Location object."""
        obj_copy = type(self)(name=deepcopy(self.config.name, memo))

        # Both objects should share the same client
        if self.client:
            obj_copy.set_client(self.client)

        return obj_copy

    @property
    def client(self) -> Engine | None:
        """Retrieve client."""
        return self._client

    def set_client(self, client: Any) -> Self:  # noqa: ANN401
        """Set client for location and return the location."""
        # Validate against all possible client types
        if not isinstance(client, tuple(CLIENT_CLASSES.values())):
            raise ValueError(
                f"Type {client.__class__} is not valid for {self.__class__}"
            )
        self._client = client
        return self

    @property
    @abstractmethod
    def location_type(self) -> LocationType:
        """Output location type string."""
        ...

    @property
    @abstractmethod
    def client_type(self) -> ClientType:
        """Client type string."""
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
            batch_size: The size used for internal batching. Overrides environment
                variable if set.
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

    @classmethod
    def from_config(cls, config: LocationConfig) -> Self:
        """Initialise location from a location config."""
        LocClass = location_type_to_class(config.type)
        return LocClass(name=config.name)


class RelationalDBLocation(Location):
    """A location for a relational database."""

    client: Engine | AdbcConnection | None
    location_type: LocationType = LocationType.RDBMS

    @property
    def client_type(self) -> ClientType | None:
        """Determine client type from the client."""
        if isinstance(self.client, Engine):
            return ClientType.SQLALCHEMY
        elif isinstance(self.client, AdbcConnection):
            return ClientType.ADBC
        return None

    @contextmanager
    def _get_connection(self) -> Generator[Connection | AdbcConnection, None, None]:
        """Context manager for getting database connections with proper cleanup."""
        if self.client_type == ClientType.ADBC:
            # To prevent memory corruption in the ADBC driver when reusing a single
            # connection for multiple queries, we clone the connection
            with self.client.adbc_clone() as connection:
                yield connection
        else:  # SQLAlchemy Engine
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

    def validate_extract_transform(self, extract_transform: str) -> None:  # noqa: D102
        """Check that the SQL statement only contains a single data-extracting command.

        We are NOT attempting a full sanitisation of the SQL statement
        # Validation is done purely to stop accidental mistakes, not malicious actors
        # Users should only run indexing using SourceConfigs they trust and have read,
        # using least privilege credentials

        Args:
            extract_transform: The SQL statement to validate

        Raises:
            ParseError: If the SQL statement cannot be parsed
            MatchboxSourceExtractTransformError: If validation requirements are not met
        """
        if not extract_transform.strip():
            raise MatchboxSourceExtractTransformError(
                "SQL statement is empty or only contains whitespace."
            )

        # Attempt to make the SQLGlot validation more targeted
        sqlglot_dialect: str | None = None
        client_dialect: str | None = None

        if self.client and self.client_type == ClientType.SQLALCHEMY:
            client_dialect = self.client.dialect.name
        elif self.client and self.client_type == ClientType.ADBC:
            client_dialect = self.client.adbc_get_info().get("vendor_name")
            client_dialect = client_dialect.lower() if client_dialect else None

        match client_dialect:
            case "postgresql":
                sqlglot_dialect = "postgres"
            case "sqlite":
                sqlglot_dialect = "sqlite"
            case _:
                logger.warning("Could not validate specific dialect.")

        try:
            expressions = sqlglot.parse(extract_transform, dialect=sqlglot_dialect)
        except ParseError as e:
            raise MatchboxSourceExtractTransformError(
                "SQL statement could not be parsed."
            ) from e

        if len(expressions) > 1:
            raise MatchboxSourceExtractTransformError(
                "SQL statement contains multiple commands."
            )

        if not expressions:
            raise MatchboxSourceExtractTransformError(
                "SQL statement does not contain any valid expressions."
            )

        expr = expressions[0]

        if not isinstance(expr, sqlglot.expressions.Select | sqlglot.expressions.Union):
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

    @requires_client
    def infer_types(self, extract_transform: str) -> dict[str, DataTypes]:  # noqa: D102
        extract_transform = extract_transform.rstrip(" \t\n;")
        one_row_query = f"select * from ({extract_transform}) as sub limit 1;"
        one_row: pl.DataFrame = next(self.execute(one_row_query))
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
        # Strip semicolon, as it can block extended query protocol
        # and slow some engines down
        extract_transform = extract_transform.replace(";", "")

        # We always work in batches to simplify higher level logic
        batch_size: int = batch_size or 10_000

        with self._get_connection() as conn:
            if keys:
                key_field, filter_values = keys
                quoted_key_values = [f"'{key_val}'" for key_val in filter_values]
                # Only filter original SQL if keys are provided
                if quoted_key_values:
                    comma_separated_values = ", ".join(quoted_key_values)
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
