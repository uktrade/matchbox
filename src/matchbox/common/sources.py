"""Classes and functions for working with data sources in Matchbox."""

import re
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import wraps
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
    Union,
)

import polars as pl
from pyarrow import Table as ArrowTable
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from matchbox.common.db import (
    QueryReturnType,
    ReturnTypeStr,
    clean_uri,
    sql_to_df,
    validate_sql_for_data_extraction,
)
from matchbox.common.dtos import DataTypes, SourceResolutionName
from matchbox.common.exceptions import MatchboxSourceCredentialsError
from matchbox.common.hash import HashMethod, hash_rows
from matchbox.common.logging import logger

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


LocationType = Union["RelationalDBLocation"]
"""Type for Location class. Currently only supports RelationalDBLocation."""

LocationTypeStr = Union[Literal["rdbms"]]
"""String literal type for Location class. Currently only supports "rdbms"."""


def requires_credentials(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator that checks if credentials are set before executing a method.

    A helper method for Location subclasses.

    Raises:
        MatchboxSourceCredentialsError: If the credentials are not set.
    """

    @wraps(method)
    def wrapper(self: "Location", *args, **kwargs) -> T:
        if self.credentials is None:
            raise MatchboxSourceCredentialsError
        return method(self, *args, **kwargs)

    return wrapper


class Location(ABC, BaseModel):
    """A location for a data source."""

    type: LocationTypeStr
    uri: AnyUrl
    credentials: Any | None = Field(exclude=True, default=None)

    def __eq__(self, other: Any) -> bool:
        """Custom equality which ignores credentials."""
        return (self.type, self.uri) == (
            other.type,
            other.uri,
        )

    def __hash__(self) -> int:
        """Custom hash which ignores credentials."""
        return hash((self.type, self.uri))

    def __deepcopy__(self, memo=None):
        """Create a deep copy of the Location object."""
        if memo is None:
            memo = {}

        obj_copy = type(self)(
            type=deepcopy(self.type, memo),
            uri=deepcopy(self.uri, memo),
        )

        # Both objects should share the same credentials
        if self.credentials is not None:
            obj_copy.credentials = self.credentials

        return obj_copy

    @abstractmethod
    def add_credentials(self, credentials: Any) -> None:
        """Adds credentials to the location."""
        ...

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data location.

        Raises:
            AttributeError: If the credentials are not set.
        """
        ...

    @abstractmethod
    def validate_extract_transform(self, extract_transform: str) -> bool:
        """Validate SQL ET logic against this location's query language.

        Raises:
            MatchboxSourceExtractTransformError: If the ET logic is invalid.
        """
        ...

    @abstractmethod
    def head(self, extract_transform: str) -> list:
        """Extract lightweight data sample using ET logic."""
        ...

    @abstractmethod
    def execute(
        self,
        extract_transform: str,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: ReturnTypeStr = "polars",
        keys: tuple[str, list[str]] | None = None,
    ) -> Iterator[QueryReturnType]:
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
            AttributeError: If the credentials are not set.
        """
        ...


class RelationalDBLocation(Location):
    """A location for a relational database."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["rdbms"] = "rdbms"
    uri: AnyUrl
    credentials: Engine | None = Field(
        exclude=True,
        default=None,
        description=(
            "The credentials for a relational database are a SQLAlchemy Engine."
        ),
    )

    @field_validator("uri", mode="after")
    @classmethod
    def validate_uri(cls, value: AnyUrl) -> AnyUrl:
        """Ensure no credentials, query params, or fragments are in the URI."""
        return clean_uri(value)

    def _validate_engine(self, credentials: Engine) -> None:
        """Validate an engine matches the URI.

        Raises:
            ValueError: If the Engine and URI do not match.
        """
        uri = clean_uri(str(credentials.url))

        if any(
            [
                uri.scheme != self.uri.scheme,
                uri.host != self.uri.host,
                uri.port != self.uri.port,
                uri.path != self.uri.path,
            ]
        ):
            raise ValueError(
                "The Engine location URI does not match the location model URI. \n"
                f"Scheme: {uri.scheme}, {self.uri.scheme} \n"
                f"Host: {uri.host}, {self.uri.host} \n"
                f"Port: {uri.port}, {self.uri.port} \n"
                f"Path: {uri.path}, {self.uri.path} \n"
            )

    def add_credentials(self, credentials: Engine) -> None:  # noqa: D102
        self._validate_engine(credentials)
        self.credentials = credentials

    @requires_credentials
    def connect(self) -> bool:  # noqa: D102
        try:
            _ = sql_to_df(stmt="select 1", connection=self.credentials)
            return True
        except OperationalError:
            return False

    def validate_extract_transform(self, extract_transform: str) -> bool:  # noqa: D102
        # We are NOT attempting a full sanitisation of the SQL statement
        # Validation is done purely to stop accidental mistakes, not malicious actors
        # Users should only run indexing using SourceConfigs they trust and have read,
        # using least privilege credentials
        return validate_sql_for_data_extraction(extract_transform)

    def head(self, extract_transform: str) -> pl.DataFrame:  # noqa: D102
        return next(self.execute(extract_transform.rstrip(" \t\n;") + " limit 100;"))

    @requires_credentials
    def execute(  # noqa: D102
        self,
        extract_transform: str,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: ReturnTypeStr = "polars",
        keys: tuple[str, list[str]] | None = None,
    ) -> Generator[QueryReturnType, None, None]:
        batch_size = batch_size or 10_000
        with self.credentials.connect() as conn:
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
                connection=conn,
                rename=rename,
                batch_size=batch_size,
                return_batches=True,
                return_type=return_type,
            )

    @classmethod
    def from_engine(cls, engine: Engine) -> "RelationalDBLocation":
        """Create a RelationalDBLocation from a SQLAlchemy Engine."""
        cleaned_url = engine.url.set(
            username=None,
            password=None,
            query={},
        )
        location = cls(uri=str(cleaned_url))
        location.add_credentials(engine)
        return location


class SourceField(BaseModel):
    """A field in a source that can be indexed in the Matchbox database."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description=(
            "The name of the field in the source after the "
            "extract/transform logic has been applied."
        )
    )
    type: DataTypes = Field(
        description="The cached field type. Used to ensure a stable hash.",
    )


class SourceConfig(BaseModel):
    """Configuration of a source that can, or has been, indexed in the backend.

    SourceConfigs are used to configure source resolutions. They are foundational
    processes on top of which linking and deduplication models can build new
    resolutions.
    """

    model_config = ConfigDict(frozen=True)

    location: LocationType = Field(
        discriminator="type",
        description=(
            "The location of the source. Used to run the extract/tansform logic."
        ),
    )
    name: SourceResolutionName = Field(
        description=(
            "A unique, human-readable name of the source resolution this "
            "object configures."
        )
    )
    extract_transform: str = Field(
        description=(
            "Logic to extract and transform data from the source. "
            "Language is location dependent."
        )
    )
    # Fields can to be set at creation, or initialised with `.default_columns()`
    key_field: SourceField = Field(
        description=textwrap.dedent("""
            The key field. This is the source's key for unique
            entities, such as a primary key in a relational database.

            Keys must ALWAYS be a string.

            For example, if the source describes companies, it may have used
            a Companies House number as its key.

            This key is ALWAYS correct. It should be something generated and
            owned by the source being indexed.
            
            For example, your organisation's CRM ID is a key field within the CRM.
            
            A CRM ID entered by hand in another dataset shouldn't be used 
            as a key field.
        """),
    )
    index_fields: tuple[SourceField, ...] = Field(
        default=None,
        description=textwrap.dedent(
            """
            The fields to index in this source, after the extract/transform logic 
            has been applied. 

            This is usually set manually, and should map onto the columns that the
            extract/transform logic returns.
            """
        ),
    )

    @property
    def prefix(self) -> str:
        """Get the prefix for the source."""
        return self.name + "_"

    @property
    def qualified_key(self) -> str:
        """Get the qualified key for the source."""
        return self.f(self.key_field.name)

    @property
    def qualified_fields(self) -> list[str]:
        """Get the qualified fields for the source."""
        return self.f([field.name for field in self.index_fields])

    def f(self, fields: str | Iterable[str]) -> str | list[str]:
        """Qualify one or more field names with the source name.

        Args:
            fields: The field name to qualify, or a list of field names.

        Returns:
            A single qualified field, or a list of qualified field names.

        """
        if isinstance(fields, str):
            return self.prefix + fields
        return [self.prefix + field_name for field_name in fields]

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Ensure the name is a valid source resolution name.

        Raises:
            ValueError: If the name is not a valid source resolution name.
        """
        if not re.match(r"^[a-zA-Z0-9_]+$", value):
            raise ValueError(
                "Source resolution names must be alphanumeric and underscore only. "
            )
        return value

    @model_validator(mode="after")
    def validate_key_field(self) -> Self:
        """Ensure that the key field is a string and not in the index fields."""
        if self.key_field in self.index_fields:
            raise ValueError("Key field must not be in the index fields. ")

        if self.key_field.type != DataTypes.STRING:
            raise ValueError("Key field must be a string. ")

        return self

    @classmethod
    def new(
        cls,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: str,
        index_fields: list[str],
    ) -> "SourceConfig":
        """Create a new SourceConfig for an indexing operation."""
        # Assumes credentials have been set on location
        sample: pl.DataFrame = location.head(extract_transform)

        remote_fields = {
            col.name: SourceField(name=col.name, type=DataTypes.from_dtype(col.dtype))
            for col in sample.iter_columns()
        }

        if remote_fields[key_field].type != DataTypes.STRING:
            raise ValueError(
                f"Your key_field, {key_field}, must coerce to a string "
                "in the extract transform logic. "
            )

        typed_key_field = SourceField(name=key_field, type=DataTypes.STRING)
        typed_index_fields = tuple(remote_fields[field] for field in index_fields)

        return cls(
            location=location,
            name=name,
            extract_transform=extract_transform,
            key_field=typed_key_field,
            index_fields=typed_index_fields,
        )

    def query(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: ReturnTypeStr = "polars",
        keys: list[str] | None = None,
    ) -> Generator[QueryReturnType, None, None]:
        """Applies the extract/transform logic to the source and returns the results.

        Args:
            qualify_names: If True, qualify the names of the columns with the
                source name.
            batch_size: Indicate the size of each batch when processing data in batches.
            return_type: The type of data to return. Defaults to "polars".
            keys: List of keys to select a subset of all source entries.

        Returns:
            The requested data in the specified format, as an iterator of tables.
        """
        _rename: Callable | None = None
        if qualify_names:

            def _rename(c: str) -> str:
                return self.name + "_" + c

        if keys:
            yield from self.location.execute(
                extract_transform=self.extract_transform,
                rename=_rename,
                batch_size=batch_size,
                return_type=return_type,
                keys=(self.key_field.name, keys),
            )
        else:
            yield from self.location.execute(
                extract_transform=self.extract_transform,
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

        key_field: str = self.key_field.name
        index_fields: list[str] = [field.name for field in self.index_fields]

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


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: SourceResolutionName
    source_id: set[str] = Field(default_factory=set)
    target: SourceResolutionName
    target_id: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def found_or_none(self) -> "Match":
        """Ensure that a match has sources and a cluster if target was found."""
        if self.target_id and not (self.source_id and self.cluster):
            raise ValueError(
                "A match must have sources and a cluster if target was found."
            )
        if self.cluster and not self.source_id:
            raise ValueError("A match must have source if cluster is set.")
        return self
