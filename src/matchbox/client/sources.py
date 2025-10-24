"""Interface to source data."""

import warnings
from collections.abc import Callable, Generator, Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar, overload

import polars as pl
from pyarrow import Table as ArrowTable

from matchbox.client import _handler
from matchbox.client.queries import Query
from matchbox.common.db import (
    QueryReturnClass,
    QueryReturnType,
)
from matchbox.common.dtos import (
    DataTypes,
    Resolution,
    ResolutionPath,
    ResolutionType,
    SourceConfig,
    SourceField,
    SourceResolutionName,
    SourceResolutionPath,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
)
from matchbox.common.hash import HashMethod, hash_rows
from matchbox.common.logging import logger

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.locations import Location
else:
    DAG = Any
    Location = Any


T = TypeVar("T")


class Source:
    """Client-side wrapper for source configs."""

    @overload
    def __init__(
        self,
        dag: DAG,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: str,
        index_fields: list[str],
        description: str | None = None,
        infer_types: bool = True,
        validate_etl: bool = True,
    ) -> None: ...

    @overload
    def __init__(
        self,
        dag: DAG,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: SourceField,
        index_fields: list[SourceField],
        description: str | None = None,
        infer_types: bool = False,
        validate_etl: bool = True,
    ) -> None: ...

    def __init__(
        self,
        dag: DAG,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: str | SourceField,
        index_fields: list[str] | list[SourceField],
        description: str | None = None,
        infer_types: bool = False,
        validate_etl: bool = True,
    ):
        """Initialise source.

        Args:
            dag: DAG containing the source.
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
            validate_etl: Whether to skip query validation. If True, it will
                perform query validation. It should be False when loading sources from
                the server. Default True.
        """
        if validate_etl:
            location.validate_extract_transform(extract_transform)

        self.last_run: datetime | None = None
        self.location = location
        self.dag = dag
        self.name = name
        self.description = description
        self.extract_transform = extract_transform

        if infer_types:
            self._validate_fields(key_field, index_fields, str)

            # Assumes client has been set on location
            inferred_types = location.infer_types(extract_transform)
            remote_fields = {
                field_name: SourceField(name=field_name, type=dtype)
                for field_name, dtype in inferred_types.items()
            }
            self.key_field = SourceField(name=key_field, type=DataTypes.STRING)
            self.index_fields = tuple(remote_fields[field] for field in index_fields)
        else:
            self.key_field, self.index_fields = self._validate_fields(
                key_field, index_fields, SourceField
            )

    def _validate_fields(
        self,
        key_field: Any,
        index_fields: list[Any],
        type_check: type[str] | type[SourceField],
    ) -> tuple[T, tuple[T, ...]]:
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

    @property
    def config(self) -> SourceConfig:
        """Generate SourceConfig from Source."""
        return SourceConfig(
            location_config=self.location.config,
            extract_transform=self.extract_transform,
            key_field=self.key_field,
            index_fields=self.index_fields,
        )

    @property
    def dependencies(self) -> list[ResolutionPath]:
        """Returns all resolution paths this source needs.

        Provided to match the interface of Model objects.
        """
        return self.config.dependencies

    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        return Resolution(
            description=self.description,
            truth=None,
            resolution_type=ResolutionType.SOURCE,
            config=self.config,
        )

    @classmethod
    def from_resolution(
        cls,
        resolution: Resolution,
        resolution_name: str,
        dag: DAG,
        location: Location,
    ) -> "Source":
        """Reconstruct from Resolution."""
        if resolution.resolution_type != ResolutionType.SOURCE:
            raise ValueError("Resolution must be of type 'source'")

        return cls(
            dag=dag,
            location=location,
            name=SourceResolutionName(resolution_name),
            extract_transform=resolution.config.extract_transform,
            key_field=resolution.config.key_field,
            index_fields=resolution.config.index_fields,
            description=resolution.description,
            infer_types=False,
        )

    def __hash__(self) -> int:
        """Return a hash of the Source based on its config."""
        return hash(self.config)

    def __eq__(self, other: Any) -> bool:
        """Check equality of two Source objects based on their config."""
        if not isinstance(other, Source):
            return False
        return self.config == other.config

    def fetch(
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

    def run(
        self, batch_size: int | None = None, full_rerun: bool = False
    ) -> ArrowTable:
        """Hash a dataset from its warehouse, ready to be inserted, and cache hashes.

        Hashes the index fields defined in the source based on the
        extract/transform logic.

        Does not hash the key field.

        Args:
            batch_size: If set, process data in batches internally. Indicates the
                size of each batch.
            full_rerun: Whether to force a re-run even if the hashes are cached


        Returns:
            A PyArrow Table containing source keys and their hashes.
        """
        if self.last_run and not full_rerun:
            warnings.warn("Source already run, skipping.", UserWarning, stacklevel=2)
            return self.hashes

        log_prefix = f"Hash {self.name}"
        batch_info = (
            f"with batch size {batch_size:,}" if batch_size else "without batching"
        )
        logger.debug(f"Retrieving and hashing {batch_info}", prefix=log_prefix)

        key_field: str = self.config.key_field.name
        index_fields: list[str] = [field.name for field in self.config.index_fields]

        all_results: list[pl.DataFrame] = []
        for batch in self.fetch(
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

        self.hashes = (
            pl.concat(all_results).group_by("hash").agg(pl.col("keys")).to_arrow()
        )

        self.last_run = datetime.now()

        return self.hashes

    # Note: name, description, truth are now instance variables, not properties

    @property
    def resolution_path(self) -> SourceResolutionPath:
        """Returns the source resolution path."""
        return SourceResolutionPath(
            collection=self.dag.name,
            run=self.dag.run,
            name=self.name,
        )

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

    def sync(self) -> None:
        """Send the source config and hashes to the server."""
        resolution = self.to_resolution()
        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None
        # Check if config matches
        if existing_resolution:
            if existing_resolution.config != self.config:
                raise ValueError(
                    f"Resolution {self.resolution_path} already exists with different "
                    "configuration. Please delete the existing resolution "
                    "or use a different name. "
                )
            else:
                log_prefix = f"Resolution {self.resolution_path}"
                logger.warning("Already exists. Passing.", prefix=log_prefix)
        else:
            _handler.create_resolution(resolution=resolution, path=self.resolution_path)

        if self.hashes:
            _handler.set_data(
                path=self.resolution_path,
                data=self.hashes,
                validate_type=ResolutionType.SOURCE,
            )

    def query(self, **kwargs) -> Query:
        """Generate a query for this source."""
        return Query(self, **kwargs, dag=self.dag)
