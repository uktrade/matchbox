"""Interface to source data."""

from collections.abc import Callable, Generator, Iterable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, overload

import polars as pl
from pyarrow import Table as ArrowTable
from pyarrow import parquet as pq

from matchbox.client import _handler
from matchbox.client.queries import Query
from matchbox.common.datatypes import DataTypes
from matchbox.common.db import (
    QueryReturnClass,
    QueryReturnType,
)
from matchbox.common.dtos import (
    Resolution,
    ResolutionType,
    SourceConfig,
    SourceField,
    SourceResolutionName,
    SourceResolutionPath,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.hash import HashMethod, hash_arrow_table, hash_rows
from matchbox.common.logging import logger, profile_time

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.locations import Location
else:
    DAG = Any
    Location = Any


T = TypeVar("T")


def post_run(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that a method is called after source run.

    Raises:
        RuntimeError: If run hasn't happened.
    """

    @wraps(method)
    def wrapper(self: "Source", *args: Any, **kwargs: Any) -> T:
        if self.hashes is None:
            raise RuntimeError(
                "The source must be run before attempting this operation."
            )
        return method(self, *args, **kwargs)

    return wrapper


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

        self.location = location
        self.dag = dag
        self.name = name
        self.description = description
        self.extract_transform = extract_transform
        self.hashes: ArrowTable | None = None

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
        key_field: str | SourceField,
        index_fields: list[str | SourceField],
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
    def sources(self) -> set[SourceResolutionName]:
        """Set of source names upstream of this node."""
        return {self.name}

    @post_run
    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        return Resolution(
            description=self.description,
            truth=None,
            resolution_type=ResolutionType.SOURCE,
            config=self.config,
            fingerprint=hash_arrow_table(self.hashes),
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

    def __eq__(self, other: object) -> bool:
        """Check equality of two Source objects based on their config."""
        if not isinstance(other, Source):
            return False
        return self.config == other.config

    @property
    def cache_path(self) -> Path:
        """The path within the DAG cache for storing this source data."""
        return self.dag.cache_path / f"{self.name}.parquet"

    def fetch(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        keys: list[str] | None = None,
    ) -> Generator[QueryReturnClass, None, None]:
        """Applies the extract/transform logic to the source and returns batches lazily.

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

    def sample(
        self, n: int = 100, return_type: QueryReturnType = QueryReturnType.POLARS
    ) -> None:
        """Peek at the top n entries in a source."""
        return next(self.fetch(batch_size=n, return_type=return_type))

    @profile_time(attr="name")
    def run(self, batch_size: int | None = None) -> ArrowTable:
        """Hash a dataset from its warehouse, ready to be inserted, and cache hashes.

        Hashes the index fields defined in the source based on the
        extract/transform logic.

        Does not hash the key field.

        Args:
            batch_size: If set, process data in batches internally. Indicates the
                size of each batch.
        """
        log_prefix = f"Run {self.name}"

        self.cache_path.unlink(missing_ok=True)

        logger.info("Retrieving source data", prefix=log_prefix)
        writer = None
        for batch in self.fetch(
            batch_size=batch_size, return_type=QueryReturnType.ARROW
        ):
            if writer is None:
                # Initialise writer with the first batch's schema
                writer = pq.ParquetWriter(
                    self.cache_path,
                    schema=batch.schema,
                    compression="snappy",
                    use_dictionary=True,
                )

            writer.write_table(batch)  # appends as a new row group

        if writer is not None:
            writer.close()

        key_field: str = self.config.key_field.name
        index_fields: list[str] = [field.name for field in self.config.index_fields]

        all_results: list[pl.DataFrame] = []

        pf = pq.ParquetFile(self.cache_path)
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            batch = pl.from_arrow(table)

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

        return self.hashes

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

    @post_run
    @profile_time(attr="name")
    def sync(self) -> None:
        """Send the source config and hashes to the server.

        Not resistant to race conditions: only one client should call sync at a time.
        """
        log_prefix = f"Sync {self.name}"
        resolution = self.to_resolution()
        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
            logger.info("Found existing resolution", prefix=log_prefix)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None

        if existing_resolution:
            if (existing_resolution.fingerprint == resolution.fingerprint) and (
                existing_resolution.config.parents == resolution.config.parents
            ):
                logger.info("Updating existing resolution", prefix=log_prefix)
                # Assumes that resolution hasn't been deleted or made incompatible
                # Else, server will error
                _handler.update_resolution(
                    resolution=resolution, path=self.resolution_path
                )
            else:
                logger.info(
                    "Update not possible. Deleting existing resolution",
                    prefix=log_prefix,
                )
                # Assumes that resolution hasn't been deleted, else server will error
                _handler.delete_resolution(path=self.resolution_path, certain=True)
                existing_resolution = None

        if not existing_resolution:
            logger.info("Creating new resolution", prefix=log_prefix)
            # Assumes that resolution hasn't since been re-created.
            # Else, server will error
            _handler.create_resolution(resolution=resolution, path=self.resolution_path)
            # Assumes resolution has not been deleted or made incompatible
            logger.info("Setting data for new resolution", prefix=log_prefix)
            _handler.set_data(path=self.resolution_path, data=self.hashes)

    def query(self, **kwargs: Any) -> Query:
        """Generate a query for this source."""
        return Query(self, **kwargs, dag=self.dag)

    def clear_data(self) -> None:
        """Deletes data computed for node."""
        self.hashes = None
