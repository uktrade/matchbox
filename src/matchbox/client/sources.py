"""Interface to source data."""

from collections.abc import Callable, Generator, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, overload

import polars as pl
import pyarrow as pa
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from pyarrow import parquet as pq

from matchbox.client.queries import Query
from matchbox.client.steps import StepABC, post_run
from matchbox.common.arrow import SCHEMA_INDEX
from matchbox.common.datatypes import DataTypes
from matchbox.common.db import QueryReturnClass, QueryReturnType
from matchbox.common.dtos import (
    SourceConfig,
    SourceField,
    SourceStepName,
    SourceStepPath,
    Step,
    StepType,
)
from matchbox.common.hash import HashMethod, hash_rows
from matchbox.common.logging import logger, profile_time

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.locations import Location
else:
    DAG = Any
    Location = Any


T = TypeVar("T")


class Source(StepABC):
    """Client-side wrapper for source configs."""

    _local_data_schema: ClassVar[pa.Schema] = SCHEMA_INDEX

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
        super().__init__(dag=dag, name=name, description=description)

        if validate_etl:
            location.validate_extract_transform(extract_transform)

        self.location = location
        self.extract_transform = extract_transform

        if infer_types:
            self._validate_fields(key_field, index_fields, str)

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

    @property
    def hashes(self) -> pl.DataFrame | None:
        """The locally computed hashes. Alias for local_data."""
        return self._local_data

    @hashes.setter
    def hashes(self, value: pl.DataFrame | None) -> None:
        self._local_data = value

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
    def sources(self) -> set[SourceStepName]:
        """Set of source names upstream of this node."""
        return {self.name}

    @post_run
    def to_dto(self) -> Step:
        """Convert to Step DTO for API calls."""
        return Step(
            description=self.description,
            step_type=StepType.SOURCE,
            config=self.config,
            fingerprint=self._fingerprint(),
        )

    @classmethod
    def from_dto(
        cls,
        step: Step,
        step_name: str,
        dag: DAG,
        location: Location,
        **kwargs: Any,
    ) -> "Source":
        """Reconstruct from Step DTO."""
        if step.step_type != StepType.SOURCE:
            raise ValueError("Step must be of type 'source'")

        return cls(
            dag=dag,
            location=location,
            name=SourceStepName(step_name),
            extract_transform=step.config.extract_transform,
            key_field=step.config.key_field,
            index_fields=step.config.index_fields,
            description=step.description,
            infer_types=False,
        )

    @overload
    def fetch(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: Literal[QueryReturnType.POLARS] = ...,
        keys: list[str] | None = None,
        limit: int | None = None,
    ) -> Generator[PolarsDataFrame, None, None]: ...

    @overload
    def fetch(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: Literal[QueryReturnType.PANDAS] = ...,
        keys: list[str] | None = None,
        limit: int | None = None,
    ) -> Generator[PandasDataFrame, None, None]: ...

    @overload
    def fetch(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: Literal[QueryReturnType.ARROW] = ...,
        keys: list[str] | None = None,
        limit: int | None = None,
    ) -> Generator[ArrowTable, None, None]: ...

    def fetch(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        keys: list[str] | None = None,
        limit: int | None = None,
    ) -> Generator[QueryReturnClass, None, None]:
        """Apply the extract/transform logic to the source and return batches lazily.

        Args:
            qualify_names: If True, qualify the names of the columns with the
                source name.
            batch_size: Indicate the size of each batch when fetching data in batches.
            return_type: The type of data to return. Defaults to "polars".
            keys: List of keys to select a subset of all source entries.
            limit: Maximum number of rows to return. If None, returns all rows.

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
                limit=limit,
            )
        else:
            yield from self.location.execute(
                extract_transform=self.config.extract_transform,
                schema_overrides=schema_overrides,
                rename=_rename,
                batch_size=batch_size,
                return_type=return_type,
                limit=limit,
            )

    def sample(
        self, n: int = 100, return_type: QueryReturnType = QueryReturnType.POLARS
    ) -> QueryReturnClass:
        """Peek at the top n entries in a source.

        Args:
            n: The number of rows to return. Must be a positive integer.
            return_type: The type of data to return. Defaults to "polars".

        Returns:
            A dataframe containing the first n rows of the source.

        Raises:
            TypeError: If n is not an integer.
            ValueError: If n is not a positive integer.
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        return next(self.fetch(limit=n, return_type=return_type))

    @profile_time(attr="name")
    def run(self, batch_size: int | None = None) -> pl.DataFrame:
        """Hash a dataset from its warehouse, ready to be inserted, and cache hashes.

        Hashes the index fields defined in the source based on the
        extract/transform logic. Does not hash the key field.

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
                writer = pq.ParquetWriter(
                    self.cache_path,
                    schema=batch.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            writer.write_table(batch)

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

        self._local_data = pl.concat(all_results).group_by("hash").agg(pl.col("keys"))

        return self._local_data

    @property
    def path(self) -> SourceStepPath:
        """Return the source step path."""
        return SourceStepPath(
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
        """Qualify a field name with the source name.

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

    def query(self, **kwargs: Any) -> Query:
        """Generate a query for this source."""
        return Query(self, **kwargs, dag=self.dag)
