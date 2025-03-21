"""Classes and functions for working with data sources in Matchbox."""

import json
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Iterator, ParamSpec, TypeVar

import polars as pl
import pyarrow as pa
from pandas import DataFrame
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    WithJsonSchema,
    model_validator,
)
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    ColumnElement,
    Engine,
    MetaData,
    String,
    Table,
    cast,
    select,
)
from sqlalchemy.sql.selectable import Select
from typing_extensions import Annotated

from matchbox.common.db import fullname_to_prefix, get_schema_table_names, sql_to_df
from matchbox.common.exceptions import (
    MatchboxSourceColumnError,
    MatchboxSourceEngineError,
)
from matchbox.common.hash import HASH_FUNC, base64_to_hash, hash_data, hash_to_base64

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class SourceColumn(BaseModel):
    """A column in a dataset that can be indexed in the Matchbox database."""

    model_config = ConfigDict(frozen=True)

    name: str
    alias: str = Field(default_factory=lambda data: data["name"])
    type: str | None = Field(
        default=None, description="The type to cast the column to before hashing data."
    )


def b64_bytes_validator(val: bytes | str) -> bytes:
    """Ensure that a value is a base64 encoded string or bytes."""
    if isinstance(val, bytes):
        return val
    elif isinstance(val, str):
        return base64_to_hash(val)
    raise ValueError(f"Value {val} could not be converted to bytes")


SerialisableBytes = Annotated[
    bytes,
    PlainValidator(b64_bytes_validator),
    PlainSerializer(lambda v: hash_to_base64(v)),
    WithJsonSchema(
        {"type": "string", "format": "base64", "description": "Base64 encoded bytes"}
    ),
]


class SourceAddress(BaseModel):
    """A unique identifier for a dataset in a warehouse."""

    model_config = ConfigDict(frozen=True)

    full_name: str
    warehouse_hash: SerialisableBytes

    def __str__(self) -> str:
        """Convert to a string."""
        return self.full_name + "@" + hash_to_base64(self.warehouse_hash)

    @classmethod
    def compose(cls, engine: Engine, full_name: str) -> "SourceAddress":
        """Generate a SourceAddress from a SQLAlchemy Engine and full source name."""
        url = engine.url
        components = {
            "dialect": url.get_dialect().name,
            "database": url.database or "",
            "host": url.host or "",
            "port": url.port or "",
            "schema": getattr(url, "schema", "") or "",
            "service_name": url.query.get("service_name", ""),
        }

        stable_str = json.dumps(components, sort_keys=True).encode()

        hash = HASH_FUNC(stable_str).digest()
        return SourceAddress(full_name=full_name, warehouse_hash=hash)


def needs_engine(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to ensure Engine is available to object."""

    @wraps(func)
    def wrapper(self: "Source", *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.engine:
            raise MatchboxSourceEngineError
        return func(self, *args, **kwargs)

    return wrapper


class Source(BaseModel):
    """A dataset that can, or has been indexed on the backend."""

    model_config = ConfigDict(frozen=True)

    address: SourceAddress
    resolution_name: str = Field(default_factory=lambda data: str(data["address"]))
    db_pk: str
    # Columns need to be set at creation, or initialised with `.default_columns()`
    columns: tuple[SourceColumn, ...] | None = None

    _engine: Engine | None = None

    @property
    def engine(self) -> Engine | None:
        """The SQLAlchemy Engine used to connect to the dataset."""
        return self._engine

    def __eq__(self, other: Any) -> bool:
        """Custom equality which ignores engine."""
        return (self.address, self.resolution_name, self.db_pk, self.columns) == (
            other.address,
            other.resolution_name,
            other.db_pk,
            other.columns,
        )

    def __hash__(self) -> int:
        """Custom hash which ignores engine."""
        return hash((self.address, self.resolution_name, self.db_pk, self.columns))

    def __deepcopy__(self, memo=None):
        """Create a deep copy of the Source object."""
        if memo is None:
            memo = {}

        obj_copy = Source(
            address=deepcopy(self.address, memo),
            resolution_name=deepcopy(self.resolution_name, memo),
            db_pk=deepcopy(self.db_pk, memo),
            columns=deepcopy(self.columns, memo),
        )

        # Both objects should share the same engine
        if self._engine is not None:
            obj_copy._engine = self._engine

        return obj_copy

    def set_engine(self, engine: Engine):
        """Adds engine, and use it to validate current columns."""
        implied_address = SourceAddress.compose(
            full_name=self.address.full_name, engine=engine
        )
        if implied_address != self.address:
            raise ValueError(
                "The engine must be the same that was used to index the source"
            )

        self._engine = engine
        remote_columns = self._get_remote_columns()
        for col in self.columns:
            if col.name not in remote_columns:
                raise MatchboxSourceColumnError(
                    f"Column {col.name} not available in {self.address.full_name}"
                )
            actual_type = str(remote_columns[col.name])
            if actual_type != col.type:
                raise MatchboxSourceColumnError(
                    f"Type {actual_type} != {col.type} for {col.name}"
                )
        return self

    @property
    def signature(self) -> bytes:
        """Generate a unique hash based on the table's metadata."""
        sorted_columns = sorted(self.columns, key=lambda c: c.alias)
        schema_representation = f"{self.resolution_name}: " + ",".join(
            f"{col.alias}:{col.type}" for col in sorted_columns
        )
        return HASH_FUNC(schema_representation.encode("utf-8")).digest()

    def format_column(self, column: str) -> str:
        """Outputs a full SQLAlchemy column representation.

        Args:
            column: the name of the column

        Returns:
            A string representing the table name and column
        """
        return fullname_to_prefix(self.address.full_name) + column

    @needs_engine
    def _get_remote_columns(self) -> dict[str, str]:
        table = self.to_table()
        return {
            col.name: col.type for col in table.columns if col.name not in self.db_pk
        }

    @needs_engine
    def default_columns(self) -> "Source":
        """Returns a new source with default columns.

        Default columns are all from the source warehouse other than `self.db_pk`.
        All other attributes are copied, and its engine (if present) is set.
        """
        remote_columns = self._get_remote_columns()
        columns_attribute = (
            SourceColumn(name=col_name, type=str(col_type))
            for col_name, col_type in remote_columns.items()
        )

        new_source = Source(
            address=self.address,
            resolution_name=self.resolution_name,
            db_pk=self.db_pk,
            columns=columns_attribute,
        )

        if self.engine:
            new_source.set_engine(self.engine)

        return new_source

    @needs_engine
    def to_table(self) -> Table:
        """Returns the dataset as a SQLAlchemy Table object."""
        db_schema, db_table = get_schema_table_names(self.address.full_name)
        metadata = MetaData(schema=db_schema)
        table = Table(db_table, metadata, autoload_with=self.engine)
        return table

    def _select(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> Select:
        """Returns a SQLAlchemy Select object to retrieve data from the dataset."""
        table = self.to_table()

        if not fields:
            fields = [col.name for col in self.columns]

        if self.db_pk not in fields:
            fields.append(self.db_pk)

        def _get_column(col_name: str) -> ColumnElement:
            """Helper to get a column with proper casting and labeling for PKs."""
            col = table.columns[col_name]
            if col_name == self.db_pk:
                return cast(col, String).label(self.format_column(col_name))
            return col

        # Determine which columns to select
        if fields:
            select_cols = [_get_column(field) for field in fields]
        else:
            select_cols = [_get_column(col.name) for col in table.columns]

        stmt = select(*select_cols)

        if pks:
            string_pks = [str(pk) for pk in pks]
            pk_col = table.columns[self.db_pk]
            stmt = stmt.where(cast(pk_col, String).in_(string_pks))

        if limit:
            stmt = stmt.limit(limit)

        return stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

    @needs_engine
    def to_arrow(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
        *,
        iter_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> pa.Table | Iterator[pa.Table]:
        """Returns the dataset as a PyArrow Table or an iterator of PyArrow Tables.

        Args:
            fields: List of column names to retrieve. If None, retrieves all columns.
            pks: List of primary keys to filter by. If None, retrieves all rows.
            limit: Maximum number of rows to retrieve. If None, retrieves all rows.
            iter_batches: If True, return an iterator that yields each batch separately.
                If False, return a single Table with all results.
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping column names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            If iter_batches is False: A PyArrow Table containing the requested data.
            If iter_batches is True: An iterator of PyArrow Tables.
        """
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(
            stmt,
            self._engine,
            return_type="arrow",
            iter_batches=iter_batches,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

    @needs_engine
    def to_pandas(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
        *,
        iter_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> DataFrame | Iterator[DataFrame]:
        """Returns the dataset as a pandas DataFrame or an iterator of DataFrames.

        Args:
            fields: List of column names to retrieve. If None, retrieves all columns.
            pks: List of primary keys to filter by. If None, retrieves all rows.
            limit: Maximum number of rows to retrieve. If None, retrieves all rows.
            iter_batches: If True, return an iterator that yields each batch separately.
                If False, return a single DataFrame with all results.
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping column names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            If iter_batches is False: A pandas DataFrame containing the requested data.
            If iter_batches is True: An iterator of pandas DataFrames.
        """
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(
            stmt,
            self._engine,
            return_type="pandas",
            iter_batches=iter_batches,
            batch_size=batch_size,
            schema_overrides=schema_overrides,
            execute_options=execute_options,
        )

    @needs_engine
    def hash_data(
        self,
        *,
        iter_batches: bool = False,
        batch_size: int | None = None,
        schema_overrides: dict[str, Any] | None = None,
        execute_options: dict[str, Any] | None = None,
    ) -> pa.Table:
        """Retrieve and hash a dataset from its warehouse, ready to be inserted.

        Args:
            iter_batches: If True, process data in batches internally.
                This is only used for internal processing and will still return a
                single table.
            batch_size: Indicate the size of each batch when processing data in batches.
            schema_overrides: A dictionary mapping column names to dtypes.
            execute_options: These options will be passed through into the underlying
                query execution method as kwargs.

        Returns:
            A PyArrow Table containing source primary keys and their hashes.
        """
        signature_hex = self.signature.hex()
        source_table = self.to_table()
        cols_to_index = tuple([col.name for col in self.columns])

        slct_stmt = select(
            *[source_table.c[col] for col in cols_to_index],
            source_table.c[self.db_pk].cast(String).label("source_pk"),
        )

        def _process_batch(
            batch: pl.DataFrame, cols_to_index: tuple, signature_hex: str
        ) -> pl.DataFrame:
            """Process a single batch of data using Polars.

            Args:
                batch: Polars DataFrame containing the data
                cols_to_index: Columns to include in the hash
                signature_hex: Signature to append to values before hashing

            Returns:
                Polars DataFrame with hash and source_pk columns
            """
            for col_name in cols_to_index:
                batch = batch.with_columns(pl.col(col_name).cast(pl.Utf8))

            batch = batch.with_columns(
                pl.concat_str([pl.col(c) for c in cols_to_index]).alias("raw_value")
            )

            batch = batch.with_columns(
                (pl.col("raw_value") + " " + pl.lit(signature_hex)).alias(
                    "value_with_sig"
                )
            )

            batch = batch.with_columns(
                pl.col("value_with_sig")
                .map_elements(lambda x: hash_data(x), return_dtype=pl.Binary)
                .alias("hash")
            )

            return batch.select(["hash", "source_pk"])

        if iter_batches:
            # Process in batches
            raw_batches = sql_to_df(
                slct_stmt,
                self._engine,
                return_type="polars",
                iter_batches=True,
                batch_size=batch_size,
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )

            all_results = []
            for batch in raw_batches:
                batch_result = _process_batch(batch, cols_to_index, signature_hex)
                all_results.append(batch_result)

            processed_df = pl.concat(all_results)

        else:
            # Non-batched processing
            raw_result = sql_to_df(
                slct_stmt,
                self._engine,
                return_type="polars",
                schema_overrides=schema_overrides,
                execute_options=execute_options,
            )

            processed_df = _process_batch(raw_result, cols_to_index, signature_hex)

        return processed_df.group_by("hash").agg(pl.col("source_pk")).to_arrow()


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: SourceAddress
    source_id: set[str] = Field(default_factory=set)
    target: SourceAddress
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
