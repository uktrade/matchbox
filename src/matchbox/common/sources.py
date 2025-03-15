import json
from copy import deepcopy
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import pyarrow as pa
from pandas import DataFrame
from pydantic import (
    BaseModel,
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
    func,
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

    name: str
    alias: str = Field(default_factory=lambda data: data["name"])
    type: str | None = Field(
        default=None, description="The type to cast the column to before hashing data."
    )


def b64_bytes_validator(val: bytes | str) -> bytes:
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
    full_name: str
    warehouse_hash: SerialisableBytes

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

    address: SourceAddress
    alias: str = Field(default_factory=lambda data: data["address"].full_name)
    db_pk: str
    columns: list[SourceColumn] = []

    _engine: Engine | None = None

    @property
    def engine(self) -> Engine | None:
        return self._engine

    def __deepcopy__(self, memo=None):
        """Create a deep copy of the Source object."""
        if memo is None:
            memo = {}

        obj_copy = Source(
            address=deepcopy(self.address, memo),
            alias=deepcopy(self.alias, memo),
            db_pk=deepcopy(self.db_pk, memo),
            columns=deepcopy(self.columns, memo),
        )

        # Both objects should share the same engine
        if self._engine is not None:
            obj_copy._engine = self._engine

        return obj_copy

    def set_engine(self, engine: Engine):
        """Adds engine, and use it to validate current columns"""
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
        schema_representation = f"{self.alias}: " + ",".join(
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
        """Overwrites columns with all non-primary keys from source warehouse."""
        remote_columns = self._get_remote_columns()
        self.columns = [
            SourceColumn(name=col_name, type=str(col_type))
            for col_name, col_type in remote_columns.items()
        ]

        return self

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
            """Helper to get a column with proper casting and labeling for PKs"""
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
    ) -> pa.Table:
        """Returns the dataset as a PyArrow Table."""
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self._engine, return_type="arrow")

    @needs_engine
    def to_pandas(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> DataFrame:
        """Returns the dataset as a pandas DataFrame."""
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self._engine, return_type="pandas")

    @needs_engine
    def hash_data(self) -> pa.Table:
        """Retrieve and hash a dataset from its warehouse, ready to be inserted."""
        source_table = self.to_table()
        cols_to_index = tuple([col.name for col in self.columns])

        slct_stmt = select(
            func.concat(*source_table.c[cols_to_index]).label("raw"),
            source_table.c[self.db_pk].cast(String).label("source_pk"),
        )

        raw_result = sql_to_df(slct_stmt, self._engine, "arrow")

        grouped = raw_result.group_by("raw").aggregate([("source_pk", "list")])
        grouped_data = pa.compute.binary_join_element_wise(
            grouped["raw"], self.signature.hex(), " "
        )
        grouped_keys = grouped["source_pk_list"]

        return pa.table(
            {
                "source_pk": grouped_keys,
                "hash": pa.array(
                    [hash_data(d) for d in grouped_data.to_pylist()],
                    type=pa.binary(),
                ),
            }
        )


class Match(BaseModel):
    """A match between primary keys in the Matchbox database."""

    cluster: int | None
    source: SourceAddress
    source_id: set[str] = Field(default_factory=set)
    target: SourceAddress
    target_id: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def found_or_none(self) -> "Match":
        if self.target_id and not (self.source_id and self.cluster):
            raise ValueError(
                "A match must have sources and a cluster if target was found."
            )
        if self.cluster and not self.source_id:
            raise ValueError("A match must have source if cluster is set.")
        return self
