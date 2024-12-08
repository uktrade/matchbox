import base64
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, overload

import connectorx as cx
import pyarrow as pa
from matchbox.common.exceptions import MatchboxValidatonError
from matchbox.common.hash import HASH_FUNC
from pandas import DataFrame
from pyarrow import Table as ArrowTable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    ColumnElement,
    MetaData,
    String,
    Table,
    cast,
    create_engine,
    select,
)
from sqlalchemy import text as sqltext
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.selectable import Select

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any

ReturnTypeStr = Literal["arrow", "pandas", "polars"]

T = TypeVar("T")


class Probability(BaseModel):
    """A probability of a match in the Matchbox database.

    A probability describes the likelihood of a match between two clusters.
    """

    hash: bytes
    left: bytes
    right: bytes
    probability: float = Field(default=None, ge=0, le=1)


class Cluster(BaseModel):
    """A cluster of data in the Matchbox database.

    A cluster describes a single entity resolved at the specified probability
    threshold or higher.
    """

    parent: bytes
    children: set[bytes]
    threshold: float = Field(default=None, ge=0, le=1)


class SourceWarehouse(BaseModel):
    """A warehouse where source data for datasets in Matchbox can be found."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    alias: str
    db_type: str
    user: str
    password: str = Field(repr=False)
    host: str
    port: int
    database: str
    _engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            connection_string = f"{self.db_type}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self._engine = create_engine(connection_string)
            self.test_connection()
        return self._engine

    def __str__(self):
        return (
            f"SourceWarehouse(alias={self.alias}, type={self.db_type}, "
            f"host={self.host}, port={self.port}, database={self.database})"
        )

    def __eq__(self, other):
        if not isinstance(other, SourceWarehouse):
            return False
        return (
            self.alias == other.alias
            and self.db_type == other.db_type
            and self.user == other.user
            and self.password == other.password
            and self.host == other.host
            and self.port == other.port
            and self.database == other.database
        )

    def test_connection(self):
        try:
            with self.engine.connect() as connection:
                connection.execute(sqltext("SELECT 1"))
        except SQLAlchemyError:
            self._engine = None
            raise

    @classmethod
    def from_engine(cls, engine: Engine, alias: str | None = None) -> "SourceWarehouse":
        """Create a SourceWarehouse instance from an SQLAlchemy Engine object."""
        url = engine.url

        warehouse = cls(
            alias=alias or url.database,
            db_type=url.drivername,
            user=url.username,
            password=url.password,
            host=url.host,
            port=url.port or 0,
            database=url.database,
        )
        _ = warehouse.engine

        return warehouse


class SourceColumnName(BaseModel):
    """A column name in the Matchbox database."""

    name: str

    @property
    def hash(self) -> bytes:
        """Generate a unique hash based on the column name."""
        return HASH_FUNC(self.name.encode("utf-8")).digest()

    @property
    def base64(self) -> str:
        """Generate a base64 encoded hash based on the column name."""
        return base64.b64encode(self.hash).decode("utf-8")


class SourceColumn(BaseModel):
    """A column in a dataset that can be indexed in the Matchbox database."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    literal: SourceColumnName = Field(
        description="The literal name of the column in the database."
    )
    alias: SourceColumnName = Field(
        default_factory=lambda data: SourceColumnName(name=data["literal"].name),
        description="The alias to use when hashing the dataset in Matchbox.",
    )
    type: str | None = Field(
        default=None, description="The type to cast the column to before hashing data."
    )
    indexed: bool = Field(description="Whether the column is indexed in the database.")

    def __eq__(self, other: object) -> bool:
        """Compare SourceColumn with another SourceColumn or bytes object.

        Two SourceColumns are equal if:

        * Their literal names match, or
        * Their alias names match, or
        * The hash of either their literal or alias matches the other object's
        corresponding hash

        A SourceColumn is equal to a bytes object if:

        * The hash of either its literal or alias matches the bytes object

        Args:
            other: Another SourceColumn or a bytes object to compare against

        Returns:
            bool: True if the objects are considered equal, False otherwise
        """
        if isinstance(other, SourceColumn):
            if self.literal == other.literal or self.alias == other.alias:
                return True

            self_hashes = {self.literal.hash, self.alias.hash}
            other_hashes = {other.literal.hash, other.alias.hash}

            return bool(self_hashes & other_hashes)

        if isinstance(other, bytes):
            return other in {self.literal.hash, self.alias.hash}

        return NotImplemented

    @field_validator("literal", "alias", mode="before")
    def string_to_name(cls: "SourceColumn", value: str) -> SourceColumnName:
        if isinstance(value, str):
            return SourceColumnName(name=value)
        else:
            raise ValueError("Column name must be a string.")


class SourceIndex(BaseModel):
    """The hashes of column names in the Matchbox database."""

    literal: list[bytes]
    alias: list[bytes]


class Source(BaseModel):
    """A dataset that can be indexed in the Matchbox database."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    database: SourceWarehouse
    db_pk: str
    db_schema: str
    db_table: str
    db_columns: list[SourceColumn]
    alias: str = Field(
        default_factory=lambda data: f"{data['db_schema']}.{data['db_table']}"
    )

    def __str__(self) -> str:
        return f"{self.db_schema}.{self.db_table}"

    def __hash__(self) -> int:
        return hash(
            (type(self), self.db_pk, self.db_schema, self.db_table, self.database.alias)
        )

    @model_validator(mode="before")
    @classmethod
    def hash_columns(cls, data: dict[str, Any]) -> "Source":
        """Shapes indices data from either the backend or TOML.

        Handles three scenarios:
            1. No columns specified - all columns except primary key are indexed
            2. Columns specified in TOML - uses specified columns with optional '*'
            3. Indices from database - uses existing column hash information
        """
        # Initialise warehouse and get table metadata
        warehouse = (
            data["database"]
            if isinstance(data["database"], SourceWarehouse)
            else SourceWarehouse(**data["database"])
        )

        metadata = MetaData(schema=data["db_schema"])
        table = Table(data["db_table"], metadata, autoload_with=warehouse.engine)

        # Get all columns except primary key
        remote_columns = [
            SourceColumn(literal=col.name, type=str(col.type), indexed=False)
            for col in table.columns
            if col.name not in data["db_pk"]
        ]

        index_data = data.get("index")

        # Case 1: No columns specified - index everything
        if not index_data:
            data["db_columns"] = [
                SourceColumn(literal=col.literal.name, type=col.type, indexed=True)
                for col in remote_columns
            ]
            return data

        # Case 2: Columns from database
        if isinstance(index_data, dict):
            source_index = SourceIndex(**index_data)
            data["db_columns"] = [
                SourceColumn(
                    literal=col.literal.name,
                    type=col.type,
                    indexed=col in source_index.literal + source_index.alias,
                )
                for col in remote_columns
            ]
            return data

        # Case 3: Columns from TOML
        local_columns = []
        star_index = None

        # Process TOML column specifications
        for i, column in enumerate(index_data):
            if column["literal"] == "*":
                star_index = i
                continue
            local_columns.append(SourceColumn(**column, indexed=True))

        # Match remote columns with local specifications
        indexed_columns = []
        non_indexed_columns = []

        for remote_col in remote_columns:
            matched = False
            for local_col in local_columns:
                if remote_col == local_col:
                    if local_col.type is None:
                        local_col.type = remote_col.type
                    indexed_columns.append(local_col)
                    matched = True
                    break
            if not matched:
                non_indexed_columns.append(remote_col)

        # Handle wildcard insertion
        if star_index is not None:
            for col in non_indexed_columns:
                col.indexed = True
            indexed_columns[star_index:star_index] = non_indexed_columns
            data["db_columns"] = indexed_columns
        else:
            data["db_columns"] = indexed_columns + non_indexed_columns

        return data

    def to_table(self) -> Table:
        """Returns the dataset as a SQLAlchemy Table object."""
        metadata = MetaData(schema=self.db_schema)
        table = Table(self.db_table, metadata, autoload_with=self.database.engine)
        return table

    def _select(
        self,
        fields: list[str] | None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> Select:
        """Returns a SQLAlchemy Select object to retrieve data from the dataset."""
        table = self.to_table()

        def _get_column(col_name: str) -> ColumnElement:
            """Helper to get a column with proper casting and labeling for PKs"""
            col = table.columns[col_name]
            if col_name == self.db_pk:
                return cast(col, String).label(
                    f"{table.schema}_{table.name}_{col_name}"
                )
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

    def to_hash(self) -> bytes:
        """Generate a unique hash based on the table's columns and datatypes."""
        schema_representation = f"{self.alias}: " + ",".join(
            f"{col.alias.name}:{col.type}" for col in self.db_columns if col.indexed
        )
        return HASH_FUNC(schema_representation.encode("utf-8")).digest()

    def to_arrow(
        self,
        fields: list[str] | None = None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> ArrowTable:
        """Returns the dataset as a PyArrow Table."""
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self.database.engine, return_type="arrow")

    def to_pandas(
        self,
        fields: list[str] | None,
        pks: list[T] | None = None,
        limit: int | None = None,
    ) -> DataFrame:
        """Returns the dataset as a pandas DataFrame."""
        stmt = self._select(fields=fields, pks=pks, limit=limit)
        return sql_to_df(stmt, self.database.engine, return_type="pandas")


def convert_large_binary_to_binary(table: pa.Table) -> pa.Table:
    """Converts Arrow large_binary fields to binary type."""
    new_fields = []
    for field in table.schema:
        if pa.types.is_large_binary(field.type):
            new_fields.append(field.with_type(pa.binary()))
        else:
            new_fields.append(field)

    new_schema = pa.schema(new_fields)
    return table.cast(new_schema)


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["arrow"]
) -> pa.Table: ...


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["pandas"]
) -> DataFrame: ...


@overload
def sql_to_df(
    stmt: Select, engine: Engine, return_type: Literal["polars"]
) -> PolarsDataFrame: ...


def sql_to_df(
    stmt: Select, engine: Engine, return_type: ReturnTypeStr = "pandas"
) -> pa.Table | DataFrame | PolarsDataFrame:
    """
    Executes the given SQLAlchemy statement using connectorx.

    Args:
        stmt (Select): A SQLAlchemy Select statement to be executed.
        engine (Engine): A SQLAlchemy Engine object for the database connection.

    Returns:
        A dataframe of the query results.

    Raises:
        ValueError: If the engine URL is not properly configured.
    """
    compiled_stmt = stmt.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )
    sql_query = str(compiled_stmt)

    url: Union[str, URL] = engine.url

    if isinstance(url, URL):
        url = url.render_as_string(hide_password=False)

    if not isinstance(url, str):
        raise ValueError("Unable to obtain a valid connection string from the engine.")

    result = cx.read_sql(conn=url, query=sql_query, return_type=return_type)

    if return_type == "arrow":
        return convert_large_binary_to_binary(table=result)

    return result


def get_schema_table_names(full_name: str, validate: bool = False) -> tuple[str, str]:
    """
    Takes a string table name and returns the unquoted schema and
    table as a tuple. If you insert these into a query, you need to
    add double quotes in from statements, or single quotes in where.

    Parameters:
        full_name: A string indicating a Postgres table
        validate: Whether to error if both schema and table aren't
        detected

    Raises:
        ValueError: When the function can't detect either a
        schema.table or table format in the input
        MatchboxValidatonError: If both schema and table can't be detected
        when the validate argument is True

    Returns:
        (schema, table): A tuple of schema and table name. If schema
        cannot be inferred, returns None.
    """

    schema_name_list = full_name.replace('"', "").split(".")

    if len(schema_name_list) == 1:
        schema = None
        table = schema_name_list[0]
    elif len(schema_name_list) == 2:
        schema = schema_name_list[0]
        table = schema_name_list[1]
    else:
        raise ValueError(f"Could not identify schema and table in {full_name}.")

    if validate and schema is None:
        raise MatchboxValidatonError(
            "Schema could not be detected and validation required."
        )

    return (schema, table)
