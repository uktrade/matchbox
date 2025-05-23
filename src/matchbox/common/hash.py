"""Utilities for hashing data and creating unique identifiers."""

import base64
import hashlib
from enum import Flag, StrEnum
from typing import Iterable, Self, TypeVar
from uuid import UUID

import polars as pl
import polars.expr as plx
import polars_hash as plh
import pyarrow as pa
from pandas import DataFrame, Series

T = TypeVar("T")
HashableItem = TypeVar("HashableItem", bytes, bool, str, int, float, bytearray)

HASH_FUNC = hashlib.sha256


class HashMethod(StrEnum):
    """Supported hash methods for row hashing."""

    XXH3_128 = "xxh3_128"
    SHA256 = "sha256"


def hash_to_base64(hash: bytes) -> str:
    """Converts a hash to a base64 string."""
    return base64.urlsafe_b64encode(hash).decode("utf-8")


def base64_to_hash(b64: str) -> bytes:
    """Converts a base64 string to a hash."""
    return base64.urlsafe_b64decode(b64)


def prep_for_hash(item: HashableItem) -> bytes:
    """Encodes strings so they can be hashed, otherwises, passes through."""
    if isinstance(item, bytes):
        return item
    elif isinstance(item, str):
        return bytes(item.encode())
    elif isinstance(item, UUID):
        return item.bytes
    elif isinstance(item, int):
        # https://stackoverflow.com/a/54141411
        signed = True
        length = ((item + ((item * signed) < 0)).bit_length() + 7 + signed) // 8
        return item.to_bytes(length, byteorder="big", signed=signed)
    else:
        raise ValueError(f"Cannot hash value of type {type(item)}")


def hash_data(data: HashableItem) -> bytes:
    """Hash the given data using the globally defined hash function.

    This function ties into the existing hashing utilities.
    """
    return HASH_FUNC(prep_for_hash(data)).digest()


def hash_values(*values: tuple[T, ...]) -> bytes:
    """Returns a single hash of a tuple of items ordered by its values.

    List must be sorted as the different orders of value must produce the same hash.
    """
    try:
        sorted_vals = sorted(values)
    except TypeError as e:
        raise TypeError("Can only order lists or fields of the same datatype.") from e

    hashed_vals_list = [HASH_FUNC(prep_for_hash(i)) for i in sorted_vals]

    hashed_vals = hashed_vals_list[0]
    for val in hashed_vals_list[1:]:
        hashed_vals.update(val.digest())

    return hashed_vals.digest()


def process_column_for_hashing(column_name: str, schema_type: pl.DataType) -> plx.Expr:
    """Process a column for hashing based on its type.

    Args:
        column_name: The column name
        schema_type: The polars schema type of the column

    Returns:
        A polars expression for processing the column
    """
    if isinstance(schema_type, pl.Binary):
        return (
            pl.col(column_name).fill_null("\x00").bin.encode("hex").alias(column_name)
        )
    elif isinstance(schema_type, pl.Struct):
        return (
            pl.col(column_name)
            .struct.json_encode()
            .fill_null("\x00")
            .alias(column_name)
        )
    elif isinstance(schema_type, pl.Object):
        return (
            pl.col(column_name)
            .map_elements(lambda x: str(x), return_dtype=pl.Utf8)
            .fill_null("\x00")
            .alias(column_name)
        )
    else:
        return pl.col(column_name).cast(pl.Utf8).fill_null("\x00").alias(column_name)


def hash_rows(
    df: pl.DataFrame, columns: list[str], method: HashMethod = HashMethod.XXH3_128
) -> pl.Series:
    """Hash all rows in a dataframe.

    Args:
        df: The DataFrame to hash rows from
        columns: The column names to include in the hash
        method: The hash method to use

    Returns:
        List of row hashes as bytes
    """
    expr_list = [
        process_column_for_hashing(column, df.schema[column]) for column in columns
    ]
    df_processed = df.with_columns(expr_list)

    if method == HashMethod.XXH3_128:
        row_hashes = df_processed.select(
            plh.concat_str(*columns, separator="␞").nchash.xxh3_128().alias("row_hash")
        )
        return row_hashes["row_hash"]
    elif method == HashMethod.SHA256:
        row_hashes = df_processed.select(
            plh.concat_str(*columns, separator="␞")
            .chash.sha2_256()
            .str.decode("hex")
            .alias("row_hash")
        )
        return row_hashes["row_hash"]
    else:
        raise ValueError(f"Unsupported hash method: {method}")


def hash_arrow_table(
    table: pa.Table,
    method: HashMethod = HashMethod.XXH3_128,
) -> bytes:
    """Computes a content hash of an Arrow table invariant to row and field order.

    This is used to content-address an Arrow table for caching.

    Args:
        table: The pyarrow Table to hash
        method: The method to use for hashing rows (XXH3_128 or SHA256)

    Returns:
        Bytes representing the content hash of the table
    """
    df = pl.from_arrow(table)

    if df.height == 0:
        return b"empty_table_hash"

    columns: list[str] = sorted(df.columns)
    df = df.select(columns)

    # Explode list fields
    for column in columns:
        if isinstance(df.schema[column], pl.List):
            df = df.explode(column)

    df = df.sort(by=columns)

    row_hashes = hash_rows(df=df, columns=columns, method=method)

    all_hashes: bytes = b"".join(row_hashes.sort().to_list())

    return HASH_FUNC(all_hashes).digest()


def fields_to_value_ordered_hash(data: DataFrame, fields: list[str]) -> Series:
    """Returns the rowwise hash ordered by the row's values, ignoring field order.

    This function is used to add a field to a dataframe that represents the
    hash of each its rows, but where the order of the row values doesn't change the
    hash value. field order is ignored in favour of value order.

    This is primarily used to give a consistent hash to a new cluster no matter whether
    its parent hashes were used in the left or right table.
    """
    bytes_records = data.filter(fields).astype(bytes).to_dict("records")

    hashed_records = []

    for record in bytes_records:
        hashed_vals = hash_values(*record.values())
        hashed_records.append(hashed_vals)

    return Series(hashed_records)


class IntMap:
    """A data structure to map integers without collisions within a dedicated space.

    A stand-in for hashing integers within pa.int64.

    Takes unordered sets of integers, and maps them a to an ID that
    1) is a negative integer; 2) does not collide with other IDs generated by other
    instances of this class, as long as they are initialised with a different salt.

    The fact that IDs are always negative means that it's possible to build a hierarchy
    where IDs are themselves parts of other sets, and it's easy to distinguish integers
    mapped to raw data points (which will be non-negative), to integers that are IDs
    (which will be negative). The salt allows to work with a parallel execution
    model, where each worker maintains their separate ID space, as long as each worker
    operates on disjoint subsets of positive integers.

    Args:
        salt (optional): A positive integer to salt the Cantor pairing function
    """

    def __init__(self, salt: int = 42):
        """Initialise the IntMap."""
        self.mapping: dict[frozenset[int], int] = {}
        if salt < 0:
            raise ValueError("The salt must be a positive integer")
        self.salt: int = salt

    def _salt_id(self, i: int) -> int:
        """Use Cantor pairing function on the salt and a negative int ID.

        It negates the Cantor pairing function to always return a negative integer.
        """
        if i >= 0:
            raise ValueError("ID must be a negative integer")
        return -int(0.5 * (self.salt - i) * (self.salt - i + 1) - i)

    def index(self, *values: int) -> int:
        """Index a set of integers.

        Args:
            values: the integers in the set you want to index

        Returns:
            The old or new ID corresponding to the set
        """
        value_set = frozenset(values)
        if value_set in self.mapping:
            return self.mapping[value_set]

        new_id: int = -len(self.mapping) - 1
        salted_id = self._salt_id(new_id)
        self.mapping[value_set] = salted_id

        return salted_id

    def has_mapping(self, *values: int) -> bool:
        """Check if index for values already exists.

        Args:
            values: the integers in the set you want to index

        Returns:
            Boolean indicating whether index for values already exists
        """
        value_set = frozenset(values)
        return value_set in self.mapping


class Cluster:
    """A cluster of connected components.

    Can be a leaf node (a single data point) or a non-leaf node (a cluster of
    clusters). The hash of a cluster is the hash of its leaves.

    We generate negative integers for IDs, allowing us to generate a true ID with
    the database after we've calculated components using these objects.
    """

    id: int
    probability: int | None
    hash: bytes
    leaves: tuple["Cluster"] | None
    flag: Flag | None = None

    _intmap: IntMap  # Reference to the IntMap singleton

    def __init__(
        self,
        intmap: IntMap,
        probability: int | None = None,
        leaves: tuple["Cluster"] | list["Cluster"] | None = None,
        id: int | None = None,
        hash: bytes | None = None,
        flag: Flag | None = None,
    ):
        """Initialise the Cluster.

        Args:
            intmap: An IntMap instance for generating unique IDs
            probability: probability of the cluster from its resolution, or None if
        source
            leaves: A list of Cluster objects that are the leaves of this cluster
            id: The ID of the cluster (only for leaf nodes)
            hash: The hash of the cluster (only for leaf nodes)
            flag: A flag indicating the role of the cluster (optional). This field can
                be used to attach additional information to the cluster that's
                particular to the hierarchical representation the Cluster is
                involved in forming. For example, in a closure table system
                we might flag pair clusters.
        """
        self._intmap = intmap
        self.flag = flag
        self.probability = probability

        if leaves:
            self.leaves = tuple(sorted(leaves, key=lambda leaf: leaf.hash))
        else:
            self.leaves = None

        # Set hash - use provided hash or calculate
        if hash is not None:
            self.hash = hash
        elif self.leaves is None:
            raise ValueError("Leaf nodes must have hash specified")
        else:
            self.hash = HASH_FUNC(b"".join(leaf.hash for leaf in self.leaves)).digest()

        # Set ID - use provided ID or calculate
        if id is not None:
            self.id = id
        elif self.leaves is None:
            raise ValueError("Leaf nodes must have id specified")
        else:
            self.id = intmap.index(leaf.id for leaf in self.leaves)

    def add_flag(self, flag: Flag) -> None:
        """Add a flag to the cluster.

        Args:
            flag: The flag to add
        """
        if self.flag is None:
            self.flag = flag
        else:
            self.flag |= flag

    @classmethod
    def combine(
        cls: type[Self],
        clusters: Iterable["Cluster"],
        intmap: IntMap | None = None,
        probability: int | None = None,
        flag: Flag | None = None,
    ) -> "Cluster":
        """Efficiently combine multiple clusters at once.

        Args:
            clusters: An iterable of Cluster objects to combine
            intmap: An IntMap instance for generating unique IDs. Defaults to the
                first cluster's IntMap if not provided.
            probability: the probability of the cluster from its resolution
            flag: A flag indicating the role of the cluster

        Returns:
            A new Cluster containing all unique leaves from the input clusters
        """
        if not clusters:
            return None

        clusters = list(clusters)
        if len(clusters) == 1:
            return clusters[0]

        intmap = intmap if intmap is not None else clusters[0]._intmap
        unique_dict: dict[int, Cluster] = {}

        for cluster in clusters:
            if cluster.leaves is None:
                unique_dict[id(cluster)] = cluster
            else:
                for leaf in cluster.leaves:
                    unique_dict[id(leaf)] = leaf

        return cls(
            intmap=intmap,
            probability=probability,
            leaves=list(unique_dict.values()),
            flag=flag,
        )
