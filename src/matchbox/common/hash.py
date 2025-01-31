import base64
import hashlib
from typing import TypeVar
from uuid import UUID

from pandas import DataFrame, Series

T = TypeVar("T")
HashableItem = TypeVar("HashableItem", bytes, bool, str, int, float, bytearray)

HASH_FUNC = hashlib.sha256


def hash_to_base64(hash: bytes) -> str:
    return base64.b64encode(hash).decode("utf-8")


def base64_to_hash(b64: str) -> bytes:
    return base64.b64decode(b64)


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
        raise TypeError("Can only order lists or columns of the same datatype.") from e

    hashed_vals_list = [HASH_FUNC(prep_for_hash(i)) for i in sorted_vals]

    hashed_vals = hashed_vals_list[0]
    for val in hashed_vals_list[1:]:
        hashed_vals.update(val.digest())

    return hashed_vals.digest()


def columns_to_value_ordered_hash(data: DataFrame, columns: list[str]) -> Series:
    """Returns the rowwise hash ordered by the row's values, ignoring column order.

    This function is used to add a column to a dataframe that represents the
    hash of each its rows, but where the order of the row values doesn't change the
    hash value. Column order is ignored in favour of value order.

    This is primarily used to give a consistent hash to a new cluster no matter whether
    its parent hashes were used in the left or right table.
    """
    bytes_records = data.filter(columns).astype(bytes).to_dict("records")

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
        self.mapping: dict[frozenset[int], int] = {}
        if salt < 0:
            raise ValueError("The salt must be a positive integer")
        self.salt: int = salt

    def _salt_id(self, i: int) -> int:
        """Use Cantor pairing function on the salt and a negative int ID.

        It negates the Cantor pairing function to always return a negative integer."""
        if i >= 0:
            raise ValueError("ID must be a negative integer")
        return -int(0.5 * (self.salt - i) * (self.salt - i + 1) - i)

    def index(self, *values: int) -> int:
        """Args:
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
        """Args:
            values: the integers in the set you want to index

        Returns:
            Boolean indicating whether index for values already exists
        """
        value_set = frozenset(values)
        return value_set in self.mapping
