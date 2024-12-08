import hashlib
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from pandas import DataFrame, Series
from sqlalchemy import String, func, select
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from matchbox.common.db import Source
else:
    Source = Any

T = TypeVar("T")
HashableItem = TypeVar("HashableItem", bytes, bool, str, int, float, bytearray)

HASH_FUNC = hashlib.sha256


def dataset_to_hashlist(dataset: Source, model_hash: bytes) -> list[dict[str, Any]]:
    """Retrieve and hash a dataset from its warehouse, ready to be inserted."""
    with Session(dataset.database.engine) as warehouse_session:
        source_table = dataset.to_table()
        cols_to_index = tuple(
            [col.literal.name for col in dataset.db_columns if col.indexed]
        )

        slct_stmt = select(
            func.concat(*source_table.c[cols_to_index]).label("raw"),
            func.array_agg(source_table.c[dataset.db_pk].cast(String)).label("id"),
        ).group_by(*source_table.c[cols_to_index])

        raw_result = warehouse_session.execute(slct_stmt)

        to_insert = [
            {
                "hash": hash_data(data.raw),
                "dataset": model_hash,
                "id": data.id,
            }
            for data in raw_result.all()
        ]

    return to_insert


def prep_for_hash(item: HashableItem) -> bytes:
    """Encodes strings so they can be hashed, otherwises, passes through."""
    if isinstance(item, bytes):
        return item
    elif isinstance(item, str):
        return bytes(item.encode())
    elif isinstance(item, UUID):
        return item.bytes
    else:
        return bytes(item)


def hash_data(data: str) -> bytes:
    """
    Hash the given data using the globally defined hash function.
    This function ties into the existing hashing utilities.
    """
    return HASH_FUNC(prep_for_hash(data)).digest()


def list_to_value_ordered_hash(list_: list[T]) -> bytes:
    """Returns a single hash of a list ordered by its values.

    List must be sorted as the different orders of value must produce the same hash.
    """
    try:
        sorted_vals = sorted(list_)
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
        hashed_vals = list_to_value_ordered_hash(record.values())
        hashed_records.append(hashed_vals)

    return Series(hashed_records)
