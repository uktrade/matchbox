"""Common Arrow utilities."""

from enum import StrEnum
from io import BytesIO
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_MB_IDS: Final[pa.Schema] = pa.schema(
    [("id", pa.int64()), ("key", pa.large_string())]
)
"""Data transfer schema for Matchbox IDs keyed to primary keys."""

SCHEMA_INDEX: Final[pa.Schema] = pa.schema(
    [("hash", pa.large_binary()), ("keys", pa.large_list(pa.large_string()))]
)
"""Data transfer schema for data to be indexed in Matchbox."""

SCHEMA_RESULTS: Final[pa.Schema] = pa.schema(
    [
        ("left_id", pa.uint64()),
        ("right_id", pa.uint64()),
        ("probability", pa.uint8()),
    ]
)
"""Data transfer schema for the results of a deduplication or linking process."""


SCHEMA_JUDGEMENTS: Final[pa.Schema] = pa.schema(
    [
        ("user_id", pa.uint64()),
        ("endorsed", pa.uint64()),
        ("shown", pa.uint64()),
    ]
)
"""Data transfer schema for retrieved evaluation judgements from users."""

SCHEMA_CLUSTER_EXPANSION: Final[pa.Schema] = pa.schema(
    [
        ("root", pa.uint64()),
        ("leaves", pa.list_(pa.uint64())),
    ]
)
"""Data transfer schema for mapping from a cluster ID to all its source cluster IDs."""

SCHEMA_EVAL_SAMPLES: Final[pa.Schema] = pa.schema(
    [
        ("root", pa.uint64()),
        ("leaf", pa.uint64()),
        ("key", pa.large_string()),
        ("source", pa.large_string()),
    ]
)


class JudgementsZipFilenames(StrEnum):
    """Enumeration of file names in ZIP file with downloaded judgements."""

    JUDGEMENTS = "judgements.parquet"
    EXPANSION = "expansion.parquet"


def table_to_buffer(table: pa.Table) -> BytesIO:
    """Converts an Arrow table to a BytesIO buffer."""
    sink = BytesIO()
    pq.write_table(table, sink)
    sink.seek(0)
    return sink
