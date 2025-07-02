"""Common Arrow utilities."""

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
        ("parent", pa.uint64()),
        ("child", pa.uint64()),
    ]
)


def table_to_buffer(table: pa.Table) -> BytesIO:
    """Converts an Arrow table to a BytesIO buffer."""
    sink = BytesIO()
    pq.write_table(table, sink)
    sink.seek(0)
    return sink
