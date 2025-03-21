"""Common Arrow utilities."""

from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_MB_IDS = pa.schema([("id", pa.int64()), ("source_pk", pa.large_string())])
SCHEMA_INDEX = pa.schema(
    [("hash", pa.large_binary()), ("source_pk", pa.large_list(pa.large_string()))]
)
SCHEMA_RESULTS = pa.schema(
    [
        ("left_id", pa.uint64()),
        ("right_id", pa.uint64()),
        ("probability", pa.uint8()),
    ]
)


def table_to_buffer(table: pa.Table) -> BytesIO:
    """Converts an Arrow table to a BytesIO buffer."""
    sink = BytesIO()
    pq.write_table(table, sink)
    sink.seek(0)
    return sink
