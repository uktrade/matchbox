from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_MB_IDS = pa.schema([("id", pa.int64()), ("source_pk", pa.string())])
SCHEMA_INDEX = pa.schema([("source_pk", pa.list_(pa.string())), ("hash", pa.binary())])
SCHEMA_RESULTS = pa.schema(
    [
        ("left_id", pa.uint64()),
        ("right_id", pa.uint64()),
        ("probability", pa.uint8()),
    ]
)


def table_to_buffer(table: pa.Table) -> BytesIO:
    sink = BytesIO()
    pq.write_table(table, sink)
    sink.seek(0)
    return sink
