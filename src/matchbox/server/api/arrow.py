"""Arrow utility functions for the Matchbox server."""

from typing import TYPE_CHECKING, Any, AsyncGenerator

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import UploadFile

from matchbox.common.exceptions import (
    MatchboxServerFileError,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


async def table_to_s3(
    client: S3Client,
    bucket: str,
    key: str,
    file: UploadFile,
    expected_schema: pa.Schema,
) -> str:
    """Upload a PyArrow Table to S3 and return the key.

    Args:
        client: The S3 client to use.
        bucket: The S3 bucket to upload to.
        key: The key to upload to.
        file: The file to upload.
        expected_schema: The schema that the file should match.

    Raises:
        MatchboxServerFileError: If the file is not a valid Parquet file or the schema
            does not match the expected schema.

    Returns:
        The key of the uploaded file.
    """
    try:
        table = pq.read_table(file.file)

        if not table.schema.equals(expected_schema):
            raise MatchboxServerFileError(
                message=(
                    "Schema mismatch. "
                    f"Expected:\n{expected_schema}\nGot:\n{table.schema}"
                )
            )

        await file.seek(0)

        client.put_object(Bucket=bucket, Key=key, Body=file.file)

    except Exception as e:
        if isinstance(e, MatchboxServerFileError):
            raise
        raise MatchboxServerFileError(message=f"Invalid Parquet file: {str(e)}") from e

    return key


async def s3_to_recordbatch(
    client: S3Client, bucket: str, key: str, batch_size: int = 1000
) -> AsyncGenerator[pa.RecordBatch, None]:
    """Download a PyArrow Table from S3 and stream it as RecordBatches."""
    response = client.get_object(Bucket=bucket, Key=key)
    buffer = pa.BufferReader(response["Body"].read())

    parquet_file = pq.ParquetFile(buffer)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch
