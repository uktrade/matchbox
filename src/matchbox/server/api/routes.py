from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any, AsyncGenerator

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendCountableType,
    BackendRetrievableType,
    BackendUploadType,
    CountResult,
    HealthCheck,
    ModelResultsType,
    NotFoundError,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import base64_to_hash
from matchbox.common.sources import Source, SourceAddress
from matchbox.server.api.cache import MetadataStore
from matchbox.server.base import BackendManager, MatchboxDBAdapter

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class ParquetResponse(Response):
    media_type = "application/octet-stream"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    get_backend()
    yield


metadata_store = MetadataStore(expiry_minutes=30)

app = FastAPI(
    title="matchbox API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Overwrite the default JSON schema for an `HTTPException`"""
    return JSONResponse(content=exc.detail, status_code=exc.status_code)


def get_backend() -> MatchboxDBAdapter:
    return BackendManager.get_backend()


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


@app.get("/health")
async def healthcheck() -> HealthCheck:
    """Perform a health check and return the status."""
    return HealthCheck(status="OK")


@app.get("/testing/count")
async def count_backend_items(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    entity: BackendCountableType | None = None,
) -> CountResult:
    def get_count(e: BackendCountableType) -> int:
        return getattr(backend, str(e)).count()

    if entity is not None:
        return CountResult(entities={str(entity): get_count(entity)})
    else:
        res = {str(e): get_count(e) for e in BackendCountableType}
        return CountResult(entities=res)


@app.post("/testing/clear")
async def clear_backend():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/sources")
async def add_source(source: Source):
    """Add a source to the backend."""
    upload_id = metadata_store.cache_source(metadata=source)
    return UploadStatus(
        id=upload_id, status="awaiting_upload", entity=BackendUploadType.INDEX
    )


@app.post("/upload/{upload_id}")
async def upload_file(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    upload_id: str,
    file: UploadFile,
):
    """Upload file and process based on metadata type"""
    source = metadata_store.get(cache_id=upload_id)
    if not source:
        raise HTTPException(status_code=404, detail="Upload ID not found or expired")

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{upload_id}.parquet"
    try:
        upload_id = await table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=source.upload_schema.value,
        )
    except MatchboxServerFileError as e:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                status="failed",
                details=f"{str(e)}",
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        ) from e

    # Read from S3
    data_hashes = pa.Table.from_batches(
        [
            batch
            async for batch in s3_to_recordbatch(client=client, bucket=bucket, key=key)
        ]
    )

    # Index
    backend.index(source=source, data_hashes=data_hashes)

    # Clean up
    metadata_store.remove(upload_id)

    return UploadStatus(id=upload_id, status="complete", entity=BackendUploadType.INDEX)


@app.get("/sources")
async def list_sources():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/sources/{address}")
async def get_source(address: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models")
async def list_models():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/resolution/{name}")
async def get_resolution(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}")
async def add_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.delete("/models/{name}")
async def delete_model(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/results")
async def get_results(name: str, result_type: ModelResultsType | None):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/results")
async def set_results(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/truth")
async def get_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/truth")
async def set_truth(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors")
async def get_ancestors(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/models/{name}/ancestors_cache")
async def get_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.post("/models/{name}/ancestors_cache")
async def set_ancestors_cache(name: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get(
    "/query",
    response_class=ParquetResponse,
    responses={404: NotFoundError.example_response_body()},
)
async def query(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
    full_name: str,
    warehouse_hash_b64: str,
    resolution_id: int | None = None,
    threshold: float | None = None,
    limit: int | None = None,
):
    warehouse_hash = base64_to_hash(warehouse_hash_b64)
    source_address = SourceAddress(full_name=full_name, warehouse_hash=warehouse_hash)
    try:
        res = backend.query(
            source_address=source_address,
            resolution_id=resolution_id,
            threshold=threshold,
            limit=limit,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=f"{str(e)}", entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=f"{str(e)}", entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@app.get("/match")
async def match():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/validate/hash")
async def validate_hashes():
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/report/resolutions")
async def get_resolutions(
    backend: Annotated[MatchboxDBAdapter, Depends(get_backend)],
) -> ResolutionGraph:
    return backend.get_resolution_graph()
