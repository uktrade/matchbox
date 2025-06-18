"""API routes for the Matchbox server."""

from importlib.metadata import version
from typing import Annotated, Any, Awaitable

import anyio
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendCountableType,
    BackendRetrievableType,
    BackendUploadType,
    CountResult,
    NotFoundError,
    OKMessage,
    ResolutionName,
    SourceResolutionName,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.logging import logger
from matchbox.common.sources import Match
from matchbox.server.api.arrow import table_to_s3
from matchbox.server.api.cache import process_upload
from matchbox.server.api.dependencies import (
    BackendDependency,
    MetadataStoreDependency,
    ParquetResponse,
    lifespan,
    validate_api_key,
)
from matchbox.server.api.routers import models, resolutions, sources

app = FastAPI(
    title="matchbox API",
    version=version("matchbox_db"),
    lifespan=lifespan,
)
app.include_router(models.router)
app.include_router(sources.router)
app.include_router(resolutions.router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Overwrite the default JSON schema for an `HTTPException`."""
    return JSONResponse(content=exc.detail, status_code=exc.status_code)


# General


@app.get("/health")
async def healthcheck() -> OKMessage:
    """Perform a health check and return the status."""
    return OKMessage()


@app.post(
    "/upload/{upload_id}",
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
    },
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(validate_api_key)],
)
async def upload_file(
    background_tasks: BackgroundTasks,
    backend: BackendDependency,
    metadata_store: MetadataStoreDependency,
    upload_id: str,
    file: UploadFile,
) -> UploadStatus:
    """Upload and process a file based on metadata type.

    The file is uploaded to S3 and then processed in a background task.
    Status can be checked using the /upload/{upload_id}/status endpoint.

    Raises HTTP 400 if:

    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    * Upload is already being processed
    * Uploaded data doesn't match the metadata schema
    """
    # Get and validate cache entry
    source_cache = metadata_store.get(cache_id=upload_id)
    if not source_cache:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                status="failed",
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
            ).model_dump(),
        )

    # Check if already processing
    if source_cache.status.status != "awaiting_upload":
        raise HTTPException(
            status_code=400,
            detail=source_cache.status.model_dump(),
        )

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{upload_id}.parquet"

    try:
        await table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=source_cache.upload_type.schema,
        )
    except MatchboxServerFileError as e:
        metadata_store.update_status(upload_id, "failed", details=str(e))
        raise HTTPException(
            status_code=400,
            detail=source_cache.status.model_dump(),
        ) from e

    metadata_store.update_status(upload_id, "queued")

    # Start background processing
    background_tasks.add_task(
        process_upload,
        backend=backend,
        upload_id=upload_id,
        bucket=bucket,
        key=key,
        metadata_store=metadata_store,
    )

    source_cache = metadata_store.get(upload_id)

    # Check for error in async task
    if source_cache.status.status == "failed":
        raise HTTPException(
            status_code=400,
            detail=source_cache.status.model_dump(),
        )
    else:
        return source_cache.status


@app.get(
    "/upload/{upload_id}/status",
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
    },
    status_code=status.HTTP_200_OK,
)
async def get_upload_status(
    metadata_store: MetadataStoreDependency,
    upload_id: str,
) -> UploadStatus:
    """Get the status of an upload process.

    Returns the current status of the upload and processing task.

    Raises HTTP 400 if:
    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    """
    source_cache = metadata_store.get(cache_id=upload_id)
    if not source_cache:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                status="failed",
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )

    return source_cache.status


# Retrieval


async def disconnected(request: Request) -> None:
    """Cancel task on client KeyboardInterrupt."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            logger.info("Query disrupted by client disconnection.")
            break


async def wrap(call: Awaitable[Any], cancel_scope: anyio.CancelScope):
    """Wrapper for calling async endpoint functions."""
    res = await call
    cancel_scope.cancel()
    return res


@app.get(
    "/query",
    responses={404: {"model": NotFoundError}},
)
async def query(
    request: Request,
    backend: BackendDependency,
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> ParquetResponse:
    """Query Matchbox for matches based on a source resolution name."""

    async def _query():
        try:
            res = backend.query(
                source=source,
                resolution=resolution,
                threshold=threshold,
                limit=limit,
            )
        except MatchboxResolutionNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=NotFoundError(
                    details=str(e), entity=BackendRetrievableType.RESOLUTION
                ).model_dump(),
            ) from e
        except MatchboxSourceNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=NotFoundError(
                    details=str(e), entity=BackendRetrievableType.SOURCE
                ).model_dump(),
            ) from e

        buffer = table_to_buffer(res)
        return ParquetResponse(buffer.getvalue())

    async with anyio.create_task_group() as tg:
        tg.start_soon(wrap, disconnected(request), tg.cancel_scope)
        return await wrap(_query(), tg.cancel_scope)


@app.get(
    "/match",
    responses={404: {"model": NotFoundError}},
)
def match(
    backend: BackendDependency,
    targets: Annotated[list[SourceResolutionName], Query()],
    source: SourceResolutionName,
    key: str,
    resolution: ResolutionName,
    threshold: int | None = None,
) -> list[Match]:
    """Match a source key against a list of target source resolutions."""
    try:
        res = backend.match(
            key=key,
            source=source,
            targets=targets,
            resolution=resolution,
            threshold=threshold,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e

    return res


# Admin


@app.get("/report/resolutions")
async def get_resolutions(backend: BackendDependency) -> ResolutionGraph:
    """Get the resolution graph."""
    return backend.get_resolution_graph()


@app.get("/database/count")
async def count_backend_items(
    backend: BackendDependency,
    entity: BackendCountableType | None = None,
) -> CountResult:
    """Count the number of various entities in the backend."""

    def get_count(e: BackendCountableType) -> int:
        return getattr(backend, str(e)).count()

    if entity is not None:
        return CountResult(entities={str(entity): get_count(entity)})
    else:
        res = {str(e): get_count(e) for e in BackendCountableType}
        return CountResult(entities=res)


@app.delete(
    "/database",
    responses={409: {"model": str}},
    dependencies=[Depends(validate_api_key)],
)
async def clear_database(
    backend: BackendDependency,
    certain: Annotated[
        bool,
        Query(
            description=(
                "Confirm deletion of all data in the database whilst retaining tables"
            )
        ),
    ] = False,
) -> OKMessage:
    """Delete all data from the backend whilst retaining tables."""
    try:
        backend.clear(certain=certain)
        return OKMessage()
    except MatchboxDeletionNotConfirmed as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        ) from e
