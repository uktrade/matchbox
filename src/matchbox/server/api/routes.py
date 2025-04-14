"""API routes for the Matchbox server."""

import logging
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, AsyncGenerator

from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    APISettings,
    BackendCountableType,
    BackendRetrievableType,
    BackendUploadType,
    CountResult,
    ModelAncestor,
    ModelMetadata,
    ModelOperationStatus,
    ModelOperationType,
    NotFoundError,
    OKMessage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.sources import Match, Source, SourceAddress
from matchbox.server.api.arrow import table_to_s3
from matchbox.server.api.cache import MetadataStore, process_upload
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)


class ParquetResponse(Response):
    """A response object for returning parquet data."""

    media_type = "application/octet-stream"


settings: MatchboxServerSettings | None = None
backend: MatchboxDBAdapter | None = None
metadata_store = MetadataStore(expiry_minutes=30)
api_key_header = APIKeyHeader(name="X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Context manager for the FastAPI lifespan events."""
    # Set up the backend
    global settings
    global backend

    SettingsClass = get_backend_settings(MatchboxServerSettings().backend_type)
    settings = SettingsClass()
    backend = settings_to_backend(settings)

    # Define common formatter
    formatter = logging.Formatter("[%(name)s %(levelname)s] %(message)s")

    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(backend.settings.log_level)
    handler.setFormatter(formatter)

    # Configure loggers with the same handler and formatter
    loggers_to_configure = [
        "matchbox",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "uvicorn.asgi",
        "fastapi",
    ]

    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(backend.settings.log_level)
        # Remove any existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        logger.addHandler(handler)

    # Set SQLAlchemy loggers
    for sql_logger in ["sqlalchemy", "sqlalchemy.engine"]:
        logging.getLogger(sql_logger).setLevel("WARNING")

    yield

    del settings
    del backend


app = FastAPI(
    title="matchbox API",
    version=version("matchbox_db"),
    lifespan=lifespan,
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Overwrite the default JSON schema for an `HTTPException`."""
    return JSONResponse(content=exc.detail, status_code=exc.status_code)


def validate_api_key(api_key: str = Security(api_key_header)) -> None:
    """Validate client API Key against settings."""
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing in server configuration.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required but not provided.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
    elif api_key != settings.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key invalid.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


# General


@app.get("/health")
async def healthcheck() -> OKMessage:
    """Perform a health check and return the status."""
    return OKMessage()


@app.get("/admin/settings", dependencies=[Depends(validate_api_key)])
async def get_user_settings() -> APISettings:
    """Get user-facing server settings."""
    return APISettings(
        batch_size=settings.batch_size,
        log_level=settings.log_level,
        log_sql=settings.log_sql,
    )


@app.post("/admin/settings", dependencies=[Depends(validate_api_key)])
async def update_settings(
    updated_settings: APISettings,
) -> APISettings:
    """Update user-facing server settings."""
    global settings

    if updated_settings.batch_size is not None:
        settings.batch_size = updated_settings.batch_size
    if updated_settings.log_level is not None:
        settings.log_level = updated_settings.log_level
    if updated_settings.log_sql is not None:
        settings.log_sql = updated_settings.log_sql

    return APISettings(
        batch_size=settings.batch_size,
        log_level=settings.log_level,
        log_sql=settings.log_sql,
    )


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


@app.get(
    "/query",
    responses={404: {"model": NotFoundError}},
)
def query(
    full_name: str,
    warehouse_hash_b64: str,
    resolution_name: str | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> ParquetResponse:
    """Query Matchbox for matches based on a source address."""
    source_address = SourceAddress(
        full_name=full_name, warehouse_hash=warehouse_hash_b64
    )
    try:
        res = backend.query(
            source_address=source_address,
            resolution_name=resolution_name,
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


@app.get(
    "/match",
    responses={404: {"model": NotFoundError}},
)
def match(
    target_full_names: Annotated[list[str], Query()],
    target_warehouse_hashes_b64: Annotated[list[str], Query()],
    source_full_name: str,
    source_warehouse_hash_b64: str,
    source_pk: str,
    resolution_name: str,
    threshold: int | None = None,
) -> list[Match]:
    """Match a source primary key against a list of target addresses."""
    targets = [
        SourceAddress(full_name=n, warehouse_hash=w)
        for n, w in zip(target_full_names, target_warehouse_hashes_b64, strict=True)
    ]
    source = SourceAddress(
        full_name=source_full_name, warehouse_hash=source_warehouse_hash_b64
    )
    try:
        res = backend.match(
            source_pk=source_pk,
            source=source,
            targets=targets,
            resolution_name=resolution_name,
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


# Data management


@app.post(
    "/sources",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(validate_api_key)],
)
async def add_source(source: Source) -> UploadStatus:
    """Create an upload and insert task for indexed source data."""
    upload_id = metadata_store.cache_source(metadata=source)
    return metadata_store.get(cache_id=upload_id).status


@app.get(
    "/sources/{warehouse_hash_b64}/{full_name}",
    responses={404: {"model": NotFoundError}},
)
async def get_source(
    warehouse_hash_b64: str,
    full_name: str,
) -> Source:
    """Get a source from the backend."""
    address = SourceAddress(full_name=full_name, warehouse_hash=warehouse_hash_b64)
    try:
        return backend.get_source(address)
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e


@app.get("/report/resolutions")
async def get_resolutions() -> ResolutionGraph:
    """Get the resolution graph."""
    return backend.get_resolution_graph()


# Model management


@app.post(
    "/models",
    responses={
        500: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_500_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(validate_api_key)],
)
async def insert_model(model: ModelMetadata) -> ModelOperationStatus:
    """Insert a model into the backend."""
    try:
        backend.insert_model(model)
        return ModelOperationStatus(
            success=True,
            model_name=model.name,
            operation=ModelOperationType.INSERT,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ModelOperationStatus(
                success=False,
                model_name=model.name,
                operation=ModelOperationType.INSERT,
                details=str(e),
            ).model_dump(),
        ) from e


@app.get(
    "/models/{name}",
    responses={404: {"model": NotFoundError}},
)
async def get_model(name: str) -> ModelMetadata:
    """Get a model from the backend."""
    try:
        return backend.get_model(model=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@app.post(
    "/models/{name}/results",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(validate_api_key)],
)
async def set_results(name: str) -> UploadStatus:
    """Create an upload task for model results."""
    try:
        metadata = backend.get_model(model=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e

    upload_id = metadata_store.cache_model(metadata=metadata)
    return metadata_store.get(cache_id=upload_id).status


@app.get(
    "/models/{name}/results",
    responses={404: {"model": NotFoundError}},
)
async def get_results(name: str) -> ParquetResponse:
    """Download results for a model as a parquet file."""
    try:
        res = backend.get_model_results(model=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@app.patch(
    "/models/{name}/truth",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(validate_api_key)],
)
async def set_truth(
    name: str,
    truth: Annotated[float, Body(ge=0.0, le=1.0)],
) -> ModelOperationStatus:
    """Set truth data for a model."""
    try:
        backend.set_model_truth(model=name, truth=truth)
        return ModelOperationStatus(
            success=True,
            model_name=name,
            operation=ModelOperationType.UPDATE_TRUTH,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ModelOperationStatus(
                success=False,
                model_name=name,
                operation=ModelOperationType.UPDATE_TRUTH,
                details=str(e),
            ).model_dump(),
        ) from e


@app.get(
    "/models/{name}/truth",
    responses={404: {"model": NotFoundError}},
)
async def get_truth(name: str) -> float:
    """Get truth data for a model."""
    try:
        return backend.get_model_truth(model=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@app.get(
    "/models/{name}/ancestors",
    responses={404: {"model": NotFoundError}},
)
async def get_ancestors(name: str) -> list[ModelAncestor]:
    """Get the ancestors for a model."""
    try:
        return backend.get_model_ancestors(model=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@app.patch(
    "/models/{name}/ancestors_cache",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(validate_api_key)],
)
async def set_ancestors_cache(
    name: str,
    ancestors: list[ModelAncestor],
):
    """Update the cached ancestors for a model."""
    try:
        backend.set_model_ancestors_cache(model=name, ancestors_cache=ancestors)
        return ModelOperationStatus(
            success=True,
            model_name=name,
            operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ModelOperationStatus(
                success=False,
                model_name=name,
                operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
                details=str(e),
            ).model_dump(),
        ) from e


@app.get(
    "/models/{name}/ancestors_cache",
    responses={404: {"model": NotFoundError}},
)
async def get_ancestors_cache(name: str) -> list[ModelAncestor]:
    """Get the cached ancestors for a model."""
    try:
        return backend.get_model_ancestors_cache(model=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e


@app.delete(
    "/models/{name}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ModelOperationStatus,
            **ModelOperationStatus.status_409_examples(),
        },
    },
    dependencies=[Depends(validate_api_key)],
)
async def delete_model(
    name: str,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the model")
    ] = False,
) -> ModelOperationStatus:
    """Delete a model from the backend."""
    try:
        backend.delete_model(model=name, certain=certain)
        return ModelOperationStatus(
            success=True,
            model_name=name,
            operation=ModelOperationType.DELETE,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxDeletionNotConfirmed as e:
        raise HTTPException(
            status_code=409,
            detail=ModelOperationStatus(
                success=False,
                model_name=name,
                operation=ModelOperationType.DELETE,
                details=str(e),
            ).model_dump(),
        ) from e


# Admin


@app.get("/database/count")
async def count_backend_items(
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
    certain: Annotated[
        bool, Query(description="Confirm deletion of all data in the database")
    ] = False,
) -> OKMessage:
    """Clear all data from the backend."""
    try:
        backend.clear(certain=certain)
        return OKMessage()
    except MatchboxDeletionNotConfirmed as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        ) from e
