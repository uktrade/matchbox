"""API routes for the Matchbox server."""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendCountableType,
    BackendResourceType,
    BackendUploadType,
    CountResult,
    LoginAttempt,
    LoginResult,
    Match,
    NotFoundError,
    OKMessage,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
)
from matchbox.common.graph import ResolutionGraph, ResolutionName, SourceResolutionName
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    SettingsDependency,
    UploadTrackerDependency,
    authorisation_dependencies,
    lifespan,
)
from matchbox.server.api.routers import eval, resolution
from matchbox.server.uploads import process_upload, process_upload_celery, table_to_s3

app = FastAPI(
    title="matchbox API",
    version=version("matchbox_db"),
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.include_router(resolution.router)
app.include_router(eval.router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Overwrite the default JSON schema for an `HTTPException`."""
    return JSONResponse(
        content=jsonable_encoder(exc.detail), status_code=exc.status_code
    )


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Improve security by adding headers to all responses."""
    response: Response = await call_next(request)
    response.headers["Cache-control"] = "no-store, no-cache"
    response.headers["Content-Security-Policy"] = (
        # Restrict by default
        "default-src 'none'; frame-ancestors 'none'; form-action 'none';"
        # Load Swagger CSS, favicon and openapi.json
        "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
        # Load Swagger JS, hard-coding the expected file hash
        "script-src 'self' 'sha256-QOOQu4W1oxGqd2nbXbxiA1Di6OHQOLQD+o+G9oWL8YY='"
    )
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/health")
async def healthcheck() -> OKMessage:
    """Perform a health check and return the status."""
    return OKMessage()


@app.post(
    "/login",
)
def login(
    backend: BackendDependency,
    credentials: LoginAttempt,
) -> LoginResult:
    """Receives a user name and returns a user ID."""
    return LoginResult(user_id=backend.login(credentials.user_name))


@app.post(
    "/upload/{upload_id}",
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
    },
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
)
def upload_file(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    settings: SettingsDependency,
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
    upload_entry = upload_tracker.get(upload_id=upload_id)
    if not upload_entry:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                update_timestamp=datetime.now(),
                stage="unknown",
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
            ).model_dump(),
        )

    # Ensure tracker is expecting file
    if upload_entry.status.stage != UploadStage.AWAITING_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=upload_entry.status.model_dump(),
        )

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{upload_id}.parquet"

    try:
        table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=upload_entry.status.entity.schema,
        )
    except MatchboxServerFileError as e:
        upload_tracker.update(upload_id, UploadStage.FAILED, details=str(e))
        updated_entry = upload_tracker.get(upload_id)
        raise HTTPException(
            status_code=400,
            detail=updated_entry.status.model_dump(),
        ) from e

    upload_tracker.update(upload_id, UploadStage.QUEUED)

    # Start background processing
    match settings.task_runner:
        case "api":
            background_tasks.add_task(
                process_upload,
                backend=backend,
                tracker=upload_tracker,
                s3_client=client,
                upload_type=upload_entry.status.entity,
                resolution_name=upload_entry.metadata.name,
                upload_id=upload_id,
                bucket=bucket,
                filename=key,
            )
        case "celery":
            process_upload_celery.delay(
                upload_type=upload_entry.status.entity,
                resolution_name=upload_entry.metadata.name,
                upload_id=upload_id,
                bucket=bucket,
                filename=key,
            )
        case _:
            raise RuntimeError("Unsupported task runner.")

    source_upload = upload_tracker.get(upload_id)

    # Check for error in async task
    if source_upload.status.stage == UploadStage.FAILED:
        raise HTTPException(
            status_code=400,
            detail=source_upload.status.model_dump(),
        )
    else:
        return source_upload.status


@app.get(
    "/upload/{upload_id}/status",
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
    },
    status_code=status.HTTP_200_OK,
)
def get_upload_status(
    upload_tracker: UploadTrackerDependency,
    upload_id: str,
) -> UploadStatus:
    """Get the status of an upload process.

    Returns the current status of the upload and processing task.

    Raises HTTP 400 if:
    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    """
    source_upload = upload_tracker.get(upload_id=upload_id)
    if not source_upload:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                id=upload_id,
                stage="unknown",
                update_timestamp=datetime.now(),
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )

    return source_upload.status


# Retrieval


@app.get(
    "/query",
    responses={404: {"model": NotFoundError}},
)
def query(
    backend: BackendDependency,
    source: SourceResolutionName,
    return_leaf_id: bool,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> ParquetResponse:
    """Query Matchbox for matches based on a source resolution name."""
    try:
        res = backend.query(
            source=source,
            resolution=resolution,
            threshold=threshold,
            return_leaf_id=return_leaf_id,
            limit=limit,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


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
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e

    return res


# Admin


@app.get("/report/resolutions")
def get_resolutions(backend: BackendDependency) -> ResolutionGraph:
    """Get the resolution graph."""
    return backend.get_resolution_graph()


@app.get("/database/count")
def count_backend_items(
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
    dependencies=[Depends(authorisation_dependencies)],
)
def clear_database(
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


# Swagger UI


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Get locally hosted docs."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    """Helper for OAuth2."""
    return get_swagger_ui_oauth2_redirect_html()
