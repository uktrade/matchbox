"""API routes for the Matchbox server."""

from collections.abc import Awaitable, Callable
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import HTMLResponse

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendCountableType,
    BackendResourceType,
    CollectionName,
    CountResult,
    CRUDOperation,
    Match,
    ModelResolutionName,
    NotFoundError,
    OKMessage,
    ResolutionPath,
    ResourceOperationStatus,
    RunID,
    SourceResolutionName,
    SourceResolutionPath,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxRunNotFoundError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    authorisation_dependencies,
    lifespan,
)
from matchbox.server.api.routers import auth, collection

app = FastAPI(
    title="matchbox API",
    version=version("matchbox_db"),
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
app.include_router(collection.router)
app.include_router(auth.router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Overwrite the default JSON schema for an `HTTPException`."""
    return JSONResponse(
        content=jsonable_encoder(exc.detail), status_code=exc.status_code
    )


@app.exception_handler(MatchboxCollectionNotFoundError)
async def collection_not_found_handler(
    request: Request, exc: MatchboxCollectionNotFoundError
) -> JSONResponse:
    """Handle collection not found errors."""
    detail = NotFoundError(
        details=str(exc), entity=BackendResourceType.COLLECTION
    ).model_dump()
    return JSONResponse(
        status_code=404,
        content=jsonable_encoder(detail),
    )


@app.exception_handler(MatchboxRunNotFoundError)
async def run_not_found_handler(
    request: Request, exc: MatchboxRunNotFoundError
) -> JSONResponse:
    """Handle run not found errors."""
    detail = NotFoundError(
        details=str(exc), entity=BackendResourceType.RUN
    ).model_dump()
    return JSONResponse(
        status_code=404,
        content=jsonable_encoder(detail),
    )


@app.exception_handler(MatchboxResolutionNotFoundError)
async def resolution_not_found_handler(
    request: Request, exc: MatchboxResolutionNotFoundError
) -> JSONResponse:
    """Handle resolution not found errors."""
    detail = NotFoundError(
        details=str(exc), entity=BackendResourceType.RESOLUTION
    ).model_dump()
    return JSONResponse(
        status_code=404,
        content=jsonable_encoder(detail),
    )


@app.exception_handler(MatchboxDeletionNotConfirmed)
async def deletion_not_confirmed_handler(
    request: Request, exc: MatchboxDeletionNotConfirmed
) -> JSONResponse:
    """Handle deletion not confirmed errors."""
    # Extract resource name from request
    path_parts = request.url.path.split("/")
    resource_name = path_parts[-1] if path_parts else "Unknown"

    detail = ResourceOperationStatus(
        success=False,
        target=resource_name,
        operation=CRUDOperation.DELETE,
        details=str(exc),
    ).model_dump()

    return JSONResponse(
        status_code=409,
        content=jsonable_encoder(detail),
    )


@app.middleware("http")
async def add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
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


# Retrieval


@app.get(
    "/query",
    responses={404: {"model": NotFoundError}},
)
def query(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    source: SourceResolutionName,
    return_leaf_id: bool,
    resolution: ModelResolutionName | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> ParquetResponse:
    """Query Matchbox for matches based on a source resolution name."""
    try:
        res = backend.query(
            source=SourceResolutionPath(
                collection=collection,
                run=run_id,
                name=source,
            ),
            point_of_truth=ResolutionPath(
                collection=collection, run=run_id, name=resolution
            )
            if resolution
            else None,
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
    collection: CollectionName,
    run_id: RunID,
    targets: Annotated[list[SourceResolutionName], Query()],
    source: SourceResolutionName,
    key: str,
    resolution: ModelResolutionName,
    threshold: int | None = None,
) -> list[Match]:
    """Match a source key against a list of target source resolutions."""
    targets = [
        SourceResolutionPath(collection=collection, run=run_id, name=t) for t in targets
    ]
    try:
        res = backend.match(
            key=key,
            source=SourceResolutionPath(
                collection=collection,
                run=run_id,
                name=source,
            ),
            targets=targets,
            point_of_truth=ResolutionPath(
                collection=collection,
                run=run_id,
                name=resolution,
            ),
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
    "/database/orphans",
    responses={
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
)
def delete_orphans(backend: BackendDependency) -> ResourceOperationStatus:
    """Delete orphans."""
    try:
        orphans = backend.delete_orphans()
        return ResourceOperationStatus(
            success=True,
            target="Database",
            operation=CRUDOperation.DELETE,
            details=f"Deleted {orphans} orphans.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                target="Database",
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


@app.delete(
    "/database",
    responses={
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
        },
    },
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
) -> ResourceOperationStatus:
    """Delete all data from the backend whilst retaining tables."""
    try:
        backend.clear(certain=certain)
        return ResourceOperationStatus(
            success=True, target="Database", operation=CRUDOperation.DELETE
        )
    except MatchboxDeletionNotConfirmed:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                target="Database",
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


# Swagger UI


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
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
async def swagger_ui_redirect() -> HTMLResponse:
    """Helper for OAuth2."""
    return get_swagger_ui_oauth2_redirect_html()
