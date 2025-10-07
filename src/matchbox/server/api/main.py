"""API routes for the Matchbox server."""

from importlib.metadata import version
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
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendCountableType,
    BackendResourceType,
    CountResult,
    LoginAttempt,
    LoginResult,
    Match,
    NotFoundError,
    OKMessage,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.graph import ResolutionGraph, ResolutionName, SourceResolutionName
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    authorisation_dependencies,
    lifespan,
)
from matchbox.server.api.routers import eval, resolution

app = FastAPI(
    title="matchbox API",
    version=version("matchbox_db"),
    lifespan=lifespan,
)
app.include_router(resolution.router)
app.include_router(eval.router)


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
        "default-src 'none'; frame-ancestors 'none'; form-action 'none'; sandbox"
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
