"""SourceConfig API routes for the Matchbox server."""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from matchbox.common.dtos import (
    BackendRetrievableType,
    ModelResolutionName,
    NotFoundError,
    SourceResolutionName,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.sources import SourceConfig
from matchbox.server.api.dependencies import (
    BackendDependency,
    MetadataStoreDependency,
    validate_api_key,
)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(validate_api_key)],
)
async def add_source(
    metadata_store: MetadataStoreDependency, source: SourceConfig
) -> UploadStatus:
    """Create an upload and insert task for indexed source data."""
    upload_id = metadata_store.cache_source(metadata=source)
    return metadata_store.get(cache_id=upload_id).status


@router.get(
    "/{name}",
    responses={404: {"model": NotFoundError}},
)
async def get_source(
    backend: BackendDependency,
    name: SourceResolutionName,
) -> SourceConfig:
    """Get a source from the backend."""
    try:
        return backend.get_source(name=name)
    except MatchboxSourceNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.SOURCE
            ).model_dump(),
        ) from e


@router.get(
    "",
    responses={404: {"model": NotFoundError}},
)
async def get_resolution_sources(
    backend: BackendDependency,
    name: ModelResolutionName,
) -> list[SourceConfig]:
    """Get all sources in scope for a resolution."""
    try:
        return backend.get_resolution_sources(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
