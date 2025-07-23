"""SourceConfig API routes for the Matchbox server."""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from matchbox.common.dtos import (
    BackendRetrievableType,
    NotFoundError,
    ResolutionName,
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
    authorisation_dependencies,
)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
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
async def get_source_config(
    backend: BackendDependency,
    name: SourceResolutionName,
) -> SourceConfig:
    """Get a source from the backend."""
    try:
        return backend.get_source_config(name=name)
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
async def get_resolution_source_configs(
    backend: BackendDependency,
    name: ResolutionName,
) -> list[SourceConfig]:
    """Get all sources in scope for a resolution."""
    try:
        return backend.get_resolution_source_configs(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        ) from e
