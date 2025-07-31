"""Model API routes for the Matchbox server."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    status,
)

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    ModelAncestor,
    ModelConfig,
    NotFoundError,
    ResolutionOperationStatus,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
)
from matchbox.common.graph import ModelResolutionName
from matchbox.server.api.dependencies import (
    BackendDependency,
    MetadataStoreDependency,
    ParquetResponse,
    authorisation_dependencies,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.post(
    "",
    responses={
        500: {
            "model": ResolutionOperationStatus,
            **ResolutionOperationStatus.status_500_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
)
async def insert_model(
    backend: BackendDependency, model: ModelConfig
) -> ResolutionOperationStatus:
    """Insert a model into the backend."""
    try:
        backend.insert_model(model)
        return ResolutionOperationStatus(
            success=True,
            name=model.name,
            operation=CRUDOperation.CREATE,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResolutionOperationStatus(
                success=False,
                name=model.name,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{name}",
    responses={404: {"model": NotFoundError}},
)
async def get_model(
    backend: BackendDependency, name: ModelResolutionName
) -> ModelConfig:
    """Get a model from the backend."""
    try:
        return backend.get_model(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e


@router.post(
    "/{name}/results",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
)
async def set_results(
    backend: BackendDependency,
    metadata_store: MetadataStoreDependency,
    name: ModelResolutionName,
) -> UploadStatus:
    """Create an upload task for model results."""
    try:
        metadata = backend.get_model(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e

    upload_id = metadata_store.cache_model(metadata=metadata)
    return metadata_store.get(cache_id=upload_id).status


@router.get(
    "/{name}/results",
    responses={404: {"model": NotFoundError}},
)
async def get_results(
    backend: BackendDependency, name: ModelResolutionName
) -> ParquetResponse:
    """Download results for a model as a parquet file."""
    try:
        res = backend.get_model_results(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@router.patch(
    "/{name}/truth",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResolutionOperationStatus,
            **ResolutionOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
)
async def set_truth(
    backend: BackendDependency,
    name: ModelResolutionName,
    truth: Annotated[int, Body(ge=0, le=100)],
) -> ResolutionOperationStatus:
    """Set truth data for a model."""
    try:
        backend.set_model_truth(name=name, truth=truth)
        return ResolutionOperationStatus(
            success=True,
            name=name,
            operation=CRUDOperation.UPDATE,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResolutionOperationStatus(
                success=False,
                name=name,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{name}/truth",
    responses={404: {"model": NotFoundError}},
)
async def get_truth(backend: BackendDependency, name: ModelResolutionName) -> float:
    """Get truth data for a model."""
    try:
        return backend.get_model_truth(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e


@router.get(
    "/{name}/ancestors",
    responses={404: {"model": NotFoundError}},
)
async def get_ancestors(
    backend: BackendDependency, name: ModelResolutionName
) -> list[ModelAncestor]:
    """Get the ancestors for a model."""
    try:
        return backend.get_model_ancestors(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e


@router.patch(
    "/{name}/ancestors_cache",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResolutionOperationStatus,
            **ResolutionOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
)
async def set_ancestors_cache(
    backend: BackendDependency,
    name: ModelResolutionName,
    ancestors: list[ModelAncestor],
):
    """Update the cached ancestors for a model."""
    try:
        backend.set_model_ancestors_cache(name=name, ancestors_cache=ancestors)
        return ResolutionOperationStatus(
            success=True,
            name=name,
            operation=CRUDOperation.UPDATE,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResolutionOperationStatus(
                success=False,
                name=name,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{name}/ancestors_cache",
    responses={404: {"model": NotFoundError}},
)
async def get_ancestors_cache(
    backend: BackendDependency, name: ModelResolutionName
) -> list[ModelAncestor]:
    """Get the cached ancestors for a model."""
    try:
        return backend.get_model_ancestors_cache(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e
