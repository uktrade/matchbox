"""Resolution API routes for the Matchbox server."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    NotFoundError,
    Resolution,
    ResolutionOperationStatus,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.graph import ModelResolutionName, ResolutionName, ResolutionType
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    UploadTrackerDependency,
    authorisation_dependencies,
)

router = APIRouter(prefix="/resolutions", tags=["resolution"])


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
def create_resolution(
    backend: BackendDependency,
    resolution: Resolution,
) -> ResolutionOperationStatus | UploadStatus:
    """Create a resolution (model or source)."""
    try:
        backend.insert_resolution(resolution=resolution)
        return ResolutionOperationStatus(
            success=True,
            name=resolution.name,
            operation=CRUDOperation.CREATE,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResolutionOperationStatus(
                success=False,
                name=resolution.name,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{name}",
    responses={404: {"model": NotFoundError}},
)
def get_resolution(
    backend: BackendDependency,
    name: ResolutionName,
    validate_type: ResolutionType | None = None,
) -> Resolution:
    """Get a resolution (model or source) from the backend."""
    try:
        return backend.get_resolution(name=name, validate=validate_type)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e


@router.get(
    "/{name}/sources",
    responses={404: {"model": NotFoundError}},
)
def get_leaf_source_resolutions(
    backend: BackendDependency,
    name: ResolutionName,
) -> list[Resolution]:
    """Get all sources in scope for a resolution."""
    try:
        return backend.get_leaf_source_resolutions(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e


@router.delete(
    "/{name}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResolutionOperationStatus,
            **ResolutionOperationStatus.status_409_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
)
def delete_resolution(
    backend: BackendDependency,
    name: ResolutionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the resolution")
    ] = False,
) -> ResolutionOperationStatus:
    """Delete a resolution from the backend."""
    try:
        backend.delete_resolution(name=name, certain=certain)
        return ResolutionOperationStatus(
            success=True,
            name=name,
            operation=CRUDOperation.DELETE,
        )
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxDeletionNotConfirmed as e:
        raise HTTPException(
            status_code=409,
            detail=ResolutionOperationStatus(
                success=False,
                name=name,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.post(
    "/{name}/data",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
)
def set_data(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    name: ResolutionName,
    validate_type: ResolutionType | None = None,
) -> UploadStatus:
    """Create an upload task for source hashes or model results."""
    try:
        resolution = backend.get_resolution(name=name, validate=validate_type)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e

    if resolution.resolution_type == ResolutionType.SOURCE:
        upload_id = upload_tracker.add_source(metadata=resolution)
        return upload_tracker.get(upload_id=upload_id).status

    upload_id = upload_tracker.add_model(metadata=resolution)
    return upload_tracker.get(upload_id=upload_id).status


@router.get(
    "/{name}/data",
    responses={404: {"model": NotFoundError}},
)
def get_results(
    backend: BackendDependency, name: ModelResolutionName
) -> ParquetResponse:
    """Download results for a model as a parquet file."""
    try:
        res = backend.get_model_data(name=name)
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
def set_truth(
    backend: BackendDependency,
    name: ModelResolutionName,
    truth: Annotated[int, Body(ge=0, le=100)],
) -> ResolutionOperationStatus:
    """Set truth data for a resolution."""
    try:
        # This will fail for a source, which is what we want for now
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
def get_truth(backend: BackendDependency, name: ModelResolutionName) -> float:
    """Get truth data for a resolution."""
    try:
        return backend.get_model_truth(name=name)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e
