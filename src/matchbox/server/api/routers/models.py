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
    ParquetResponse,
    UploadTrackerDependency,
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
def insert_model(
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
def get_model(backend: BackendDependency, name: ModelResolutionName) -> ModelConfig:
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
def set_results(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
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

    upload_id = upload_tracker.add_model(metadata=metadata)
    return upload_tracker.get(upload_id=upload_id).status


@router.get(
    "/{name}/results",
    responses={404: {"model": NotFoundError}},
)
def get_results(
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
def set_truth(
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
def get_truth(backend: BackendDependency, name: ModelResolutionName) -> float:
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
