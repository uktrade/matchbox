"""Resolutions API routes for the Matchbox server."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    NotFoundError,
    ResolutionOperationStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.graph import ResolutionName
from matchbox.server.api.dependencies import (
    BackendDependency,
    authorisation_dependencies,
)

router = APIRouter(prefix="/resolutions", tags=["resolutions"])


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
async def delete_resolution(
    backend: BackendDependency,
    name: ResolutionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the model")
    ] = False,
) -> ResolutionOperationStatus:
    """Delete a model from the backend."""
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
