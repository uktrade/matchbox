"""Resolutions API routes for the Matchbox server."""

import zipfile
from io import BytesIO
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
)

from matchbox.common.arrow import JudgementsZipFilenames, table_to_buffer
from matchbox.common.dtos import (
    BackendParameterType,
    BackendResourceType,
    CollectionName,
    InvalidParameterError,
    ModelResolutionName,
    ModelResolutionPath,
    NotFoundError,
    PermissionType,
    RunID,
    User,
)
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxResolutionNotFoundError,
    MatchboxTooManySamplesRequested,
    MatchboxUserNotFoundError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    CurrentUserDependency,  # [Added Import]
    ParquetResponse,
    RequiresPermission,
    ZipResponse,
)

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post(
    "/judgements",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_201_CREATED,
)
def insert_judgement(
    backend: BackendDependency,
    judgement: Judgement,
    user: CurrentUserDependency,
) -> Response:
    """Submit judgement from human evaluator."""
    if not user or user.user_name == "_public":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        judgement.user_name = user.user_name
        backend.insert_judgement(judgement=judgement)
        return Response(status_code=status.HTTP_201_CREATED)
    except MatchboxDataNotFound as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.CLUSTER
            ).model_dump(),
        ) from e
    except MatchboxUserNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.USER
            ).model_dump(),
        ) from e


@router.get(
    "/judgements",
)
def get_judgements(
    backend: BackendDependency,
    tag: Annotated[
        str | None, Query(description="Tag by which to filter judgements")
    ] = None,
) -> ParquetResponse:
    """Retrieve all judgements from human evaluators."""
    judgements, expansion = backend.get_judgements(tag)
    judgements_buffer, expansion_buffer = (
        table_to_buffer(judgements),
        table_to_buffer(expansion),
    )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(JudgementsZipFilenames.JUDGEMENTS, judgements_buffer.read())
        zip_file.writestr(JudgementsZipFilenames.EXPANSION, expansion_buffer.read())

    zip_buffer.seek(0)

    return ZipResponse(zip_buffer.getvalue())


@router.get(
    "/samples",
    responses={404: {"model": NotFoundError}, 422: {"model": InvalidParameterError}},
)
def sample(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
    n: int,
    user: Annotated[
        User,
        Depends(
            RequiresPermission(
                PermissionType.READ,
                resource_from_param="collection",
                allow_public=False,
            )
        ),
    ],
) -> ParquetResponse:
    """Sample n cluster to validate."""
    try:
        sample = backend.sample_for_eval(
            path=ModelResolutionPath(
                collection=collection, run=run_id, name=resolution
            ),
            n=n,
            user_name=user.user_name,
        )
        buffer = table_to_buffer(sample)
        return ParquetResponse(buffer.getvalue())
    except MatchboxResolutionNotFoundError:
        raise
    except MatchboxUserNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.USER
            ).model_dump(),
        ) from e
    except MatchboxTooManySamplesRequested as e:
        raise HTTPException(
            status_code=422,
            detail=InvalidParameterError(
                details=str(e), parameter=BackendParameterType.SAMPLE_SIZE
            ).model_dump(),
        ) from e
