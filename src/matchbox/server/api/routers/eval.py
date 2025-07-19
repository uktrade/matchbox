"""Resolutions API routes for the Matchbox server."""

import zipfile
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response, status

from matchbox.common.arrow import JudgementsZipFilenames, table_to_buffer
from matchbox.common.dtos import (
    BackendParameterType,
    BackendResourceType,
    InvalidParameterError,
    ModelResolutionName,
    NotFoundError,
)
from matchbox.common.eval import Judgement, ModelComparison
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxNoJudgements,
    MatchboxResolutionNotFoundError,
    MatchboxTooManySamplesRequested,
    MatchboxUserNotFoundError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    ZipResponse,
)

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post(
    "/judgements",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_201_CREATED,
)
async def insert_judgement(
    backend: BackendDependency,
    judgement: Judgement,
):
    """Submit judgement from human evaluator."""
    try:
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
async def get_judgements(backend: BackendDependency) -> ParquetResponse:
    """Retrieve all judgements from human evaluators."""
    judgements, expansion = backend.get_judgements()
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
    "/compare",
    responses={404: {"model": NotFoundError}},
)
async def compare_models(
    backend: BackendDependency,
    resolutions: Annotated[
        list[ModelResolutionName],
        Query(description="The resolution names for the models to compare."),
    ],
) -> ModelComparison:
    """Return comparison of selected models."""
    try:
        return backend.compare_models(resolutions)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e
    except MatchboxNoJudgements as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.JUDGEMENT
            ).model_dump(),
        ) from e


@router.get(
    "/samples",
    responses={404: {"model": NotFoundError}, 422: {"model": InvalidParameterError}},
)
async def sample(
    backend: BackendDependency, n: int, resolution: ModelResolutionName, user_id: int
) -> ParquetResponse:
    """Sample n cluster to validate."""
    try:
        sample = backend.sample_for_eval(n=n, resolution=resolution, user_id=user_id)
        buffer = table_to_buffer(sample)
        return ParquetResponse(buffer.getvalue())
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e
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
