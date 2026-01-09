"""Resolutions API routes for the Matchbox server."""

import zipfile
from io import BytesIO
from typing import Annotated

from fastapi import (
    APIRouter,
    Query,
    Response,
    status,
)

from matchbox.common.arrow import JudgementsZipFilenames, table_to_buffer
from matchbox.common.dtos import (
    CollectionName,
    ErrorResponse,
    ModelResolutionName,
    ModelResolutionPath,
    RunID,
)
from matchbox.common.eval import Judgement
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    ZipResponse,
)

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post(
    "/judgements",
    responses={404: {"model": ErrorResponse}},
    status_code=status.HTTP_201_CREATED,
)
def insert_judgement(
    backend: BackendDependency,
    judgement: Judgement,
) -> Response:
    """Submit judgement from human evaluator."""
    backend.insert_judgement(judgement=judgement)
    return Response(status_code=status.HTTP_201_CREATED)


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
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def sample(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
    n: int,
    user_name: str,
) -> ParquetResponse:
    """Sample n cluster to validate."""
    sample = backend.sample_for_eval(
        path=ModelResolutionPath(collection=collection, run=run_id, name=resolution),
        n=n,
        user_name=user_name,
    )
    buffer = table_to_buffer(sample)
    return ParquetResponse(buffer.getvalue())
