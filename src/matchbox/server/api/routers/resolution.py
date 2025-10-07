"""Resolution API routes for the Matchbox server."""

import datetime
from typing import Annotated

import pyarrow.parquet as pq
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pyarrow import ArrowInvalid

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    BackendUploadType,
    CRUDOperation,
    NotFoundError,
    Resolution,
    ResolutionOperationStatus,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
)
from matchbox.common.graph import ModelResolutionName, ResolutionName, ResolutionType
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    SettingsDependency,
    UploadTrackerDependency,
    authorisation_dependencies,
)
from matchbox.server.uploads import process_upload, process_upload_celery, table_to_s3

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
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
    },
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
)
def upload_data(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    name: ResolutionName,
    settings: SettingsDependency,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    validate_type: ResolutionType | None = None,
) -> UploadStatus:
    """Upload and process a file based on metadata type.

    The file is uploaded to S3 and then processed in a background task.
    Status can be checked using the /upload/{upload_id}/status endpoint.

    Raises HTTP 400 if:

    * Upload name not found or expired (entries expire after 30 minutes of inactivity)
    * Upload is already being processed
    * Uploaded data doesn't match the metadata schema
    * Uploaded data is not a parquet file
    """
    try:
        resolution = backend.get_resolution(name=name, validate=validate_type)
    except MatchboxResolutionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=NotFoundError(
                details=str(e), entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        ) from e

    # Validate file
    if ".parquet" not in file.filename:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                name=name,
                update_timestamp=datetime.now(),
                stage=upload_tracker.get(name).stage,
                details=(
                    f"Server expected .parquet file, got {file.filename.split('.')[-1]}"
                ),
            ).model_dump(),
        )

    # pyarrow validates Parquet magic numbers when loading file
    try:
        pq.ParquetFile(file.file)
    except ArrowInvalid as e:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                name=name,
                update_timestamp=datetime.now(),
                stage=upload_tracker.get(name).stage,
                details=(f"Invalid Parquet file: {str(e)}"),
            ).model_dump(),
        ) from e

    # add queue logic for items with same name here
    ...

    # Create entry
    if resolution.resolution_type == ResolutionType.SOURCE:
        upload_tracker.add_source(metadata=resolution)
    else:
        upload_tracker.add_model(metadata=resolution)

    # Get and validate cache entry
    upload_entry = upload_tracker.get(name=name)
    if not upload_entry:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                name=name,
                update_timestamp=datetime.now(),
                stage=upload_tracker.get(name).stage,
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
            ).model_dump(),
        )

    # Ensure tracker is expecting file
    # if upload_entry.stage != UploadStage.AWAITING_UPLOAD:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=upload_entry.model_dump(),
    #     )a

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{name}.parquet"

    try:
        table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=upload_entry.entity.schema,
        )
    except MatchboxServerFileError as e:
        upload_tracker.update(name, UploadStage.FAILED, details=str(e))
        updated_entry = upload_tracker.get(name)
        raise HTTPException(
            status_code=400,
            detail=updated_entry.model_dump(),
        ) from e

    upload_tracker.update(name, UploadStage.QUEUED)

    # Start background processing
    match settings.task_runner:
        case "api":
            background_tasks.add_task(
                process_upload,
                backend=backend,
                tracker=upload_tracker,
                s3_client=client,
                upload_type=upload_entry.entity,
                resolution_name=upload_entry.name,
                name=name,
                bucket=bucket,
                filename=key,
            )
        case "celery":
            process_upload_celery.delay(
                upload_type=upload_entry.entity,
                resolution_name=upload_entry.name,
                name=name,
                bucket=bucket,
                filename=key,
            )
        case _:
            raise RuntimeError("Unsupported task runner.")

    source_upload = upload_tracker.get(name)

    # Check for error in async task
    if source_upload.stage == UploadStage.FAILED:
        raise HTTPException(
            status_code=400,
            detail=source_upload.model_dump(),
        )
    else:
        return source_upload


@router.get(
    "/resolutions/{name}/data/status",
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
    },
    status_code=status.HTTP_200_OK,
)
def get_upload_status(
    upload_tracker: UploadTrackerDependency,
    name: str,
) -> UploadStatus:
    """Get the status of an upload process.

    Returns the current status of the upload and processing task.

    Raises HTTP 400 if:
    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    """
    source_upload = upload_tracker.get(name=name)
    if not source_upload:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                name=name,
                stage=UploadStage.UNKNOWN,
                update_timestamp=datetime.now(),
                details=(
                    "Upload ID not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )

    return source_upload


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
