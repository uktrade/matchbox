"""Collection and resolution API routes for the Matchbox server."""

from datetime import datetime
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
    BackendUploadType,
    Collection,
    CollectionName,
    CRUDOperation,
    ModelResolutionName,
    NotFoundError,
    Resolution,
    ResolutionName,
    ResolutionPath,
    ResolutionType,
    ResourceOperationStatus,
    Run,
    RunID,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionNotFoundError,
    MatchboxRunNotFoundError,
    MatchboxServerFileError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    SettingsDependency,
    UploadTrackerDependency,
    authorisation_dependencies,
)
from matchbox.server.uploads import process_upload, process_upload_celery, table_to_s3

router = APIRouter(prefix="/collections", tags=["collection"])


# Collection management endpoints


@router.get(
    "",
    summary="List all collections",
    description="Retrieve a list of all collection names in the system.",
)
def list_collections(backend: BackendDependency) -> list[CollectionName]:
    """List all collections."""
    return backend.list_collections()


@router.get(
    "/{collection}",
    responses={404: {"model": NotFoundError}},
    summary="Get collection details",
    description=(
        "Retrieve details for a specific collection, including all its versions "
        "and resolutions."
    ),
)
def get_collection(
    backend: BackendDependency,
    collection: CollectionName,
) -> Collection:
    """Get collection details with all versions and resolutions."""
    return backend.get_collection(name=collection)


@router.post(
    "/{collection}",
    responses={
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Create a new collection",
    description="Create a new collection with the specified name.",
)
def create_collection(
    backend: BackendDependency,
    collection: CollectionName,
) -> ResourceOperationStatus:
    """Create a new collection."""
    try:
        backend.create_collection(name=collection)
        return ResourceOperationStatus(
            success=True, name=collection, operation=CRUDOperation.CREATE
        )
    except MatchboxCollectionAlreadyExists as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                name=collection,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e


@router.delete(
    "/{collection}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Delete a collection",
    description="Delete a collection and all its versions. Requires confirmation.",
)
def delete_collection(
    backend: BackendDependency,
    collection: CollectionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the collection")
    ] = False,
) -> ResourceOperationStatus:
    """Delete a collection."""
    try:
        backend.delete_collection(name=collection, certain=certain)
        return ResourceOperationStatus(
            success=True, name=collection, operation=CRUDOperation.DELETE
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxDeletionNotConfirmed,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=collection,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


# Run management endpoints


@router.get(
    "/{collection}/runs/{run_id}",
    responses={404: {"model": NotFoundError}},
    summary="Get specific run",
    description="Retrieve details for a specific run within a collection.",
)
def get_run(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
) -> Run:
    """Get a specific run."""
    return backend.get_run(collection=collection, run_id=run_id)


@router.post(
    "/{collection}/runs",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Create a new run",
    description="Create a new run within the specified collection.",
)
def create_run(
    backend: BackendDependency,
    collection: CollectionName,
) -> Run:
    """Create a new run in a collection."""
    try:
        return backend.create_run(collection=collection)
    except MatchboxCollectionNotFoundError as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                name=collection,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e


@router.patch(
    "/{collection}/runs/{run_id}/mutable",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Change run mutability",
    description="Set whether a run can be modified.",
)
def set_run_mutable(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    mutable: Annotated[bool, Body(description="Mutability setting")],
) -> ResourceOperationStatus:
    """Set run mutability."""
    try:
        backend.set_run_mutable(collection=collection, run_id=run_id, mutable=mutable)
        return ResourceOperationStatus(
            success=True,
            name=run_id,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=run_id,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.patch(
    "/{collection}/runs/{run_id}/default",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Change default run",
    description="Set whether a run is the default for its collection.",
)
def set_run_default(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    default: Annotated[bool, Body(description="Default setting")],
) -> ResourceOperationStatus:
    """Set run as default."""
    try:
        backend.set_run_default(collection=collection, run_id=run_id, default=default)
        return ResourceOperationStatus(
            success=True,
            name=run_id,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=run_id,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.delete(
    "/{collection}/runs/{run_id}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Delete a run",
    description="Delete a run and all its resolutions. Requires confirmation.",
)
def delete_run(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    certain: Annotated[bool, Query(description="Confirm deletion of the run")] = False,
) -> ResourceOperationStatus:
    """Delete a run."""
    try:
        backend.delete_run(collection=collection, run_id=run_id, certain=certain)
        return ResourceOperationStatus(
            success=True, name=run_id, operation=CRUDOperation.DELETE
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
        MatchboxDeletionNotConfirmed,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=run_id,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


# Resolution management


@router.post(
    "/{collection}/runs/{run_id}/resolutions/{resolution_name}",
    responses={
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Create a resolution",
    description="Create a new resolution (model or source) in the specified run.",
)
def create_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution_name: ResolutionName,
    resolution: Resolution,
) -> ResourceOperationStatus:
    """Create a resolution (model or source)."""
    try:
        backend.create_resolution(
            resolution=resolution,
            path=ResolutionPath(
                name=resolution_name, collection=collection, run=run_id
            ),
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution_name,
            operation=CRUDOperation.CREATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
    ):
        raise
    except MatchboxResolutionAlreadyExists as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution_name,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution_name,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}",
    responses={404: {"model": NotFoundError}},
    summary="Get a resolution",
    description="Retrieve a specific resolution (model or source) from the backend.",
)
def get_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ResolutionName,
    validate_type: ResolutionType | None = None,
) -> Resolution:
    """Get a resolution (model or source) from the backend."""
    return backend.get_resolution(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution),
        validate=validate_type,
    )


@router.delete(
    "/{collection}/runs/{run_id}/resolutions/{resolution}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Delete a resolution",
    description="Delete a resolution from the backend. Requires confirmation.",
)
def delete_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ResolutionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the resolution")
    ] = False,
) -> ResourceOperationStatus:
    """Delete a resolution from the backend."""
    try:
        backend.delete_resolution(
            name=ResolutionPath(collection=collection, run=run_id, name=resolution),
            certain=certain,
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution,
            operation=CRUDOperation.DELETE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
        MatchboxDeletionNotConfirmed,
        MatchboxResolutionNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


@router.post(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/data",
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
        404: {"model": NotFoundError},
    },
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Upload data",
    description="Upload and process file for source hashes or model results.",
)
def upload_data(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    settings: SettingsDependency,
    background_tasks: BackgroundTasks,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
    file: UploadFile,
    validate_type: ResolutionType | None = None,
) -> UploadStatus:
    """Upload and process a file based on metadata type.

    The file is uploaded to S3 and then processed in a background task.
    Status can be checked using the /upload/{upload_id}/status endpoint.

    Raises HTTP 400 if:

    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    * Upload is already being processed
    * Uploaded data doesn't match the metadata schema
    """
    # Get resolution from the specified run
    resolution_path = ResolutionPath(collection=collection, run=run_id, name=resolution)
    resolution = backend.get_resolution(path=resolution_path, validate=validate_type)

    if resolution.resolution_type == ResolutionType.SOURCE:
        full_path = upload_tracker.add_source(path=resolution_path)

    full_path = upload_tracker.add_model(path=resolution_path)

    # Get and validate cache entry
    upload_entry = upload_tracker.get(full_path=full_path)
    if not upload_entry:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                full_path=full_path,
                update_timestamp=datetime.now(),
                stage=UploadStage.UNKNOWN,
                details=(
                    "Upload path not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
            ).model_dump(),
        )

    # Validate file
    if ".parquet" not in file.filename:
        upload_tracker.update(
            full_path=full_path,
            stage=UploadStage.READY,
            details=(
                f"Server expected .parquet file, got {file.filename.split('.')[-1]}"
            ),
        )
        updated_entry = upload_tracker.get(full_path)
        raise HTTPException(
            status_code=400,
            detail=updated_entry.status.model_dump(),
        )

    # pyarrow validates Parquet magic numbers when loading file
    try:
        pq.ParquetFile(file.file)
    except ArrowInvalid as e:
        upload_tracker.update(
            full_path=full_path,
            stage=UploadStage.READY,
            details=f"Invalid Parquet file: {str(e)}",
        )
        updated_entry = upload_tracker.get(full_path)
        raise HTTPException(
            status_code=400,
            detail=updated_entry.status.model_dump(),
        ) from e

    # Ensure tracker is expecting file
    if upload_entry.status.stage != UploadStage.AWAITING_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=upload_entry.status.model_dump(),
        )

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{collection}_{run_id}_{resolution}.parquet"

    try:
        table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=upload_entry.status.entity.schema,
        )
    except MatchboxServerFileError as e:
        upload_tracker.update(full_path, UploadStage.FAILED, details=str(e))
        updated_entry = upload_tracker.get(full_path)
        raise HTTPException(
            status_code=400,
            detail=updated_entry.status.model_dump(),
        ) from e

    upload_tracker.update(full_path, UploadStage.QUEUED)

    # Start background processing
    match settings.task_runner:
        case "api":
            background_tasks.add_task(
                process_upload,
                backend=backend,
                tracker=upload_tracker,
                s3_client=client,
                upload_type=upload_entry.status.entity,
                resolution_name=upload_entry.path.name,
                full_path=full_path,
                bucket=bucket,
                filename=key,
            )
        case "celery":
            process_upload_celery.delay(
                upload_type=upload_entry.status.entity,
                resolution_name=upload_entry.path.name,
                full_path=full_path,
                bucket=bucket,
                filename=key,
            )
        case _:
            raise RuntimeError("Unsupported task runner.")

    source_upload = upload_tracker.get(full_path)

    # Check for error in async task
    if source_upload.status.stage == UploadStage.FAILED:
        raise HTTPException(
            status_code=400,
            detail=source_upload.status.model_dump(),
        )
    else:
        return source_upload.status


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/data/status",
    responses={
        400: {"model": UploadStatus, **UploadStatus.status_400_examples()},
    },
    status_code=status.HTTP_200_OK,
)
def get_upload_status(
    upload_tracker: UploadTrackerDependency,
    full_path: str,
) -> UploadStatus:
    """Get the status of an upload process.

    Returns the current status of the upload and processing task.

    Raises HTTP 400 if:
    * Upload ID not found or expired (entries expire after 30 minutes of inactivity)
    """
    source_upload = upload_tracker.get(full_path=full_path)
    if not source_upload:
        raise HTTPException(
            status_code=400,
            detail=UploadStatus(
                full_path=full_path,
                update_timestamp=datetime.now(),
                stage=UploadStage.UNKNOWN,
                details=(
                    "Upload path not found or expired. Entries expire after 30 minutes "
                    "of inactivity, including failed processes."
                ),
                entity=BackendUploadType.INDEX,
            ).model_dump(),
        )

    return source_upload.status


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/data",
    responses={404: {"model": NotFoundError}},
    summary="Get resolution results",
    description="Download results for a model as a parquet file.",
)
def get_results(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
) -> ParquetResponse:
    """Download results for a model as a parquet file."""
    res = backend.get_model_data(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution)
    )

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@router.patch(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/truth",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Set resolution truth",
    description="Set truth data for a resolution.",
)
def set_truth(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
    truth: Annotated[int, Body(ge=0, le=100)],
) -> ResourceOperationStatus:
    """Set truth data for a resolution."""
    try:
        backend.set_model_truth(
            path=ResolutionPath(collection=collection, run=run_id, name=resolution),
            truth=truth,
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
        MatchboxResolutionNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=resolution,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/truth",
    responses={404: {"model": NotFoundError}},
    summary="Get resolution truth",
    description="Get truth data for a resolution.",
)
def get_truth(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ModelResolutionName,
) -> float:
    """Get truth data for a resolution."""
    return backend.get_model_truth(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution)
    )
