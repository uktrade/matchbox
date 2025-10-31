"""Collection and resolution API routes for the Matchbox server."""

import uuid
from typing import Annotated

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
from pyarrow import parquet as pq

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS, table_to_buffer
from matchbox.common.dtos import (
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
    UploadInfo,
    UploadStage,
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
            **ResourceOperationStatus.error_examples(),
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
            success=True,
            target=f"Collection {collection}",
            operation=CRUDOperation.CREATE,
        )
    except MatchboxCollectionAlreadyExists as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                target=f"Collection {collection}",
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
            **ResourceOperationStatus.error_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
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
            success=True,
            target=f"Collection {collection}",
            operation=CRUDOperation.DELETE,
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
                target=f"Collection {collection}",
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
            **ResourceOperationStatus.error_examples(),
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
                target=f"Collection {collection}",
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
            **ResourceOperationStatus.error_examples(),
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
            target=f"Run {run_id}",
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
                target=f"Run {run_id}",
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
            **ResourceOperationStatus.error_examples(),
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
            target=f"Run {run_id}",
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
                target=f"Run {run_id}",
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
            **ResourceOperationStatus.error_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
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
            success=True, target=f"Run {run_id}", operation=CRUDOperation.DELETE
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
                target=f"Run {run_id}",
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
            **ResourceOperationStatus.error_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
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
    resolution_path = ResolutionPath(
        name=resolution_name, collection=collection, run=run_id
    )
    try:
        backend.create_resolution(
            resolution=resolution,
            path=resolution_path,
        )
        return ResourceOperationStatus(
            success=True,
            target=f"Resolution {resolution_path}",
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
                target=f"Resolution {resolution_path}",
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                target=f"Resolution {resolution_path}",
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
) -> Resolution:
    """Get a resolution (model or source) from the backend."""
    return backend.get_resolution(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution)
    )


@router.put(
    "/{collection}/runs/{run_id}/resolutions/{resolution_name}",
    responses={
        404: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
        },
    },
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Update a resolution",
    description="Update an existing resolution (model or source) in the specified run.",
)
def update_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution_name: ResolutionName,
    resolution: Resolution,
) -> ResourceOperationStatus:
    """Update an existing resolution (model or source)."""
    resolution_path = ResolutionPath(
        name=resolution_name, collection=collection, run=run_id
    )
    try:
        backend.update_resolution(
            resolution=resolution,
            path=resolution_path,
        )
        return ResourceOperationStatus(
            success=True,
            target=f"Resolution {resolution_path}",
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxRunNotFoundError,
        MatchboxResolutionNotFoundError,
    ) as e:
        raise HTTPException(
            status_code=404,
            detail=ResourceOperationStatus(
                success=False,
                target=f"Resolution {resolution_path}",
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                target=f"Resolution {resolution_path}",
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.delete(
    "/{collection}/runs/{run_id}/resolutions/{resolution}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
        },
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
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
    resolution_path = ResolutionPath(collection=collection, run=run_id, name=resolution)
    try:
        backend.delete_resolution(
            name=resolution_path,
            certain=certain,
        )
        return ResourceOperationStatus(
            success=True,
            target=f"Resolution {resolution_path}",
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
                target=f"Resolution {resolution_path}",
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


@router.post(
    "/{collection}/runs/{run_id}/resolutions/{resolution_name}/data",
    responses={
        404: {"model": NotFoundError},
        400: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.error_examples(),
        },
    },
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Set resolution data",
    description="Create an upload task for source hashes or model results.",
)
def set_data(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    settings: SettingsDependency,
    background_tasks: BackgroundTasks,
    collection: CollectionName,
    run_id: RunID,
    resolution_name: ModelResolutionName,
    file: UploadFile,
) -> ResourceOperationStatus:
    """Create an upload task for source hashes or model results."""
    resolution_path = ResolutionPath(
        collection=collection, run=run_id, name=resolution_name
    )
    # Signal to clients that resolution is not ready to receive data
    backend.set_resolution_stage(path=resolution_path, stage=UploadStage.PROCESSING)
    resolution = backend.get_resolution(path=resolution_path)

    upload_id = str(uuid.uuid4())

    # Validate file
    if ".parquet" not in file.filename:
        raise HTTPException(
            status_code=400,
            detail=ResourceOperationStatus(
                success=False,
                target=f"Resolution data {resolution_path}",
                operation=CRUDOperation.CREATE,
                details=f"Expected .parquet file, got {file.filename.split('.')[-1]}",
            ),
        )

    # pyarrow validates Parquet magic numbers when loading file
    try:
        pq.ParquetFile(file.file)
    except ArrowInvalid as e:
        raise HTTPException(
            status_code=400,
            detail=ResourceOperationStatus(
                success=False,
                target=f"Resolution data {resolution_path}",
                operation=CRUDOperation.CREATE,
                details=f"Invalid Parquet file: {str(e)}",
            ),
        ) from e

    # Upload to S3
    client = backend.settings.datastore.get_client()
    bucket = backend.settings.datastore.cache_bucket_name
    key = f"{upload_id}.parquet"

    if resolution.resolution_type == ResolutionType.SOURCE:
        expected_schema = SCHEMA_INDEX
    else:
        expected_schema = SCHEMA_RESULTS

    try:
        table_to_s3(
            client=client,
            bucket=bucket,
            key=key,
            file=file,
            expected_schema=expected_schema,
        )
    except MatchboxServerFileError as e:
        raise HTTPException(
            status_code=400,
            detail=ResourceOperationStatus(
                success=False,
                target=f"Resolution data {resolution_path}",
                operation=CRUDOperation.CREATE,
                details=f"Could not upload file to object storage: {str(e)}",
            ),
        ) from e

    # Start background processing
    match settings.task_runner:
        case "api":
            background_tasks.add_task(
                process_upload,
                backend=backend,
                tracker=upload_tracker,
                s3_client=client,
                resolution_path=resolution_path,
                upload_id=upload_id,
                bucket=bucket,
                filename=key,
            )
        case "celery":
            process_upload_celery.delay(
                resolution_path_json=resolution_path.model_dump_json(),
                upload_id=upload_id,
                bucket=bucket,
                filename=key,
            )
        case _:
            raise RuntimeError("Unsupported task runner.")

    return ResourceOperationStatus(
        success=True,
        target=f"Resolution data {resolution_path}",
        operation=CRUDOperation.CREATE,
        details=upload_id,
    )


@router.get(
    "/{collection}/runs/{run_id}/resolutions/{resolution}/data/status",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_200_OK,
)
def get_upload_status(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    collection: CollectionName,
    run_id: RunID,
    resolution: ResolutionName,
    upload_id: Annotated[str | None, Query()] = None,
) -> UploadInfo:
    """Get the status of an upload process.

    Optionally looks for error message if given an upload ID.
    """
    error = None
    if upload_id:
        error = upload_tracker.get(upload_id=upload_id)
    resolution_stage = backend.get_resolution_stage(
        path=ResolutionPath(collection=collection, run=run_id, name=resolution)
    )

    return UploadInfo(stage=resolution_stage, error=error)


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
