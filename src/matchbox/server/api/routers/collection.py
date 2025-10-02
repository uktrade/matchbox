"""Collection and resolution API routes for the Matchbox server."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from matchbox.common.arrow import table_to_buffer
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
    UploadStatus,
    Version,
    VersionName,
)
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionNotFoundError,
    MatchboxVersionAlreadyExists,
    MatchboxVersionNotFoundError,
)
from matchbox.server.api.dependencies import (
    BackendDependency,
    ParquetResponse,
    UploadTrackerDependency,
    authorisation_dependencies,
)

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


# Version management endpoints


@router.get(
    "/{collection}/versions/{version}",
    responses={404: {"model": NotFoundError}},
    summary="Get specific version",
    description="Retrieve details for a specific version within a collection.",
)
def get_version(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
) -> Version:
    """Get a specific version."""
    return backend.get_version(collection=collection, name=version)


@router.post(
    "/{collection}/versions/{version}",
    responses={
        404: {"model": NotFoundError},
        409: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_409_examples(),
        },
    },
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Create a new version",
    description="Create a new version within the specified collection.",
)
def create_version(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
) -> ResourceOperationStatus:
    """Create a new version in a collection."""
    try:
        backend.create_version(collection=collection, name=version)
        return ResourceOperationStatus(
            success=True,
            name=version,
            operation=CRUDOperation.CREATE,
        )
    except (MatchboxVersionAlreadyExists, MatchboxCollectionNotFoundError) as e:
        raise HTTPException(
            status_code=409,
            detail=ResourceOperationStatus(
                success=False,
                name=version,
                operation=CRUDOperation.CREATE,
                details=str(e),
            ),
        ) from e


@router.patch(
    "/{collection}/versions/{version}/mutable",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Change version mutability",
    description="Set whether a version can be modified.",
)
def set_version_mutable(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
    mutable: Annotated[bool, Body(description="Mutability setting")],
) -> ResourceOperationStatus:
    """Set version mutability."""
    try:
        backend.set_version_mutable(
            collection=collection, name=version, mutable=mutable
        )
        return ResourceOperationStatus(
            success=True,
            name=version,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxVersionNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=version,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.patch(
    "/{collection}/versions/{version}/default",
    responses={
        404: {"model": NotFoundError},
        500: {
            "model": ResourceOperationStatus,
            **ResourceOperationStatus.status_500_examples(),
        },
    },
    dependencies=[Depends(authorisation_dependencies)],
    summary="Change default version",
    description="Set whether a version is the default for its collection.",
)
def set_version_default(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
    default: Annotated[bool, Body(description="Default setting")],
) -> ResourceOperationStatus:
    """Set version as default."""
    try:
        backend.set_version_default(
            collection=collection, name=version, default=default
        )
        return ResourceOperationStatus(
            success=True,
            name=version,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxVersionNotFoundError,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=version,
                operation=CRUDOperation.UPDATE,
                details=str(e),
            ).model_dump(),
        ) from e


@router.delete(
    "/{collection}/versions/{version}",
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
    summary="Delete a version",
    description="Delete a version and all its resolutions. Requires confirmation.",
)
def delete_version(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the version")
    ] = False,
) -> ResourceOperationStatus:
    """Delete a version."""
    try:
        backend.delete_version(collection=collection, name=version, certain=certain)
        return ResourceOperationStatus(
            success=True, name=version, operation=CRUDOperation.DELETE
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxVersionNotFoundError,
        MatchboxDeletionNotConfirmed,
    ):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ResourceOperationStatus(
                success=False,
                name=version,
                operation=CRUDOperation.DELETE,
                details=str(e),
            ),
        ) from e


# Resolution management


@router.post(
    "/{collection}/versions/{version}/resolutions/{resolution_name}",
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
    description="Create a new resolution (model or source) in the specified version.",
)
def create_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
    resolution_name: ResolutionName,
    resolution: Resolution,
) -> ResourceOperationStatus:
    """Create a resolution (model or source)."""
    try:
        backend.create_resolution(
            resolution=resolution,
            path=ResolutionPath(
                name=resolution_name, collection=collection, version=version
            ),
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution_name,
            operation=CRUDOperation.CREATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxVersionNotFoundError,
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
    "/{collection}/versions/{version}/resolutions/{resolution}",
    responses={404: {"model": NotFoundError}},
    summary="Get a resolution",
    description="Retrieve a specific resolution (model or source) from the backend.",
)
def get_resolution(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
    resolution: ResolutionName,
    validate_type: ResolutionType | None = None,
) -> Resolution:
    """Get a resolution (model or source) from the backend."""
    return backend.get_resolution(
        path=ResolutionPath(collection=collection, version=version, name=resolution),
        validate=validate_type,
    )


@router.delete(
    "/{collection}/versions/{version}/resolutions/{resolution}",
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
    version: VersionName,
    resolution: ResolutionName,
    certain: Annotated[
        bool, Query(description="Confirm deletion of the resolution")
    ] = False,
) -> ResourceOperationStatus:
    """Delete a resolution from the backend."""
    try:
        backend.delete_resolution(
            name=ResolutionPath(
                collection=collection, version=version, name=resolution
            ),
            certain=certain,
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution,
            operation=CRUDOperation.DELETE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxVersionNotFoundError,
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
    "/{collection}/versions/{version}/resolutions/{resolution}/data",
    responses={404: {"model": NotFoundError}},
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(authorisation_dependencies)],
    summary="Set resolution data",
    description="Create an upload task for source hashes or model results.",
)
def set_data(
    backend: BackendDependency,
    upload_tracker: UploadTrackerDependency,
    collection: CollectionName,
    version: VersionName,
    resolution: ModelResolutionName,
    validate_type: ResolutionType | None = None,
) -> UploadStatus:
    """Create an upload task for source hashes or model results."""
    # Get resolution from the specified version
    resolution_path = ResolutionPath(
        collection=collection, version=version, name=resolution
    )
    resolution = backend.get_resolution(path=resolution_path, validate=validate_type)

    if resolution.resolution_type == ResolutionType.SOURCE:
        upload_id = upload_tracker.add_source(path=resolution_path)
        return upload_tracker.get(upload_id=upload_id).status

    upload_id = upload_tracker.add_model(path=resolution_path)
    return upload_tracker.get(upload_id=upload_id).status


@router.get(
    "/{collection}/versions/{version}/resolutions/{resolution}/data",
    responses={404: {"model": NotFoundError}},
    summary="Get resolution results",
    description="Download results for a model as a parquet file.",
)
def get_results(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
    resolution: ModelResolutionName,
) -> ParquetResponse:
    """Download results for a model as a parquet file."""
    res = backend.get_model_data(
        path=ResolutionPath(collection=collection, version=version, name=resolution)
    )

    buffer = table_to_buffer(res)
    return ParquetResponse(buffer.getvalue())


@router.patch(
    "/{collection}/versions/{version}/resolutions/{resolution}/truth",
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
    version: VersionName,
    resolution: ModelResolutionName,
    truth: Annotated[int, Body(ge=0, le=100)],
) -> ResourceOperationStatus:
    """Set truth data for a resolution."""
    try:
        backend.set_model_truth(
            path=ResolutionPath(
                collection=collection, version=version, name=resolution
            ),
            truth=truth,
        )
        return ResourceOperationStatus(
            success=True,
            name=resolution,
            operation=CRUDOperation.UPDATE,
        )
    except (
        MatchboxCollectionNotFoundError,
        MatchboxVersionNotFoundError,
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
    "/{collection}/versions/{version}/resolutions/{resolution}/truth",
    responses={404: {"model": NotFoundError}},
    summary="Get resolution truth",
    description="Get truth data for a resolution.",
)
def get_truth(
    backend: BackendDependency,
    collection: CollectionName,
    version: VersionName,
    resolution: ModelResolutionName,
) -> float:
    """Get truth data for a resolution."""
    return backend.get_model_truth(
        path=ResolutionPath(collection=collection, version=version, name=resolution)
    )
