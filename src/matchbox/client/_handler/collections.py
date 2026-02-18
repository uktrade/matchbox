"""Collection, run and resolution management functions for the client handler."""

import time
from io import BytesIO

import polars as pl
from pyarrow import Table
from pyarrow.parquet import read_table

from matchbox.client._handler.main import CLIENT, http_retry, url_params
from matchbox.client._settings import settings
from matchbox.common.arrow import (
    table_to_buffer,
)
from matchbox.common.dtos import (
    Collection,
    CollectionName,
    DefaultGroup,
    GroupName,
    ModelResolutionPath,
    PermissionGrant,
    PermissionType,
    Resolution,
    ResolutionPath,
    ResolverResolutionPath,
    ResourceOperationStatus,
    Run,
    RunID,
    UploadInfo,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxServerFileError,
)
from matchbox.common.logging import logger, profile_time

# Collection management


@http_retry
def list_collections() -> list[CollectionName]:
    """Get all existing collection names."""
    logger.debug("Retrieving all collections")

    res = CLIENT.get("/collections")
    return [CollectionName(name) for name in res.json()]


@http_retry
def get_collection(name: CollectionName) -> Collection:
    """Get all runs and resolutions in a collection."""
    log_prefix = f"Collection {name}"
    logger.debug("Retrieving", prefix=log_prefix)

    res = CLIENT.get(f"/collections/{name}")
    return Collection.model_validate(res.json())


@http_retry
def create_collection(
    name: CollectionName, admin_group: GroupName = DefaultGroup.PUBLIC
) -> ResourceOperationStatus:
    """Create a new collection.

    Args:
        name: Name of the collection to create
        admin_group: The group that will get admin permissions on this collection by
            default. The default options is the public group, meaning anyone can read
            or write to the collection.
    """
    log_prefix = f"Collection {name}"
    logger.debug(f"Creating with admin group {admin_group}", prefix=log_prefix)

    permission_grant = PermissionGrant(
        group_name=admin_group,
        permission=PermissionType.ADMIN,
    )

    res = CLIENT.post(
        f"/collections/{name}",
        json=[permission_grant.model_dump()],
    )

    return ResourceOperationStatus.model_validate(res.json())


@http_retry
def delete_collection(
    name: CollectionName, certain: bool = False
) -> ResourceOperationStatus:
    """Delete a collection.

    Args:
        name: Name of the collection to delete.
        certain: Confirmation flag.
    """
    log_prefix = f"Collection {name}"
    logger.debug("Deleting", prefix=log_prefix)

    res = CLIENT.delete(
        f"/collections/{name}",
        params={"certain": certain},
    )
    return ResourceOperationStatus.model_validate(res.json())


# Collection permissions


@http_retry
def get_collection_permissions(collection: CollectionName) -> list[PermissionGrant]:
    """Get permissions for a collection."""
    response = CLIENT.get(f"/collections/{collection}/permissions")
    return [PermissionGrant(**p) for p in response.json()]


@http_retry
def grant_collection_permission(
    collection: CollectionName,
    group_name: GroupName,
    permission: PermissionType,
) -> ResourceOperationStatus:
    """Grant a permission on a collection."""
    response = CLIENT.post(
        f"/collections/{collection}/permissions",
        json={"group_name": group_name, "permission": permission},
    )
    return ResourceOperationStatus(**response.json())


@http_retry
def revoke_collection_permission(
    collection: CollectionName,
    group_name: GroupName,
    permission: PermissionType,
) -> ResourceOperationStatus:
    """Revoke a permission on a collection."""
    response = CLIENT.delete(
        f"/collections/{collection}/permissions/{permission}/{group_name}"
    )
    return ResourceOperationStatus(**response.json())


# Run management


@http_retry
def get_run(collection: CollectionName, run_id: RunID) -> Run:
    """Get all resolutions in a run."""
    log_prefix = f"Collection {collection}, run {run_id}"
    logger.debug("Retrieving", prefix=log_prefix)

    res = CLIENT.get(f"/collections/{collection}/runs/{run_id}")
    return Run.model_validate(res.json())


@http_retry
def create_run(collection: CollectionName) -> ResourceOperationStatus:
    """Create a new run."""
    log_prefix = f"Collection {collection}, new run"
    logger.debug("Creating", prefix=log_prefix)

    res = CLIENT.post(f"/collections/{collection}/runs")

    return Run.model_validate(res.json())


@http_retry
def delete_run(
    collection: CollectionName, run_id: RunID, certain: bool = False
) -> ResourceOperationStatus:
    """Delete a run in Matchbox."""
    log_prefix = f"Collection {collection}, run {run_id}"
    logger.debug("Deleting", prefix=log_prefix)

    res = CLIENT.delete(
        f"/collections/{collection}/runs/{run_id}",
        params={"certain": certain},
    )
    return ResourceOperationStatus.model_validate(res.json())


@http_retry
def set_run_mutable(
    collection: CollectionName, run_id: RunID, mutable: bool
) -> ResourceOperationStatus:
    """Set a run as mutable for a collection."""
    log_prefix = f"Collection {collection}, run {run_id}"
    logger.debug("Setting mutability", prefix=log_prefix)

    res = CLIENT.patch(f"/collections/{collection}/runs/{run_id}/mutable", json=mutable)
    return ResourceOperationStatus.model_validate(res.json())


@http_retry
def set_run_default(
    collection: CollectionName, run_id: RunID, default: bool
) -> ResourceOperationStatus:
    """Set a run as the default run for a collection."""
    log_prefix = f"Collection {collection}, run {run_id}"
    logger.debug("Setting as default", prefix=log_prefix)

    res = CLIENT.patch(f"/collections/{collection}/runs/{run_id}/default", json=default)
    return ResourceOperationStatus.model_validate(res.json())


# Resolution management


@http_retry
def create_resolution(
    resolution: Resolution,
    path: ResolutionPath,
) -> ResourceOperationStatus:
    """Create a resolution (model or source)."""
    log_prefix = f"Resolution {path}"
    logger.debug("Creating", prefix=log_prefix)

    res = CLIENT.post(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}",
        json=resolution.model_dump(),
    )

    return ResourceOperationStatus.model_validate(res.json())


@http_retry
def update_resolution(
    resolution: Resolution,
    path: ResolutionPath,
) -> ResourceOperationStatus:
    """Update a resolution (model or source)."""
    log_prefix = f"Resolution {path}"
    logger.debug("Updating", prefix=log_prefix)

    res = CLIENT.put(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}",
        json=resolution.model_dump(),
    )

    return ResourceOperationStatus.model_validate(res.json())


@profile_time(kwarg="path")
@http_retry
def get_resolution(path: ResolutionPath) -> Resolution | None:
    """Get a resolution from Matchbox."""
    log_prefix = f"Resolution {path}"
    logger.debug("Retrieving metadata", prefix=log_prefix)

    res = CLIENT.get(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}"
    )
    return Resolution.model_validate(res.json())


@profile_time(kwarg="path")
@http_retry
def set_data(path: ResolutionPath, data: pl.DataFrame | Table) -> str:
    """Upload source hashes or model results to server."""
    log_prefix = f"Resolution {path}"
    logger.debug("Uploading results", prefix=log_prefix)

    data_arrow = data.to_arrow() if isinstance(data, pl.DataFrame) else data
    buffer = table_to_buffer(table=data_arrow)

    # Initialise upload
    logger.debug("Uploading data", prefix=log_prefix)
    metadata_res = CLIENT.post(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/data",
        files={"file": ("data.parquet", buffer, "application/octet-stream")},
    )

    upload_id = ResourceOperationStatus.model_validate(metadata_res.json()).details

    # Poll until complete with retry/timeout configuration
    stage = UploadStage.PROCESSING
    while stage == UploadStage.PROCESSING:
        status_res = CLIENT.get(
            f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/data/status",
            params=url_params({"upload_id": upload_id}),
        )
        info = UploadInfo.model_validate(status_res.json())
        stage = info.stage
        logger.debug(f"Uploading data: {stage}", prefix=log_prefix)

        if stage == UploadStage.READY:
            raise MatchboxServerFileError(info.error)

        time.sleep(settings.retry_delay)

    logger.debug("Finished", prefix=log_prefix)
    return upload_id


@http_retry
def get_resolution_stage(path: ResolutionPath) -> UploadStage:
    status_res = CLIENT.get(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/data/status"
    )
    upload_info = UploadInfo.model_validate(status_res.json())
    return upload_info.stage


@profile_time(kwarg="path")
@http_retry
def get_results(path: ModelResolutionPath) -> Table:
    """Get model results from Matchbox."""
    log_prefix = f"Model {path}"
    logger.debug("Retrieving results", prefix=log_prefix)

    res = CLIENT.get(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/data"
    )
    buffer = BytesIO(res.content)
    return read_table(buffer)


@profile_time(kwarg="path")
@http_retry
def get_resolver_data(path: ResolverResolutionPath) -> Table:
    """Get resolver assignments from Matchbox."""
    log_prefix = f"Resolver {path}"
    logger.debug("Retrieving assignments", prefix=log_prefix)

    res = CLIENT.get(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/data"
    )
    buffer = BytesIO(res.content)
    return read_table(buffer)


@profile_time(kwarg="path")
@http_retry
def get_resolver_mapping(path: ResolverResolutionPath, upload_id: str) -> Table:
    """Get resolver upload mapping from Matchbox."""
    log_prefix = f"Resolver {path}"
    logger.debug("Retrieving upload mapping", prefix=log_prefix)

    res = CLIENT.get(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/data/mapping",
        params={"upload_id": upload_id},
    )
    buffer = BytesIO(res.content)
    return read_table(buffer)


@http_retry
def delete_resolution(
    path: ResolutionPath, certain: bool = False
) -> ResourceOperationStatus:
    """Delete a resolution in Matchbox."""
    log_prefix = f"Resolution {path}"
    logger.debug("Deleting", prefix=log_prefix)

    res = CLIENT.delete(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}",
        params={"certain": certain},
    )
    return ResourceOperationStatus.model_validate(res.json())
