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
    ModelResolutionPath,
    PermissionGrant,
    PermissionType,
    Resolution,
    ResolutionPath,
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
def create_collection(name: CollectionName) -> ResourceOperationStatus:
    """Create a new collection."""
    log_prefix = f"Collection {name}"
    logger.debug("Creating", prefix=log_prefix)

    # All collections are public r/w for now
    permission_grant = PermissionGrant(
        group_name=DefaultGroup.PUBLIC,
        permission=PermissionType.WRITE,
    )

    res = CLIENT.post(
        f"/collections/{name}",
        json=[permission_grant.model_dump()],
    )

    return ResourceOperationStatus.model_validate(res.json())


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
def set_data(path: ResolutionPath, data: pl.DataFrame | Table) -> None:
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


@http_retry
def delete_resolution(
    path: ModelResolutionPath, certain: bool = False
) -> ResourceOperationStatus:
    """Delete a resolution in Matchbox."""
    log_prefix = f"Model {path}"
    logger.debug("Deleting", prefix=log_prefix)

    res = CLIENT.delete(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}",
        params={"certain": certain},
    )
    return ResourceOperationStatus.model_validate(res.json())
