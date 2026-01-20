"""Admin functions for the client handler."""

from matchbox.client._handler.main import CLIENT, http_retry, url_params
from matchbox.common.dtos import (
    BackendCountableType,
    ResourceOperationStatus,
)
from matchbox.common.logging import logger


@http_retry
def count_backend_items(
    entity: BackendCountableType | None = None,
) -> dict[str, int]:
    """Count the number of various entities in the backend."""
    if entity is not None and entity not in BackendCountableType:
        raise ValueError(
            f"Invalid entity type: {entity}. "
            f"Must be one of {list(BackendCountableType)} "
        )

    log_prefix = "Backend count"
    logger.debug("Counting", prefix=log_prefix)

    res = CLIENT.get("/database/count", params=url_params({"entity": entity}))

    counts = res.json()
    logger.debug(f"Counts: {counts}", prefix=log_prefix)

    return counts


@http_retry
def delete_orphans() -> ResourceOperationStatus:
    """Delete orphaned clusters."""
    logger.debug("Deleting orphans")

    res = CLIENT.delete("/database/orphans")
    return ResourceOperationStatus.model_validate(res.json())
