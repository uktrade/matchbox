"""Functions abstracting the interaction with the server API."""

import time
from collections.abc import Iterable
from importlib.metadata import version
from io import BytesIO

import httpx
from pyarrow import Table
from pyarrow.parquet import read_table

from matchbox.client._settings import ClientSettings, settings
from matchbox.common.arrow import SCHEMA_MB_IDS, table_to_buffer
from matchbox.common.dtos import (
    BackendRetrievableType,
    ModelAncestor,
    ModelMetadata,
    ModelResolutionName,
    NotFoundError,
    ResolutionName,
    ResolutionOperationStatus,
    SourceResolutionName,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxClientFileError,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
)
from matchbox.common.graph import ResolutionGraph
from matchbox.common.hash import hash_to_base64
from matchbox.common.logging import logger
from matchbox.common.sources import Match, SourceConfig

URLEncodeHandledType = str | int | float | bytes


def encode_param_value(
    v: URLEncodeHandledType | Iterable[URLEncodeHandledType],
) -> str | list[str]:
    if isinstance(v, str):
        return v
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, bytes):
        return hash_to_base64(v)
    # Needs to be at the end, so we don't apply it to e.g. strings
    if isinstance(v, Iterable):
        return [encode_param_value(item) for item in v]
    raise ValueError(f"It was not possible to parse {v} as an URL parameter")


def url_params(
    params: dict[str, URLEncodeHandledType | Iterable[URLEncodeHandledType]],
) -> dict[str, str | list[str]]:
    """Prepares a dictionary of parameters to be encoded in a URL."""
    non_null = {k: v for k, v in params.items() if v}
    return {k: encode_param_value(v) for k, v in non_null.items()}


def handle_http_code(res: httpx.Response) -> httpx.Response:
    """Handle HTTP status codes and raise appropriate exceptions."""
    res.read()

    if 299 >= res.status_code >= 200:
        return res

    if res.status_code == 400:
        if UploadStatus.model_validate_json(res.content, strict=False):
            error = UploadStatus.model_validate(res.json())
            raise MatchboxServerFileError(error.details)
        else:
            raise RuntimeError(f"Unexpected 400 error: {res.content}")

    if res.status_code == 404:
        error = NotFoundError.model_validate(res.json())
        if error.entity == BackendRetrievableType.SOURCE:
            raise MatchboxSourceNotFoundError(error.details)
        if error.entity == BackendRetrievableType.RESOLUTION:
            raise MatchboxResolutionNotFoundError(error.details)
        else:
            raise RuntimeError(f"Unexpected 404 error: {error.details}")

    if res.status_code == 409:
        error = ResolutionOperationStatus.model_validate(res.json())
        raise MatchboxDeletionNotConfirmed(message=error.details)

    if res.status_code == 422:
        raise MatchboxUnparsedClientRequest(res.content)

    raise MatchboxUnhandledServerResponse(res.content)


def create_client(settings: ClientSettings) -> httpx.Client:
    """Create an HTTPX client with proper configuration."""
    return httpx.Client(
        base_url=settings.api_root,
        timeout=httpx.Timeout(60 * 30, connect=settings.timeout, pool=settings.timeout),
        event_hooks={"response": [handle_http_code]},
        headers=create_headers(settings),
    )


def create_headers(settings: ClientSettings) -> dict[str, str]:
    """Creates client headers."""
    headers = {"X-Matchbox-Client-Version": version("matchbox_db")}
    if settings.api_key is not None:
        headers["X-API-Key"] = settings.api_key.get_secret_value()
    return headers


CLIENT = create_client(settings=settings)


# Retrieval


def query(
    source: SourceResolutionName,
    resolution: ResolutionName | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> Table:
    log_prefix = f"Query {source}"
    logger.debug(f"Using {resolution}", prefix=log_prefix)

    res = CLIENT.get(
        "/query",
        params=url_params(
            {
                "name": source,
                "resolution": resolution,
                "threshold": threshold,
                "limit": limit,
            }
        ),
    )

    buffer = BytesIO(res.content)
    table = read_table(buffer)

    logger.debug("Finished", prefix=log_prefix)

    if not table.schema.equals(SCHEMA_MB_IDS):
        raise MatchboxClientFileError(
            message=(
                f"Schema mismatch. Expected:\n{SCHEMA_MB_IDS}\nGot:\n{table.schema}"
            )
        )

    return table


def match(
    targets: list[SourceResolutionName],
    source: SourceResolutionName,
    identifier: str,
    resolution: ModelResolutionName,
    threshold: int | None = None,
) -> Match:
    log_prefix = f"Query {source}"
    logger.debug(
        f"{identifier} to {', '.join(str(t) for t in targets)} using {resolution}",
        prefix=log_prefix,
    )

    res = CLIENT.get(
        "/match",
        params=url_params(
            {
                "target": targets,
                "source": source,
                "identifier": identifier,
                "resolution": resolution,
                "threshold": threshold,
            }
        ),
    )

    logger.debug("Finished", prefix=log_prefix)

    return [Match.model_validate(m) for m in res.json()]


# Data management


def index(source: SourceConfig, batch_size: int | None = None) -> UploadStatus:
    """Index a SourceConfig in Matchbox."""
    log_prefix = f"Index {source.address.pretty}"
    log_batch = f"with batch size {batch_size:,}" if batch_size else "without batching"
    logger.debug(f"Started {log_batch}", prefix=log_prefix)

    logger.debug("Retrieving and hashing", prefix=log_prefix)

    data_hashes = source.hash_data(batch_size=batch_size)

    buffer = table_to_buffer(table=data_hashes)

    # Upload metadata
    logger.debug("Uploading metadata", prefix=log_prefix)

    metadata_res = CLIENT.post("/sources", json=source.model_dump())

    upload = UploadStatus.model_validate(metadata_res.json())

    # Upload data
    logger.debug("Uploading data", prefix=log_prefix)

    upload_res = CLIENT.post(
        f"/upload/{upload.id}",
        files={"file": (f"{upload.id}.parquet", buffer, "application/octet-stream")},
    )

    # Poll until complete with retry/timeout configuration
    status = UploadStatus.model_validate(upload_res.json())
    while status.status not in ["complete", "failed"]:
        status_res = CLIENT.get(f"/upload/{upload.id}/status")
        status = UploadStatus.model_validate(status_res.json())

        logger.debug(f"Uploading data: {status.status}", prefix=log_prefix)

        if status.status == "failed":
            raise MatchboxServerFileError(status.details)

        time.sleep(settings.retry_delay)

    logger.debug("Finished")

    return status


def get_source(name: SourceResolutionName) -> SourceConfig:
    """Get a source configuration from Matchbox."""
    log_prefix = f"SourceConfig {name}"
    logger.debug("Retrieving", prefix=log_prefix)

    res = CLIENT.get(f"/sources/{name}")

    return SourceConfig.model_validate(res.json())


def get_resolution_sources(name: ModelResolutionName) -> list[SourceConfig]:
    """Get all sources for a model resolution from Matchbox."""
    log_prefix = f"Resolution {name}"
    logger.debug("Retrieving", prefix=log_prefix)

    res = CLIENT.get("/sources", params={"name": name})

    return [SourceConfig.model_validate(s) for s in res.json()]


def get_resolution_graph() -> ResolutionGraph:
    """Get the resolution graph from Matchbox."""
    log_prefix = "Visualisation"
    logger.debug("Fetching resolution graph", prefix=log_prefix)

    res = CLIENT.get("/report/resolutions")
    return ResolutionGraph.model_validate(res.json())


# Model management


def insert_model(model: ModelMetadata) -> ResolutionOperationStatus:
    """Insert a model in Matchbox."""
    log_prefix = f"Model {model.name}"
    logger.debug("Inserting metadata", prefix=log_prefix)

    res = CLIENT.post("/models", json=model.model_dump())
    return ResolutionOperationStatus.model_validate(res.json())


def get_model(name: ModelResolutionName) -> ModelMetadata:
    """Get model metadata from Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Retrieving metadata", prefix=log_prefix)

    res = CLIENT.get(f"/models/{name}")
    return ModelMetadata.model_validate(res.json())


def add_model_results(name: ModelResolutionName, results: Table) -> UploadStatus:
    """Upload model results in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Uploading results", prefix=log_prefix)

    buffer = table_to_buffer(table=results)

    # Initialise upload
    metadata_res = CLIENT.post(f"/models/{name}/results")

    upload = UploadStatus.model_validate(metadata_res.json())

    # Upload data
    upload_res = CLIENT.post(
        f"/upload/{upload.id}",
        files={"file": (f"{upload.id}.parquet", buffer, "application/octet-stream")},
    )

    logger.debug("Uploading data", prefix=log_prefix)

    # Poll until complete with retry/timeout configuration
    status = UploadStatus.model_validate(upload_res.json())
    while status.status not in ["complete", "failed"]:
        status_res = CLIENT.get(f"/upload/{upload.id}/status")
        status = UploadStatus.model_validate(status_res.json())

        logger.debug(f"Uploading data: {status.status}", prefix=log_prefix)

        if status.status == "failed":
            raise MatchboxServerFileError(status.details)

        time.sleep(settings.retry_delay)

    logger.debug("Finished", prefix=log_prefix)

    return status


def get_model_results(name: ModelResolutionName) -> Table:
    """Get model results from Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Retrieving results", prefix=log_prefix)

    res = CLIENT.get(f"/models/{name}/results")
    buffer = BytesIO(res.content)
    return read_table(buffer)


def set_model_truth(name: ModelResolutionName, truth: int) -> ResolutionOperationStatus:
    """Set the truth threshold for a model in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Setting truth value", prefix=log_prefix)

    res = CLIENT.patch(f"/models/{name}/truth", json=truth)
    return ResolutionOperationStatus.model_validate(res.json())


def get_model_truth(name: ModelResolutionName) -> int:
    """Get the truth threshold for a model in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Retrieving truth value", prefix=log_prefix)

    res = CLIENT.get(f"/models/{name}/truth")
    return res.json()


def get_model_ancestors(name: ModelResolutionName) -> list[ModelAncestor]:
    """Get the ancestors of a model in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Retrieving ancestors", prefix=log_prefix)

    res = CLIENT.get(f"/models/{name}/ancestors")
    return [ModelAncestor.model_validate(m) for m in res.json()]


def set_model_ancestors_cache(
    name: ModelResolutionName, ancestors: list[ModelAncestor]
) -> ResolutionOperationStatus:
    """Set the ancestors cache for a model in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Setting ancestors cached truth values", prefix=log_prefix)

    res = CLIENT.post(
        f"/models/{name}/ancestors_cache",
        json=[a.model_dump() for a in ancestors],
    )
    return ResolutionOperationStatus.model_validate(res.json())


def get_model_ancestors_cache(name: ModelResolutionName) -> list[ModelAncestor]:
    """Get the ancestors cache for a model in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Getting ancestors cached truth values", prefix=log_prefix)

    res = CLIENT.get(f"/models/{name}/ancestors_cache")
    return [ModelAncestor.model_validate(m) for m in res.json()]


def delete_model(
    name: ModelResolutionName, certain: bool = False
) -> ResolutionOperationStatus:
    """Delete a model in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Deleting", prefix=log_prefix)

    res = CLIENT.delete(f"/resolutions/{name}", params={"certain": certain})
    return ResolutionOperationStatus.model_validate(res.json())
