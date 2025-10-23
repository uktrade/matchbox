"""Functions abstracting the interaction with the server API."""

import time
import zipfile
from collections.abc import Iterable
from enum import StrEnum
from importlib.metadata import version
from io import BytesIO

import httpx
import polars as pl
from pyarrow import Table
from pyarrow.parquet import read_table
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from matchbox.client._settings import ClientSettings, settings
from matchbox.client.authorisation import generate_json_web_token
from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_JUDGEMENTS,
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
    JudgementsZipFilenames,
    check_schema,
    table_to_buffer,
)
from matchbox.common.dtos import (
    BackendCountableType,
    BackendParameterType,
    BackendResourceType,
    Collection,
    CollectionName,
    LoginAttempt,
    LoginResult,
    Match,
    ModelResolutionPath,
    NotFoundError,
    OKMessage,
    Resolution,
    ResolutionPath,
    ResolutionType,
    ResourceOperationStatus,
    Run,
    RunID,
    SourceResolutionPath,
    UploadStage,
    UploadStatus,
)
from matchbox.common.eval import Judgement, ModelComparison
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxEmptyServerResponse,
    MatchboxResolutionNotFoundError,
    MatchboxRunNotFoundError,
    MatchboxServerFileError,
    MatchboxTooManySamplesRequested,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
    MatchboxUserNotFoundError,
)
from matchbox.common.hash import hash_to_base64
from matchbox.common.logging import logger

URLEncodeHandledType = str | int | float | bytes


# Retry configuration for HTTP operations
http_retry = retry(
    stop=stop_after_attempt(5),  # Try up to 5 times
    wait=wait_exponential(
        multiplier=1, min=1, max=180
    ),  # Exponential backoff: 1s, 2s, 4s, 8s, up to 3 minutes
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)
    ),
)


def encode_param_value(
    v: URLEncodeHandledType | Iterable[URLEncodeHandledType],
) -> str | list[str]:
    if isinstance(v, str):
        return v
    if isinstance(v, StrEnum | int | float):
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
    non_null = {k: v for k, v in params.items() if v is not None}
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
        try:
            error = NotFoundError.model_validate(res.json())
        # Validation will fail if endpoint does not exist
        except ValidationError as e:
            raise RuntimeError(f"Error with request {res._request}: {res}") from e

        match error.entity:
            case BackendResourceType.COLLECTION:
                raise MatchboxCollectionNotFoundError(error.details)
            case BackendResourceType.RUN:
                raise MatchboxRunNotFoundError(error.details)
            case BackendResourceType.RESOLUTION:
                raise MatchboxResolutionNotFoundError(error.details)
            case BackendResourceType.CLUSTER:
                raise MatchboxDataNotFound(error.details)
            case BackendResourceType.USER:
                raise MatchboxUserNotFoundError(error.details)
            case _:
                raise RuntimeError(f"Unexpected 404 error: {error.details}")

    if res.status_code == 409:
        error = ResourceOperationStatus.model_validate(res.json())
        raise MatchboxDeletionNotConfirmed(message=error.details)

    if res.status_code == 422:
        match res.json().get("parameter"):
            case BackendParameterType.SAMPLE_SIZE:
                raise MatchboxTooManySamplesRequested(res.content)
            case _:
                # Not a custom Matchbox exception, most likely a Pydantic error
                raise MatchboxUnparsedClientRequest(res.content)

    raise MatchboxUnhandledServerResponse(
        details=res.content, http_status=res.status_code
    )


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
    if settings.jwt:
        headers["Authorization"] = settings.jwt
    elif settings.user and settings.private_key:
        headers["Authorization"] = generate_json_web_token(sub=settings.user)
    return headers


CLIENT = create_client(settings=settings)


@http_retry
def healthcheck() -> OKMessage:
    """Checks the health of the Matchbox server."""
    return OKMessage.model_validate(CLIENT.get("/health").json())


@http_retry
def login(user_name: str) -> int:
    """Log into Matchbox and return the user ID."""
    logger.debug(f"Log in attempt for {user_name}")
    response = CLIENT.post(
        "/login", json=LoginAttempt(user_name=user_name).model_dump()
    )
    return LoginResult.model_validate(response.json()).user_id


# Retrieval


@http_retry
def query(
    source: SourceResolutionPath,
    return_leaf_id: bool,
    resolution: ResolutionPath | None = None,
    threshold: int | None = None,
    limit: int | None = None,
) -> Table:
    """Query a source in Matchbox."""
    log_prefix = f"Query {source}"
    logger.debug(f"Using {resolution}", prefix=log_prefix)

    res = CLIENT.get(
        "/query",
        params=url_params(
            {
                "collection": source.collection,
                "run_id": source.run,
                "source": source.name,
                "resolution": resolution.name if resolution else None,
                "return_leaf_id": return_leaf_id,
                "threshold": threshold,
                "limit": limit,
            }
        ),
    )

    buffer = BytesIO(res.content)
    table = read_table(buffer)

    logger.debug("Finished", prefix=log_prefix)

    expected_schema = SCHEMA_QUERY
    if return_leaf_id:
        expected_schema = SCHEMA_QUERY_WITH_LEAVES

    check_schema(expected_schema, table.schema)

    if table.num_rows == 0:
        raise MatchboxEmptyServerResponse(operation="query")

    return table


@http_retry
def match(
    targets: list[SourceResolutionPath],
    source: SourceResolutionPath,
    key: str,
    resolution: ResolutionPath,
    threshold: int | None = None,
) -> list[Match]:
    """Match a source against a list of targets."""
    log_prefix = f"Query {source}"
    logger.debug(
        f"{key} to {', '.join(str(targets))} using {resolution}",
        prefix=log_prefix,
    )

    res = CLIENT.get(
        "/match",
        params=url_params(
            {
                "collection": resolution.collection,
                "run_id": resolution.run,
                "targets": [t.name for t in targets],
                "source": source.name,
                "key": key,
                "resolution": resolution.name,
                "threshold": threshold,
            }
        ),
    )

    logger.debug("Finished", prefix=log_prefix)

    matches = [Match.model_validate(m) for m in res.json()]

    if not matches:
        raise MatchboxEmptyServerResponse(operation="match")

    return matches


# Collection management


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

    res = CLIENT.post(
        f"/collections/{name}",
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
def get_resolution(
    path: ResolutionPath, validate_type: ResolutionType | None = None
) -> Resolution | None:
    """Get a resolution from Matchbox."""
    log_prefix = f"Resolution {path}"
    logger.debug("Retrieving metadata", prefix=log_prefix)

    res = CLIENT.get(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}",
        params=url_params({"validate_type": validate_type}),
    )
    return Resolution.model_validate(res.json())


@http_retry
def set_data(
    path: ResolutionPath, data: pl.DataFrame | Table, validate_type: ResolutionType
) -> UploadStatus:
    """Upload source hashes or model results to server."""
    log_prefix = f"Resolution {path}"
    logger.debug("Uploading results", prefix=log_prefix)

    data_arrow = data.to_arrow() if isinstance(data, pl.DataFrame) else data
    buffer = table_to_buffer(table=data_arrow)

    # Initialise upload
    metadata_res = CLIENT.post(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/data",
        params=url_params({"validate_type": validate_type}),
    )

    upload = UploadStatus.model_validate(metadata_res.json())

    # Upload data
    upload_res = CLIENT.post(
        f"/upload/{upload.id}",
        files={"file": (f"{upload.id}.parquet", buffer, "application/octet-stream")},
    )

    logger.debug("Uploading data", prefix=log_prefix)

    # Poll until complete with retry/timeout configuration
    status = UploadStatus.model_validate(upload_res.json())
    while status.stage not in [UploadStage.COMPLETE, UploadStage.FAILED]:
        status_res = CLIENT.get(f"/upload/{upload.id}/status")
        status = UploadStatus.model_validate(status_res.json())

        logger.debug(f"Uploading data: {status.stage}", prefix=log_prefix)

        if status.stage == UploadStage.FAILED:
            raise MatchboxServerFileError(status.details)

        time.sleep(settings.retry_delay)

    logger.debug("Finished", prefix=log_prefix)

    return status


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
def set_truth(path: ModelResolutionPath, truth: int) -> ResourceOperationStatus:
    """Set the truth threshold for a model in Matchbox."""
    log_prefix = f"Model {path}"
    logger.debug("Setting truth value", prefix=log_prefix)

    res = CLIENT.patch(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/truth",
        json=truth,
    )
    return ResourceOperationStatus.model_validate(res.json())


@http_retry
def get_truth(path: ModelResolutionPath) -> int:
    """Get the truth threshold for a model in Matchbox."""
    log_prefix = f"Model {path}"
    logger.debug("Retrieving truth value", prefix=log_prefix)

    res = CLIENT.get(
        f"/collections/{path.collection}/runs/{path.run}/resolutions/{path.name}/truth"
    )
    return res.json()


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


# Evaluation


@http_retry
def sample_for_eval(n: int, resolution: ModelResolutionPath, user_id: int) -> Table:
    """Sample model results for evaluation."""
    res = CLIENT.get(
        "/eval/samples",
        params=url_params(
            {
                "n": n,
                "collection": resolution.collection,
                "run_id": resolution.run,
                "resolution": resolution.name,
                "user_id": user_id,
            }
        ),
    )

    return read_table(BytesIO(res.content))


@http_retry
def compare_models(
    resolutions: list[ModelResolutionPath],
) -> ModelComparison:
    """Get a model comparison for a set of model resolutions."""
    res = CLIENT.post("/eval/compare", json=[r.model_dump() for r in resolutions])
    scores = {resolution: tuple(pr) for resolution, pr in res.json().items()}
    return scores


@http_retry
def send_eval_judgement(judgement: Judgement) -> None:
    """Send judgements to the server."""
    logger.debug(
        f"Submitting judgement {judgement.shown}:{judgement.endorsed} "
        f"for {judgement.user_id}"
    )
    CLIENT.post("/eval/judgements", json=judgement.model_dump())


@http_retry
def download_eval_data() -> tuple[Table, Table]:
    """Download all judgements from the server."""
    logger.debug("Retrieving all judgements.")
    res = CLIENT.get("/eval/judgements")

    zip_bytes = BytesIO(res.content)
    with zipfile.ZipFile(zip_bytes, "r") as zip_file:
        with zip_file.open(JudgementsZipFilenames.JUDGEMENTS) as f1:
            judgements = read_table(f1)

        with zip_file.open(JudgementsZipFilenames.EXPANSION) as f2:
            expansion = read_table(f2)

    logger.debug("Finished retrieving judgements.")

    check_schema(SCHEMA_JUDGEMENTS, judgements.schema)
    check_schema(SCHEMA_CLUSTER_EXPANSION, expansion.schema)

    return pl.from_arrow(judgements), pl.from_arrow(expansion)


# Admin


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

    params = {"entity": entity} if entity else {}
    res = CLIENT.get("/database/count", params=url_params(params))

    counts = res.json()
    logger.debug(f"Counts: {counts}", prefix=log_prefix)

    return counts
