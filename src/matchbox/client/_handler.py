"""Functions abstracting the interaction with the server API."""

import time
import zipfile
from collections.abc import Iterable
from importlib.metadata import version
from io import BytesIO

import httpx
from pyarrow import Table
from pyarrow.parquet import read_table

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
    LoginAttempt,
    LoginResult,
    ModelAncestor,
    ModelConfig,
    NotFoundError,
    ResolutionOperationStatus,
    UploadStage,
    UploadStatus,
)
from matchbox.common.eval import Judgement, ModelComparison
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxSourceNotFoundError,
    MatchboxTooManySamplesRequested,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
    MatchboxUserNotFoundError,
)
from matchbox.common.graph import (
    ModelResolutionName,
    ResolutionGraph,
    ResolutionName,
    SourceResolutionName,
)
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
        error = NotFoundError.model_validate(res.json())
        match error.entity:
            case BackendResourceType.SOURCE:
                raise MatchboxSourceNotFoundError(error.details)
            case BackendResourceType.RESOLUTION:
                raise MatchboxResolutionNotFoundError(error.details)
            case BackendResourceType.CLUSTER:
                raise MatchboxDataNotFound(error.details)
            case BackendResourceType.USER:
                raise MatchboxUserNotFoundError(error.details)
            case _:
                raise RuntimeError(f"Unexpected 404 error: {error.details}")

    if res.status_code == 409:
        error = ResolutionOperationStatus.model_validate(res.json())
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


def login(user_name: str) -> int:
    logger.debug(f"Log in attempt for {user_name}")
    response = CLIENT.post(
        "/login", json=LoginAttempt(user_name=user_name).model_dump()
    )
    return LoginResult.model_validate(response.json()).user_id


# Retrieval


def query(
    source: SourceResolutionName,
    return_leaf_id: bool,
    resolution: ResolutionName | None = None,
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
                "source": source,
                "resolution": resolution,
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

    return table


def match(
    targets: list[SourceResolutionName],
    source: SourceResolutionName,
    key: str,
    resolution: ResolutionName,
    threshold: int | None = None,
) -> Match:
    """Match a source against a list of targets."""
    log_prefix = f"Query {source}"
    logger.debug(
        f"{key} to {', '.join(targets)} using {resolution}",
        prefix=log_prefix,
    )

    res = CLIENT.get(
        "/match",
        params=url_params(
            {
                "targets": targets,
                "source": source,
                "key": key,
                "resolution": resolution,
                "threshold": threshold,
            }
        ),
    )

    logger.debug("Finished", prefix=log_prefix)

    return [Match.model_validate(m) for m in res.json()]


# Data management


def index(source_config: SourceConfig, data_hashes: Table) -> UploadStatus:
    """Index from a SourceConfig in Matchbox."""
    log_prefix = f"Index {source_config.name}"

    buffer = table_to_buffer(table=data_hashes)

    # Upload metadata
    logger.debug("Uploading metadata", prefix=log_prefix)

    metadata_res = CLIENT.post("/sources", json=source_config.model_dump(mode="json"))

    upload = UploadStatus.model_validate(metadata_res.json())

    # Upload data
    logger.debug("Uploading data", prefix=log_prefix)

    upload_res = CLIENT.post(
        f"/upload/{upload.id}",
        files={"file": (f"{upload.id}.parquet", buffer, "application/octet-stream")},
    )

    # Poll until complete with retry/timeout configuration
    status = UploadStatus.model_validate(upload_res.json())
    while status.stage not in [UploadStage.COMPLETE, UploadStage.FAILED]:
        status_res = CLIENT.get(f"/upload/{upload.id}/status")
        status = UploadStatus.model_validate(status_res.json())

        logger.debug(f"Uploading data: {status.stage}", prefix=log_prefix)

        if status.stage == UploadStage.FAILED:
            raise MatchboxServerFileError(status.details)

        time.sleep(settings.retry_delay)

    logger.debug("Finished")

    return status


def get_source_config(name: SourceResolutionName) -> SourceConfig:
    log_prefix = f"SourceConfig {name}"
    logger.debug("Retrieving", prefix=log_prefix)

    res = CLIENT.get(f"/sources/{name}")

    return SourceConfig.model_validate(res.json())


def get_resolution_source_configs(name: ModelResolutionName) -> list[SourceConfig]:
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


def insert_model(model_config: ModelConfig) -> ResolutionOperationStatus:
    """Insert a model in Matchbox."""
    log_prefix = f"Model {model_config.name}"
    logger.debug("Inserting metadata", prefix=log_prefix)

    res = CLIENT.post("/models", json=model_config.model_dump())
    return ResolutionOperationStatus.model_validate(res.json())


def get_model(name: ModelResolutionName) -> ModelConfig | None:
    """Get model metadata from Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Retrieving metadata", prefix=log_prefix)

    try:
        res = CLIENT.get(f"/models/{name}")
        return ModelConfig.model_validate(res.json())
    except MatchboxResolutionNotFoundError:
        return None


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
    while status.stage not in [UploadStage.COMPLETE, UploadStage.FAILED]:
        status_res = CLIENT.get(f"/upload/{upload.id}/status")
        status = UploadStatus.model_validate(status_res.json())

        logger.debug(f"Uploading data: {status.stage}", prefix=log_prefix)

        if status.stage == UploadStage.FAILED:
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


def delete_resolution(
    name: ModelResolutionName, certain: bool = False
) -> ResolutionOperationStatus:
    """Delete a resolution in Matchbox."""
    log_prefix = f"Model {name}"
    logger.debug("Deleting", prefix=log_prefix)

    res = CLIENT.delete(f"/resolutions/{name}", params={"certain": certain})
    return ResolutionOperationStatus.model_validate(res.json())


# Evaluation


def sample_for_eval(n: int, resolution: ModelResolutionName, user_id: int) -> Table:
    res = CLIENT.get(
        "/eval/samples",
        params=url_params({"n": n, "resolution": resolution, "user_id": user_id}),
    )

    return read_table(BytesIO(res.content))


def compare_models(resolutions: list[ModelResolutionName]) -> ModelComparison:
    res = CLIENT.get("/eval/compare", params=url_params({"resolutions": resolutions}))
    scores = {resolution: tuple(pr) for resolution, pr in res.json().items()}
    return scores


def send_eval_judgement(judgement: Judgement) -> None:
    logger.debug(
        f"Submitting judgement {judgement.shown}:{judgement.endorsed} "
        f"for {judgement.user_id}"
    )
    CLIENT.post("/eval/judgements", json=judgement.model_dump())


def download_eval_data() -> tuple[Table, Table]:
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

    return judgements, expansion


# Admin


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
