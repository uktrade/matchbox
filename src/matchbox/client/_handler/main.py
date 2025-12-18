"""Functions abstracting the interaction with the server API."""

from collections.abc import Iterable
from enum import StrEnum
from importlib.metadata import version

import httpx
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from matchbox.client._settings import ClientSettings, settings
from matchbox.common.dtos import (
    BackendParameterType,
    BackendResourceType,
    NotFoundError,
    OKMessage,
    ResourceOperationStatus,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxRunNotFoundError,
    MatchboxTooManySamplesRequested,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
    MatchboxUserNotFoundError,
)
from matchbox.common.hash import hash_to_base64

URLEncodeHandledType = str | int | float | bytes | StrEnum


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
    # Also covers bool (subclass of int)
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
    return headers


CLIENT = create_client(settings=settings)


@http_retry
def healthcheck() -> OKMessage:
    """Checks the health of the Matchbox server."""
    return OKMessage.model_validate(CLIENT.get("/health").json())
