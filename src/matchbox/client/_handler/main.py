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
    ErrorResponse,
    OKMessage,
)
from matchbox.common.exceptions import (
    HTTP_EXCEPTION_REGISTRY,
    MatchboxHttpException,
    MatchboxUnhandledServerResponse,
)
from matchbox.common.hash import hash_to_base64

URLEncodeHandledType = str | int | float | bytes | StrEnum | None


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
) -> str | list[str] | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    # Also covers bool (subclass of int)
    if isinstance(v, StrEnum | int | float):
        return str(v)
    elif isinstance(v, bytes):
        return hash_to_base64(v)
    # Needs to be at the end, so we don't apply it to e.g. strings
    if isinstance(v, Iterable):
        return [
            str(encoded)
            for item in v
            if (encoded := encode_param_value(item)) is not None
        ]
    raise ValueError(f"It was not possible to parse {v} as an URL parameter")


def url_params(
    params: dict[str, URLEncodeHandledType | Iterable[URLEncodeHandledType]],
) -> dict[str, str | list[str]]:
    """Prepares a dictionary of parameters to be encoded in a URL.

    Removes None values.
    """
    encoded_params = {}
    for k, v in params.items():
        encoded_v = encode_param_value(v)
        if encoded_v is not None:
            encoded_params[k] = encoded_v
    return encoded_params


def reconstruct_exception(
    ExceptionClass: type[MatchboxHttpException], error: ErrorResponse
) -> MatchboxHttpException:
    """Reconstruct an exception from ErrorResponse data."""
    # Handle exceptions with special constructor signatures
    if error.details:
        try:
            return ExceptionClass(message=error.message, **error.details)
        except TypeError:
            pass

    # Default: just pass the message
    return ExceptionClass(error.message)


def handle_http_code(res: httpx.Response) -> httpx.Response:
    """Handle HTTP status codes and raise appropriate exceptions."""
    res.read()

    if 299 >= res.status_code >= 200:
        return res

    try:
        data = res.json()

        if "exception_type" in data:
            error = ErrorResponse.model_validate(data)
            ExceptionClass = HTTP_EXCEPTION_REGISTRY.get(error.exception_type)

            if ExceptionClass:
                raise reconstruct_exception(ExceptionClass, error)
            raise MatchboxUnhandledServerResponse(
                http_status=res.status_code,
                details=f"Unknown exception type: {error.exception_type}",
            )

        raise MatchboxUnhandledServerResponse(
            http_status=res.status_code, details=str(res.content)
        )

    except ValidationError as e:
        raise MatchboxUnhandledServerResponse(
            http_status=res.status_code, details=str(res.content)
        ) from e


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
