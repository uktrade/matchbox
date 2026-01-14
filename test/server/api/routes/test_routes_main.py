from importlib.metadata import version
from io import BytesIO
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from matchbox.common.arrow import SCHEMA_QUERY
from matchbox.common.dtos import (
    Match,
    OKMessage,
    SourceResolutionPath,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxRunNotFoundError,
    MatchboxUnhandledServerResponse,
)

# General


def test_healthcheck(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test the healthcheck endpoint."""
    test_client, _, _ = api_client_and_mocks
    response = test_client.get("/health")
    assert response.status_code == 200
    response = OKMessage.model_validate(response.json())
    assert response.status == "OK"
    assert response.version == version("matchbox-db")


# Retrieval


def test_query(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.query = Mock(
        return_value=pa.Table.from_pylist(
            [
                {"keys": "a", "id": 1},
                {"keys": "b", "id": 2},
            ],
            schema=SCHEMA_QUERY,
        )
    )

    response = test_client.get(
        "/query",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "source": "foo",
            "return_leaf_id": False,
        },
    )

    assert response.status_code == 200

    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    assert table.schema.equals(SCHEMA_QUERY)


def test_query_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    test_client, mock_backend, _ = api_client_and_mocks

    mock_backend.query = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.get(
        "/query",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "source": "foo",
            "return_leaf_id": True,
        },
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxCollectionNotFoundError"

    mock_backend.query = Mock(side_effect=MatchboxRunNotFoundError())

    response = test_client.get(
        "/query",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "source": "foo",
            "return_leaf_id": True,
        },
    )
    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxRunNotFoundError"

    mock_backend.query = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.get(
        "/query",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "source": "foo",
            "return_leaf_id": True,
        },
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxResolutionNotFoundError"


def test_match(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_matches = [
        Match(
            cluster=1,
            source=SourceResolutionPath(
                collection="test_collection", name="foo", run=1
            ),
            source_id={"1"},
            target=SourceResolutionPath(
                collection="test_collection", name="bar", run=1
            ),
            target_id={"a"},
        )
    ]
    mock_backend.match = Mock(return_value=mock_matches)

    response = test_client.get(
        "/match",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "targets": "foo",
            "source": "bar",
            "key": 1,
            "resolution": "res",
            "threshold": 50,
        },
    )

    assert response.status_code == 200
    [Match.model_validate(m) for m in response.json()]


def test_match_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    test_client, mock_backend, _ = api_client_and_mocks

    mock_backend.match = Mock(side_effect=MatchboxCollectionNotFoundError())

    response = test_client.get(
        "/match",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "targets": ["foo"],
            "source": "bar",
            "key": 1,
            "resolution": "res",
        },
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxCollectionNotFoundError"

    mock_backend.match = Mock(side_effect=MatchboxRunNotFoundError())

    response = test_client.get(
        "/match",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "targets": ["foo"],
            "source": "bar",
            "key": 1,
            "resolution": "res",
        },
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxRunNotFoundError"

    mock_backend.match = Mock(side_effect=MatchboxResolutionNotFoundError())

    response = test_client.get(
        "/match",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "targets": ["foo"],
            "source": "bar",
            "key": 1,
            "resolution": "res",
        },
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxResolutionNotFoundError"


# Admin


def test_count_all_backend_items(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test the unparameterised entity counting endpoint."""
    test_client, mock_backend, _ = api_client_and_mocks
    entity_counts = {
        "sources": 1,
        "models": 2,
        "source_clusters": 3,
        "model_clusters": 4,
        "all_clusters": 7,
        "creates": 6,
        "merges": 7,
        "proposes": 8,
    }
    for e, c in entity_counts.items():
        mock_e = Mock()
        mock_e.count = Mock(return_value=c)
        setattr(mock_backend, e, mock_e)

    response = test_client.get("/database/count")
    assert response.status_code == 200
    assert response.json() == {"entities": entity_counts}


def test_count_backend_item(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test the parameterised entity counting endpoint."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.models.count = Mock(return_value=20)

    response = test_client.get("/database/count", params={"entity": "models"})
    assert response.status_code == 200
    assert response.json() == {"entities": {"models": 20}}


def test_clear_backend_ok(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.clear = Mock()

    response = test_client.delete("/database", params={"certain": "true"})
    assert response.status_code == 200
    OKMessage.model_validate(response.json())


def test_clear_backend_errors(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.clear = Mock(side_effect=MatchboxDeletionNotConfirmed)

    response = test_client.delete("/database")
    assert response.status_code == 409
    # We send some explanatory message
    assert response.content


def test_delete_orphans_ok(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_orphans = Mock(return_value=42)

    response = test_client.delete("/database/orphans")
    assert response.status_code == 200
    OKMessage.model_validate(response.json())
    assert response.json()["details"] == "Deleted 42 orphans."


def test_delete_orphans_errors(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.delete_orphans = Mock(side_effect=MatchboxUnhandledServerResponse)

    response = test_client.delete("/database/orphans")
    assert response.status_code == 500
    # We send some explanatory message
    assert response.content
