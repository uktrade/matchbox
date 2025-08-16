import zipfile
from io import BytesIO
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES,
    SCHEMA_JUDGEMENTS,
    JudgementsZipFilenames,
)
from matchbox.common.dtos import BackendParameterType, BackendResourceType
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxNoJudgements,
    MatchboxResolutionNotFoundError,
    MatchboxTooManySamplesRequested,
    MatchboxUserNotFoundError,
)


def test_insert_judgement_ok(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test that a judgement is passed on to backend."""
    test_client, mock_backend, _ = api_client_and_mocks
    judgement = Judgement(user_id=1, shown=10, endorsed=[[1]])
    response = test_client.post("/eval/judgements", json=judgement.model_dump())
    assert response.status_code == 201
    assert (
        mock_backend.insert_judgement.call_args_list[0].kwargs["judgement"] == judgement
    )


def test_insert_judgement_error(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test that judgement insertion bubbles up errors."""
    test_client, mock_backend, _ = api_client_and_mocks
    fake_judgement = Judgement(user_id=1, shown=10, endorsed=[[1]]).model_dump()

    mock_backend.insert_judgement = Mock(side_effect=MatchboxDataNotFound)
    response = test_client.post("/eval/judgements", json=fake_judgement)
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.CLUSTER

    mock_backend.insert_judgement = Mock(side_effect=MatchboxUserNotFoundError)
    response = test_client.post("/eval/judgements", json=fake_judgement)
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.USER


def test_get_judgements(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test that all judgements can be retrieved."""
    judgements = pa.Table.from_pylist(
        [
            {"user_id": 1, "shown": 10, "endorsed": 11},
            {"user_id": 1, "shown": 10, "endorsed": 12},
        ],
        schema=SCHEMA_JUDGEMENTS,
    )
    # There will be nulls in case of a schema mismatch
    assert len(judgements.drop_null()) == len(judgements)

    expansion = pa.Table.from_pylist(
        [
            {"root": 10, "leaves": [1, 2, 3]},
            {"root": 11, "leaves": [1]},
            {"root": 12, "leaves": [2, 3]},
        ],
        schema=SCHEMA_CLUSTER_EXPANSION,
    )
    # There will be nulls in case of a schema mismatch
    assert len(expansion.drop_null()) == len(expansion)

    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.get_judgements = Mock(return_value=(judgements, expansion))

    # Process response
    response = test_client.get("/eval/judgements")
    zip_bytes = BytesIO(response.content)

    with zipfile.ZipFile(zip_bytes, "r") as zip_file:
        with zip_file.open(JudgementsZipFilenames.JUDGEMENTS) as f1:
            downloaded_judgements = pq.read_table(f1)

        with zip_file.open(JudgementsZipFilenames.EXPANSION) as f2:
            downloaded_expansion = pq.read_table(f2)

    # Check response
    assert response.status_code == 200
    assert downloaded_judgements.equals(judgements)
    assert downloaded_expansion.equals(expansion)


def test_compare_models_ok(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test errors in requesting samples."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_pr = {"a": (1, 0.5), "b": (0.5, 1)}
    mock_backend.compare_models = Mock(return_value=mock_pr)

    response = test_client.get(
        "/eval/compare",
        params={"resolutions": ["a", "b"]},
    )

    assert response.status_code == 200
    result = response.json()
    assert sorted(result.keys()) == ["a", "b"]
    assert tuple(result["a"]) == mock_pr["a"]
    assert tuple(result["b"]) == mock_pr["b"]


def test_compare_models_404(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test errors in requesting samples."""
    test_client, mock_backend, _ = api_client_and_mocks

    # Resolution not found
    mock_backend.compare_models = Mock(side_effect=MatchboxResolutionNotFoundError)
    response = test_client.get(
        "/eval/compare",
        params={"resolutions": ["a", "b", "c"]},
    )
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.RESOLUTION

    # No judgements available
    mock_backend.compare_models = Mock(side_effect=MatchboxNoJudgements)
    response = test_client.get(
        "/eval/compare",
        params={"resolutions": ["a", "b", "c"]},
    )
    assert response.status_code == 404
    assert response.json()["entity"] == BackendResourceType.JUDGEMENT


def test_get_samples(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test that samples can be requested."""
    sample = pa.Table.from_pylist(
        [
            {"root": 10, "leaf": 1, "key": "1", "source": "source_a"},
            {"root": 10, "leaf": 1, "key": "2", "source": "source_a"},
        ],
        schema=SCHEMA_EVAL_SAMPLES,
    )
    # There will be nulls in case of a schema mismatch
    assert len(sample.drop_null()) == len(sample)

    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.sample_for_eval = Mock(return_value=sample)

    # Process response
    response = test_client.get(
        "/eval/samples",
        params={"n": 10, "resolution": "resolution", "user_id": 12},
    )
    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)

    # Check response
    assert response.status_code == 200
    assert table.equals(sample)


@pytest.mark.parametrize(
    ("exception", "entity"),
    [
        pytest.param(
            MatchboxUserNotFoundError,
            BackendResourceType.USER,
            id="user_not_found",
        ),
        pytest.param(
            MatchboxResolutionNotFoundError,
            BackendResourceType.RESOLUTION,
            id="resolution_not_found",
        ),
    ],
)
def test_get_samples_404(
    exception: BaseException,
    entity: BackendResourceType,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
):
    """Test errors in requesting samples."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.sample_for_eval = Mock(side_effect=exception)

    response = test_client.get(
        "/eval/samples",
        params={"n": 10, "resolution": "resolution", "user_id": 12},
    )

    assert response.status_code == 404
    assert response.json()["entity"] == entity


def test_get_samples_422(api_client_and_mocks: tuple[TestClient, Mock, Mock]):
    """Test errors in requesting samples."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.sample_for_eval = Mock(side_effect=MatchboxTooManySamplesRequested)

    response = test_client.get(
        "/eval/samples",
        params={"n": 10, "resolution": "resolution", "user_id": 12},
    )

    assert response.status_code == 422
    assert response.json()["parameter"] == BackendParameterType.SAMPLE_SIZE
