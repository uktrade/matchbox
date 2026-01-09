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
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxResolutionNotFoundError,
    MatchboxTooManySamplesRequested,
    MatchboxUserNotFoundError,
)


def test_insert_judgement_ok(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test that a judgement is passed on to backend."""
    test_client, mock_backend, _ = api_client_and_mocks
    judgement = Judgement(user_name="alice", shown=[1, 2], endorsed=[[1], [2]])
    response = test_client.post("/eval/judgements", json=judgement.model_dump())
    assert response.status_code == 201
    assert (
        mock_backend.insert_judgement.call_args_list[0].kwargs["judgement"] == judgement
    )


def test_insert_judgement_error(
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test that judgement insertion bubbles up errors."""
    test_client, mock_backend, _ = api_client_and_mocks
    fake_judgement = Judgement(
        user_name="alice", shown=[1, 2], endorsed=[[1], [2]]
    ).model_dump()

    mock_backend.insert_judgement = Mock(side_effect=MatchboxDataNotFound)
    response = test_client.post("/eval/judgements", json=fake_judgement)
    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxDataNotFound"

    mock_backend.insert_judgement = Mock(side_effect=MatchboxUserNotFoundError)
    response = test_client.post("/eval/judgements", json=fake_judgement)
    assert response.status_code == 404
    assert response.json()["exception_type"] == "MatchboxUserNotFoundError"


def test_get_judgements(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test that all judgements can be retrieved."""
    judgements = pa.Table.from_pylist(
        [
            {"user_name": "alice", "shown": 10, "endorsed": 11},
            {"user_name": "alice", "shown": 10, "endorsed": 12},
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


def test_get_samples(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
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
        params={
            "collection": "test_collection",
            "run_id": 1,
            "resolution": "a",
            "n": 10,
            "user_name": "alice",
        },
    )
    assert response.status_code == 200

    buffer = BytesIO(response.content)
    table = pq.read_table(buffer)
    assert table.equals(sample)


@pytest.mark.parametrize(
    ("exception", "expected_exception_name"),
    [
        pytest.param(
            MatchboxUserNotFoundError,
            "MatchboxUserNotFoundError",
            id="user_not_found",
        ),
        pytest.param(
            MatchboxResolutionNotFoundError,
            "MatchboxResolutionNotFoundError",
            id="resolution_not_found",
        ),
    ],
)
def test_get_samples_404(
    exception: type[Exception],
    expected_exception_name: str,
    api_client_and_mocks: tuple[TestClient, Mock, Mock],
) -> None:
    """Test errors in requesting samples."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.sample_for_eval = Mock(side_effect=exception)

    response = test_client.get(
        "/eval/samples",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "resolution": "a",
            "n": 10,
            "user_name": "alice",
        },
    )

    assert response.status_code == 404
    assert response.json()["exception_type"] == expected_exception_name


def test_get_samples_422(api_client_and_mocks: tuple[TestClient, Mock, Mock]) -> None:
    """Test errors in requesting samples."""
    test_client, mock_backend, _ = api_client_and_mocks
    mock_backend.sample_for_eval = Mock(side_effect=MatchboxTooManySamplesRequested)

    response = test_client.get(
        "/eval/samples",
        params={
            "collection": "test_collection",
            "run_id": 1,
            "resolution": "a",
            "n": 10,
            "user_name": "alice",
        },
    )

    assert response.status_code == 422
    assert response.json()["exception_type"] == "MatchboxTooManySamplesRequested"
