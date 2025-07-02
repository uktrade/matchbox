from unittest.mock import Mock

from fastapi.testclient import TestClient

from matchbox.common.eval import Judgement
from matchbox.common.exceptions import MatchboxDataNotFound, MatchboxUserNotFoundError
from matchbox.server.api.dependencies import backend
from matchbox.server.api.main import app


def test_insert_judgement_ok(test_client: TestClient):
    """Test that a judgement is passed on to backend."""
    mock_backend = Mock()

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend
    judgement = Judgement(user_id=1, clusters=[[1]])
    response = test_client.post("/eval/", json=judgement.model_dump())
    assert response.status_code == 201
    assert (
        mock_backend.insert_judgement.call_args_list[0].kwargs["judgement"] == judgement
    )


def test_insert_judgement_error(test_client: TestClient):
    mock_backend = Mock()
    mock_backend.insert_judgement = Mock(side_effect=MatchboxDataNotFound)

    # Override app dependencies with mocks
    app.dependency_overrides[backend] = lambda: mock_backend

    response = test_client.post(
        "/eval/", json=Judgement(user_id=1, clusters=[[1]]).model_dump()
    )
    assert response.status_code == 422
    assert response.json()["entity"] == "clusters"

    mock_backend.insert_judgement = Mock(side_effect=MatchboxUserNotFoundError)
    response = test_client.post(
        "/eval/", json=Judgement(user_id=1, clusters=[[1]]).model_dump()
    )
    assert response.status_code == 422
    assert response.json()["entity"] == "users"
