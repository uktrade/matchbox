from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from matchbox.common.graph import (
    ResolutionGraph,
)
from matchbox.server import app

client = TestClient(app)


class TestMatchboxAPI:
    def test_healthcheck(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

    @patch("matchbox.server.base.BackendManager.get_backend")
    def test_count_all_backend_items(self, get_backend):
        entity_counts = {
            "datasets": 1,
            "models": 2,
            "data": 3,
            "clusters": 4,
            "creates": 5,
            "merges": 6,
            "proposes": 7,
        }
        mock_backend = Mock()
        for e, c in entity_counts.items():
            mock_e = Mock()
            mock_e.count = Mock(return_value=c)
            setattr(mock_backend, e, mock_e)
        get_backend.return_value = mock_backend

        response = client.get("/testing/count")
        assert response.status_code == 200
        assert response.json() == {"entities": entity_counts}

    @patch("matchbox.server.base.BackendManager.get_backend")
    def test_count_backend_item(self, get_backend):
        mock_backend = Mock()
        mock_backend.models.count = Mock(return_value=20)
        get_backend.return_value = mock_backend

        response = client.get("/testing/count", params={"entity": "models"})
        assert response.status_code == 200
        assert response.json() == {"entities": {"models": 20}}

    # def test_clear_backend():
    #     response = client.post("/testing/clear")
    #     assert response.status_code == 200

    # def test_list_sources():
    #     response = client.get("/sources")
    #     assert response.status_code == 200

    # def test_get_source():
    #     response = client.get("/sources/test_source")
    #     assert response.status_code == 200

    # def test_add_source():
    #     response = client.post("/sources")
    #     assert response.status_code == 200

    # def test_list_models():
    #     response = client.get("/models")
    #     assert response.status_code == 200

    # def test_get_model():
    #     response = client.get("/models/test_model")
    #     assert response.status_code == 200

    # def test_add_model():
    #     response = client.post("/models")
    #     assert response.status_code == 200

    # def test_delete_model():
    #     response = client.delete("/models/test_model")
    #     assert response.status_code == 200

    # def test_get_results():
    #     response = client.get("/models/test_model/results")
    #     assert response.status_code == 200

    # def test_set_results():
    #     response = client.post("/models/test_model/results")
    #     assert response.status_code == 200

    # def test_get_truth():
    #     response = client.get("/models/test_model/truth")
    #     assert response.status_code == 200

    # def test_set_truth():
    #     response = client.post("/models/test_model/truth")
    #     assert response.status_code == 200

    # def test_get_ancestors():
    #     response = client.get("/models/test_model/ancestors")
    #     assert response.status_code == 200

    # def test_get_ancestors_cache():
    #     response = client.get("/models/test_model/ancestors_cache")
    #     assert response.status_code == 200

    # def test_set_ancestors_cache():
    #     response = client.post("/models/test_model/ancestors_cache")
    #     assert response.status_code == 200

    # def test_query():
    #     response = client.get("/query")
    #     assert response.status_code == 200

    # def test_validate_ids():
    #     response = client.get("/validate/id")
    #     assert response.status_code == 200

    @patch("matchbox.server.base.BackendManager.get_backend")
    def test_get_resolution_graph(self, get_backend, resolution_graph):
        mock_backend = Mock()
        mock_backend.get_resolution_graph = Mock(return_value=resolution_graph)
        get_backend.return_value = mock_backend

        response = client.get("/report/resolutions")
        assert response.status_code == 200
        assert ResolutionGraph.model_validate(response.json())
