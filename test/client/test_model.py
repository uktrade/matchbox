import json
from os import getenv
from unittest.mock import Mock

import pytest
from httpx import Response
from pandas import DataFrame
from respx.router import MockRouter

from matchbox.client.models.models import Model
from matchbox.client.results import Results
from matchbox.common.arrow import SCHEMA_RESULTS, table_to_buffer
from matchbox.common.dtos import (
    BackendRetrievableType,
    BackendUploadType,
    ModelAncestor,
    ModelOperationStatus,
    ModelOperationType,
    NotFoundError,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxConfirmDelete,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
)
from matchbox.common.factories.models import model_factory


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_insert_model(respx_mock: MockRouter):
    """Test inserting a model via the API."""
    # Create test model using factory
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the POST /models endpoint
    route = respx_mock.post("/models").mock(
        return_value=Response(
            200,
            json=ModelOperationStatus(
                success=True,
                model_name=dummy.model.name,
                operation=ModelOperationType.INSERT,
            ).model_dump(),
        )
    )

    # Call insert_model
    model.insert_model()

    # Verify the API call
    assert route.called
    assert route.calls.last.request.content.decode() == dummy.model.model_dump_json()


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_insert_model_error(respx_mock: MockRouter):
    """Test handling of model insertion errors."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the POST /models endpoint with an error response
    route = respx_mock.post("/models").mock(
        return_value=Response(
            500,
            json=ModelOperationStatus(
                success=False,
                model_name=dummy.model.name,
                operation=ModelOperationType.INSERT,
                details="Internal server error",
            ).model_dump(),
        )
    )

    # Call insert_model and verify it raises an exception
    with pytest.raises(MatchboxUnhandledServerResponse, match="Internal server error"):
        model.insert_model()

    assert route.called


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_results_getter(respx_mock: MockRouter):
    """Test getting model results via the API."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the GET /models/{name}/results endpoint
    route = respx_mock.get(f"/models/{dummy.model.name}/results").mock(
        return_value=Response(200, content=table_to_buffer(dummy.data).read())
    )

    # Get results
    results = model.results

    # Verify the API call
    assert route.called
    assert isinstance(results, Results)
    assert results.probabilities.schema.equals(SCHEMA_RESULTS)


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_results_getter_not_found(respx_mock: MockRouter):
    """Test getting model results when they don't exist."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the GET endpoint with a 404 response
    route = respx_mock.get(f"/models/{dummy.model.name}/results").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Results not found", entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        )
    )

    # Verify that accessing results raises an exception
    with pytest.raises(MatchboxResolutionNotFoundError, match="Results not found"):
        _ = model.results

    assert route.called


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_results_setter(respx_mock: MockRouter):
    """Test setting model results via the API."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the endpoints needed for results upload
    init_route = respx_mock.post(f"/models/{dummy.model.name}/results").mock(
        return_value=Response(
            200,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.RESULTS,
            ).model_dump(),
        )
    )

    upload_route = respx_mock.post("/upload/test-upload-id").mock(
        return_value=Response(
            200,
            json=UploadStatus(
                id="test-upload-id",
                status="processing",
                entity=BackendUploadType.RESULTS,
            ).model_dump(),
        )
    )

    status_route = respx_mock.get("/upload/test-upload-id/status").mock(
        return_value=Response(
            200,
            json=UploadStatus(
                id="test-upload-id",
                status="complete",
                entity=BackendUploadType.RESULTS,
            ).model_dump(),
        )
    )

    # Set results
    test_results = Results(probabilities=dummy.data, metadata=dummy.model)
    model.results = test_results

    # Verify API calls
    assert init_route.called
    assert upload_route.called
    assert status_route.called
    assert (
        b"PAR1" in upload_route.calls.last.request.content
    )  # Check for parquet file signature


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_results_setter_upload_failure(respx_mock: MockRouter):
    """Test handling of upload failures when setting results."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the initial POST endpoint
    init_route = respx_mock.post(f"/models/{dummy.model.name}/results").mock(
        return_value=Response(
            200,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.RESULTS,
            ).model_dump(),
        )
    )

    # Mock the upload endpoint with a failure
    upload_route = respx_mock.post("/upload/test-upload-id").mock(
        return_value=Response(
            400,
            json=UploadStatus(
                id="test-upload-id",
                status="failed",
                entity=BackendUploadType.RESULTS,
                details="Invalid data format",
            ).model_dump(),
        )
    )

    # Attempt to set results and verify it raises an exception
    test_results = Results(probabilities=dummy.data, metadata=dummy.model)
    with pytest.raises(MatchboxServerFileError, match="Invalid data format"):
        model.results = test_results

    assert init_route.called
    assert upload_route.called


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_truth_getter(respx_mock: MockRouter):
    """Test getting model truth threshold via the API."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the GET /models/{name}/truth endpoint
    route = respx_mock.get(f"/models/{dummy.model.name}/truth").mock(
        return_value=Response(200, json=0.9)
    )

    # Get truth
    truth = model.truth

    # Verify the API call
    assert route.called
    assert truth == 0.9


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_truth_setter(respx_mock: MockRouter):
    """Test setting model truth threshold via the API."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the POST /models/{name}/truth endpoint
    route = respx_mock.post(f"/models/{dummy.model.name}/truth").mock(
        return_value=Response(
            200,
            json=ModelOperationStatus(
                success=True,
                model_name=dummy.model.name,
                operation=ModelOperationType.UPDATE_TRUTH,
            ).model_dump(),
        )
    )

    # Set truth
    model.truth = 0.9

    # Verify the API call
    assert route.called
    assert float(route.calls.last.request.read()) == 0.9


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_truth_setter_validation_error(respx_mock: MockRouter):
    """Test setting invalid truth values."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the POST endpoint with a validation error
    route = respx_mock.post(f"/models/{dummy.model.name}/truth").mock(
        return_value=Response(422)
    )

    # Attempt to set an invalid truth value
    with pytest.raises(MatchboxUnparsedClientRequest):
        model.truth = 1.5

    assert route.called


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_ancestors_getter(respx_mock: MockRouter):
    """Test getting model ancestors via the API."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    ancestors_data = [
        ModelAncestor(name="model1", truth=0.9).model_dump(),
        ModelAncestor(name="model2", truth=0.8).model_dump(),
    ]

    # Mock the GET /models/{name}/ancestors endpoint
    route = respx_mock.get(f"/models/{dummy.model.name}/ancestors").mock(
        return_value=Response(200, json=ancestors_data)
    )

    # Get ancestors
    ancestors = model.ancestors

    # Verify the API call
    assert route.called
    assert ancestors == {"model1": 0.9, "model2": 0.8}


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_ancestors_cache_operations(respx_mock: MockRouter):
    """Test getting and setting model ancestors cache via the API."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the GET endpoint
    get_route = respx_mock.get(f"/models/{dummy.model.name}/ancestors_cache").mock(
        return_value=Response(
            200, json=[ModelAncestor(name="model1", truth=0.9).model_dump()]
        )
    )

    # Mock the POST endpoint
    set_route = respx_mock.post(f"/models/{dummy.model.name}/ancestors_cache").mock(
        return_value=Response(
            200,
            json=ModelOperationStatus(
                success=True,
                model_name=dummy.model.name,
                operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
            ).model_dump(),
        )
    )

    # Get ancestors cache
    cache = model.ancestors_cache
    assert get_route.called
    assert cache == {"model1": 0.9}

    # Set ancestors cache
    model.ancestors_cache = {"model2": 0.8}
    assert set_route.called
    assert json.loads(set_route.calls.last.request.content.decode()) == [
        ModelAncestor(name="model2", truth=0.8).model_dump()
    ]


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_ancestors_cache_set_error(respx_mock: MockRouter):
    """Test error handling when setting ancestors cache."""
    dummy = model_factory(model_type="linker")
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the POST endpoint with an error
    route = respx_mock.post(f"/models/{dummy.model.name}/ancestors_cache").mock(
        return_value=Response(
            500,
            json=ModelOperationStatus(
                success=False,
                model_name=dummy.model.name,
                operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
                details="Database error",
            ).model_dump(),
        )
    )

    # Attempt to set ancestors cache
    with pytest.raises(MatchboxUnhandledServerResponse, match="Database error"):
        model.ancestors_cache = {"model1": 0.9}

    assert route.called


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_delete_model(respx_mock: MockRouter):
    """Test successfully deleting a model."""
    # Create test model using factory
    dummy = model_factory()
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the DELETE endpoint with success response
    route = respx_mock.delete(
        f"/models/{dummy.model.name}", params={"certain": True}
    ).mock(
        return_value=Response(
            200,
            json=ModelOperationStatus(
                success=True,
                model_name=dummy.model.name,
                operation=ModelOperationType.DELETE,
            ).model_dump(),
        )
    )

    # Delete the model
    response = model.delete(certain=True)

    # Verify the response and API call
    assert response
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "true"


@pytest.mark.respx(base_url=getenv("MB__CLIENT__API_ROOT"))
def test_delete_model_needs_confirmation(respx_mock: MockRouter):
    """Test attempting to delete a model without confirmation returns 409."""
    # Create test model using factory
    dummy = model_factory()
    model = Model(
        metadata=dummy.model,
        model_instance=Mock(),
        left_data=DataFrame(),
        right_data=DataFrame(),
    )

    # Mock the DELETE endpoint with 409 confirmation required response
    error_details = "Cannot delete model with dependent models: dedupe1, dedupe2"
    route = respx_mock.delete(f"/models/{dummy.model.name}").mock(
        return_value=Response(
            409,
            json=ModelOperationStatus(
                success=False,
                model_name=dummy.model.name,
                operation=ModelOperationType.DELETE,
                details=error_details,
            ).model_dump(),
        )
    )

    # Attempt to delete without certain=True
    with pytest.raises(MatchboxConfirmDelete):
        model.delete()

    # Verify the response and API call
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "false"
