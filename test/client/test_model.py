import json

import pytest
from httpx import Response
from respx.router import MockRouter

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
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxServerFileError,
    MatchboxUnhandledServerResponse,
    MatchboxUnparsedClientRequest,
)
from matchbox.common.factories.models import model_factory


def test_insert_model(matchbox_api: MockRouter):
    """Test inserting a model via the API."""
    # Create test model using factory
    testkit = model_factory(model_type="linker")

    # Mock the POST /models endpoint
    route = matchbox_api.post("/models").mock(
        return_value=Response(
            201,
            json=ModelOperationStatus(
                success=True,
                model_name=testkit.model.metadata.name,
                operation=ModelOperationType.INSERT,
            ).model_dump(),
        )
    )

    # Call insert_model
    testkit.model.insert_model()

    # Verify the API call
    assert route.called
    assert (
        route.calls.last.request.content.decode()
        == testkit.model.metadata.model_dump_json()
    )


def test_insert_model_error(matchbox_api: MockRouter):
    """Test handling of model insertion errors."""
    testkit = model_factory(model_type="linker")

    # Mock the POST /models endpoint with an error response
    route = matchbox_api.post("/models").mock(
        return_value=Response(
            500,
            json=ModelOperationStatus(
                success=False,
                model_name=testkit.model.metadata.name,
                operation=ModelOperationType.INSERT,
                details="Internal server error",
            ).model_dump(),
        )
    )

    # Call insert_model and verify it raises an exception
    with pytest.raises(MatchboxUnhandledServerResponse, match="Internal server error"):
        testkit.model.insert_model()

    assert route.called


def test_results_getter(matchbox_api: MockRouter):
    """Test getting model results via the API."""
    testkit = model_factory(model_type="linker")

    # Mock the GET /models/{name}/results endpoint
    route = matchbox_api.get(f"/models/{testkit.model.metadata.name}/results").mock(
        return_value=Response(
            200, content=table_to_buffer(testkit.probabilities).read()
        )
    )

    # Get results
    results = testkit.model.results

    # Verify the API call
    assert route.called
    assert isinstance(results, Results)
    assert results.probabilities.schema.equals(SCHEMA_RESULTS)


def test_results_getter_not_found(matchbox_api: MockRouter):
    """Test getting model results when they don't exist."""
    testkit = model_factory(model_type="linker")

    # Mock the GET endpoint with a 404 response
    route = matchbox_api.get(f"/models/{testkit.model.metadata.name}/results").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Results not found", entity=BackendRetrievableType.RESOLUTION
            ).model_dump(),
        )
    )

    # Verify that accessing results raises an exception
    with pytest.raises(MatchboxResolutionNotFoundError, match="Results not found"):
        _ = testkit.model.results

    assert route.called


def test_results_setter(matchbox_api: MockRouter):
    """Test setting model results via the API."""
    testkit = model_factory(model_type="linker")

    # Mock the endpoints needed for results upload
    init_route = matchbox_api.post(
        f"/models/{testkit.model.metadata.name}/results"
    ).mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.RESULTS,
            ).model_dump(),
        )
    )

    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id",
                status="processing",
                entity=BackendUploadType.RESULTS,
            ).model_dump(),
        )
    )

    status_route = matchbox_api.get("/upload/test-upload-id/status").mock(
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
    test_results = Results(
        probabilities=testkit.probabilities, metadata=testkit.model.metadata
    )
    testkit.model.results = test_results

    # Verify API calls
    assert init_route.called
    assert upload_route.called
    assert status_route.called
    assert (
        b"PAR1" in upload_route.calls.last.request.content
    )  # Check for parquet file signature


def test_results_setter_upload_failure(matchbox_api: MockRouter):
    """Test handling of upload failures when setting results."""
    testkit = model_factory(model_type="linker")

    # Mock the initial POST endpoint
    init_route = matchbox_api.post(
        f"/models/{testkit.model.metadata.name}/results"
    ).mock(
        return_value=Response(
            202,
            json=UploadStatus(
                id="test-upload-id",
                status="awaiting_upload",
                entity=BackendUploadType.RESULTS,
            ).model_dump(),
        )
    )

    # Mock the upload endpoint with a failure
    upload_route = matchbox_api.post("/upload/test-upload-id").mock(
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
    test_results = Results(
        probabilities=testkit.probabilities, metadata=testkit.model.metadata
    )
    with pytest.raises(MatchboxServerFileError, match="Invalid data format"):
        testkit.model.results = test_results

    assert init_route.called
    assert upload_route.called


def test_truth_getter(matchbox_api: MockRouter):
    """Test getting model truth threshold via the API."""
    testkit = model_factory(model_type="linker")

    # Mock the GET /models/{name}/truth endpoint
    route = matchbox_api.get(f"/models/{testkit.model.metadata.name}/truth").mock(
        return_value=Response(200, json=0.9)
    )

    # Get truth
    truth = testkit.model.truth

    # Verify the API call
    assert route.called
    assert truth == 0.9


def test_truth_setter(matchbox_api: MockRouter):
    """Test setting model truth threshold via the API."""
    testkit = model_factory(model_type="linker")

    # Mock the PATCH /models/{name}/truth endpoint
    route = matchbox_api.patch(f"/models/{testkit.model.metadata.name}/truth").mock(
        return_value=Response(
            200,
            json=ModelOperationStatus(
                success=True,
                model_name=testkit.model.metadata.name,
                operation=ModelOperationType.UPDATE_TRUTH,
            ).model_dump(),
        )
    )

    # Set truth
    testkit.model.truth = 0.9

    # Verify the API call
    assert route.called
    assert float(route.calls.last.request.read()) == 0.9


def test_truth_setter_validation_error(matchbox_api: MockRouter):
    """Test setting invalid truth values."""
    testkit = model_factory(model_type="linker")

    # Mock the PATCH endpoint with a validation error
    route = matchbox_api.patch(f"/models/{testkit.model.metadata.name}/truth").mock(
        return_value=Response(422)
    )

    # Attempt to set an invalid truth value
    with pytest.raises(MatchboxUnparsedClientRequest):
        testkit.model.truth = 1.5

    assert route.called


def test_ancestors_getter(matchbox_api: MockRouter):
    """Test getting model ancestors via the API."""
    testkit = model_factory(model_type="linker")

    ancestors_data = [
        ModelAncestor(name="model1", truth=90).model_dump(),
        ModelAncestor(name="model2", truth=80).model_dump(),
    ]

    # Mock the GET /models/{name}/ancestors endpoint
    route = matchbox_api.get(f"/models/{testkit.model.metadata.name}/ancestors").mock(
        return_value=Response(200, json=ancestors_data)
    )

    # Get ancestors
    ancestors = testkit.model.ancestors

    # Verify the API call
    assert route.called
    # assert ancestors == {"model1": 0.9, "model2": 0.8}
    assert ancestors == {"model1": 90, "model2": 80}


def test_ancestors_cache_operations(matchbox_api: MockRouter):
    """Test getting and setting model ancestors cache via the API."""
    testkit = model_factory(model_type="linker")

    # Mock the GET endpoint
    get_route = matchbox_api.get(
        f"/models/{testkit.model.metadata.name}/ancestors_cache"
    ).mock(
        return_value=Response(
            200, json=[ModelAncestor(name="model1", truth=90).model_dump()]
        )
    )

    # Mock the POST endpoint
    set_route = matchbox_api.post(
        f"/models/{testkit.model.metadata.name}/ancestors_cache"
    ).mock(
        return_value=Response(
            200,
            json=ModelOperationStatus(
                success=True,
                model_name=testkit.model.metadata.name,
                operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
            ).model_dump(),
        )
    )

    # Get ancestors cache
    cache = testkit.model.ancestors_cache
    assert get_route.called
    assert cache == {"model1": 90}

    # Set ancestors cache
    testkit.model.ancestors_cache = {"model2": 80}
    assert set_route.called
    assert json.loads(set_route.calls.last.request.content.decode()) == [
        ModelAncestor(name="model2", truth=80).model_dump()
    ]


def test_ancestors_cache_set_error(matchbox_api: MockRouter):
    """Test error handling when setting ancestors cache."""
    testkit = model_factory(model_type="linker")

    # Mock the POST endpoint with an error
    route = matchbox_api.post(
        f"/models/{testkit.model.metadata.name}/ancestors_cache"
    ).mock(
        return_value=Response(
            500,
            json=ModelOperationStatus(
                success=False,
                model_name=testkit.model.metadata.name,
                operation=ModelOperationType.UPDATE_ANCESTOR_CACHE,
                details="Database error",
            ).model_dump(),
        )
    )

    # Attempt to set ancestors cache
    with pytest.raises(MatchboxUnhandledServerResponse, match="Database error"):
        testkit.model.ancestors_cache = {"model1": 90}

    assert route.called


def test_delete_model(matchbox_api: MockRouter):
    """Test successfully deleting a model."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with success response
    route = matchbox_api.delete(
        f"/models/{testkit.model.metadata.name}", params={"certain": True}
    ).mock(
        return_value=Response(
            200,
            json=ModelOperationStatus(
                success=True,
                model_name=testkit.model.metadata.name,
                operation=ModelOperationType.DELETE,
            ).model_dump(),
        )
    )

    # Delete the model
    response = testkit.model.delete(certain=True)

    # Verify the response and API call
    assert response
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "true"


def test_delete_model_needs_confirmation(matchbox_api: MockRouter):
    """Test attempting to delete a model without confirmation returns 409."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with 409 confirmation required response
    error_details = "Cannot delete model with dependent models: dedupe1, dedupe2"
    route = matchbox_api.delete(f"/models/{testkit.model.metadata.name}").mock(
        return_value=Response(
            409,
            json=ModelOperationStatus(
                success=False,
                model_name=testkit.model.metadata.name,
                operation=ModelOperationType.DELETE,
                details=error_details,
            ).model_dump(),
        )
    )

    # Attempt to delete without certain=True
    with pytest.raises(MatchboxDeletionNotConfirmed):
        testkit.model.delete()

    # Verify the response and API call
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "false"
