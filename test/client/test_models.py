import json
from datetime import datetime

import pyarrow as pa
import pytest
from httpx import Response
from respx.router import MockRouter
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.models import Model, add_model_class
from matchbox.client.models.linkers.base import LinkerSettings
from matchbox.client.queries import Query
from matchbox.client.results import Results
from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    BackendUploadType,
    CRUDOperation,
    ModelConfig,
    ModelType,
    NotFoundError,
    ResolutionOperationStatus,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxServerFileError,
)
from matchbox.common.factories.models import MockLinker, model_factory
from matchbox.common.factories.sources import source_factory


def test_init_and_run_model(sqlite_warehouse: Engine, matchbox_api: MockRouter):
    """Test that model can be initialised and run correctly."""
    # Register "custom" model
    add_model_class(MockLinker)

    # Mock API
    foo = source_factory(engine=sqlite_warehouse).write_to_location()
    bar = source_factory(engine=sqlite_warehouse).write_to_location()

    foo_leafy_data = foo.data.append_column(
        "leaf_id", pa.array(range(len(foo.data)), type=pa.int64())
    )
    bar_leafy_data = foo.data.append_column(
        "leaf_id",
        pa.array(
            range(len(foo_leafy_data), len(foo_leafy_data) + len(bar.data)),
            type=pa.int64(),
        ),
    )

    # Mock API
    matchbox_api.get("/query").mock(
        side_effect=[
            # First query
            Response(200, content=table_to_buffer(foo.data).read()),
            Response(200, content=table_to_buffer(bar.data).read()),
            # Second query (for validation)
            Response(200, content=table_to_buffer(foo_leafy_data).read()),
            Response(200, content=table_to_buffer(bar_leafy_data).read()),
        ]
    )
    dag = DAG("collection")
    foo_query = Query(foo.source, dag=dag)
    bar_query = Query(bar.source, dag=dag)

    model = Model(
        dag=dag,
        name="name",
        description="description",
        model_class=MockLinker,
        model_settings=LinkerSettings(),
        left_query=foo_query,
        right_query=bar_query,
    )

    assert model.config == ModelConfig(
        type=ModelType.LINKER,
        model_class="MockLinker",
        model_settings=json.dumps({"left_id": "l.field", "right_id": "r.field"}),
        left_query=foo_query.config,
        right_query=bar_query.config,
    )

    model.run()
    # This won't actually run anything new
    with pytest.warns(match="already run"):
        model.run(for_validation=True)
    assert model.results.left_root_leaf is None
    assert model.results.right_root_leaf is None

    model.run(for_validation=True, full_rerun=True)
    assert model.results.left_root_leaf is not None
    assert model.results.right_root_leaf is not None


def test_model_sync(matchbox_api: MockRouter):
    """Test syncing a model, its truth and results."""
    # Create test model using factory
    testkit = model_factory(model_type="linker")

    # Mock endpoints
    get_route = matchbox_api.get(f"/resolutions/{testkit.model.name}").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Model not found", entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        )
    )
    insert_config_route = matchbox_api.post("/resolutions").mock(
        return_value=Response(
            201,
            json=ResolutionOperationStatus(
                success=True,
                name=testkit.model.name,
                operation=CRUDOperation.CREATE,
            ).model_dump(),
        )
    )

    set_truth_route = matchbox_api.patch(
        f"/resolutions/{testkit.model.name}/truth"
    ).mock(
        return_value=Response(
            200,
            json=ResolutionOperationStatus(
                success=True,
                name=testkit.model.name,
                operation=CRUDOperation.UPDATE,
            ).model_dump(),
        )
    )

    upload_route = matchbox_api.post(f"/resolutions/{testkit.model.name}/data").mock(
        return_value=Response(
            202,
            content=UploadStatus(
                name=testkit.model.name,
                stage=UploadStage.PROCESSING,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
            ).model_dump_json(),
        )
    )

    status_route = matchbox_api.get(
        f"resolutions/{testkit.model.name}/data/status"
    ).mock(
        return_value=Response(
            200,
            content=UploadStatus(
                name=testkit.model.name,
                stage=UploadStage.COMPLETE,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
            ).model_dump_json(),
        )
    )

    # Call sync
    testkit.model.sync()

    # Verify the API call
    assert get_route.called
    assert insert_config_route.called
    assert (
        insert_config_route.calls.last.request.content.decode()
        == testkit.model.to_resolution().model_dump_json()
    )
    assert set_truth_route.called
    assert float(set_truth_route.calls.last.request.read()) == 100

    # Set results
    test_results = Results(probabilities=testkit.probabilities)

    testkit.model.results = test_results
    testkit.model.sync()

    # Verify API calls
    assert upload_route.called
    assert status_route.called
    assert (
        b"PAR1" in upload_route.calls.last.request.content
    )  # Check for parquet file signature

    # Mock the upload endpoint with a failure
    matchbox_api.post(f"/resolutions/{testkit.model.name}/data").mock(
        return_value=Response(
            400,
            content=UploadStatus(
                name=testkit.model.name,
                stage=UploadStage.FAILED,
                update_timestamp=datetime.now(),
                entity=BackendUploadType.RESULTS,
                details="Invalid data format",
            ).model_dump_json(),
        )
    )

    with pytest.raises(MatchboxServerFileError, match="Invalid data format"):
        testkit.model.sync()

    # Mock earlier endpoint generating a name clash
    source = source_factory().source
    matchbox_api.get(f"/resolutions/{testkit.model.name}").mock(
        return_value=Response(200, json=source.to_resolution().model_dump())
    )

    with pytest.raises(ValueError, match="existing resolution"):
        testkit.model.sync()


def test_truth_getter():
    """Test getting model truth threshold from config."""
    # Create testkit with specific truth value
    testkit = model_factory(model_type="linker")
    # Update the model to have a truth value
    testkit.model._truth = 90  # Integer truth value (90 = 0.9 as float)

    # Get truth as float
    truth = testkit.model.truth

    # Verify it returns the correct value converted to float
    assert truth == 0.9


def test_truth_setter_validation_error():
    """Test setting invalid truth values."""
    testkit = model_factory(model_type="linker")

    # Attempt to set an invalid truth value using the validated setter
    with pytest.raises(ValueError):
        testkit.model.truth = 1.5


def test_delete_resolution(matchbox_api: MockRouter):
    """Test successfully deleting a resolution."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with success response
    route = matchbox_api.delete(
        f"/resolutions/{testkit.model.name}", params={"certain": True}
    ).mock(
        return_value=Response(
            200,
            json=ResolutionOperationStatus(
                success=True,
                name=testkit.model.name,
                operation=CRUDOperation.DELETE,
            ).model_dump(),
        )
    )

    # Delete the model
    response = testkit.model.delete(certain=True)

    # Verify the response and API call
    assert response
    assert route.called
    assert route.calls.last.request.url.params["certain"] == "true"


def test_delete_resolution_needs_confirmation(matchbox_api: MockRouter):
    """Test attempting to delete a resolution without confirmation returns 409."""
    # Create test model using factory
    testkit = model_factory()

    # Mock the DELETE endpoint with 409 confirmation required response
    error_details = "Cannot delete model with dependent models: dedupe1, dedupe2"
    route = matchbox_api.delete(f"/resolutions/{testkit.model.name}").mock(
        return_value=Response(
            409,
            json=ResolutionOperationStatus(
                success=False,
                name=testkit.model.name,
                operation=CRUDOperation.DELETE,
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
