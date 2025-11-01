import json

import pyarrow as pa
import pytest
from httpx import Response
from respx.router import MockRouter
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.models import Model, add_model_class
from matchbox.client.models.linkers.base import LinkerSettings
from matchbox.client.queries import Query
from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    ModelConfig,
    ModelType,
    NotFoundError,
    Resolution,
    ResourceOperationStatus,
    UploadInfo,
    UploadStage,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxServerFileError,
)
from matchbox.common.factories.models import MockLinker, model_factory
from matchbox.common.factories.sources import (
    source_factory,
)


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
    assert model.results.left_root_leaf is None
    assert model.results.right_root_leaf is None

    model.run(for_validation=True)
    assert model.results.left_root_leaf is not None
    assert model.results.right_root_leaf is not None


def test_model_sync(matchbox_api: MockRouter):
    # Mock model
    testkit = model_factory(model_type="linker")

    # Mock the routes:
    # Resolution doesn't yet exist
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}"
    ).mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Model not found", entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        )
    )

    # Resolution can be inserted
    insert_config_route = matchbox_api.post(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}"
    ).mock(
        return_value=Response(
            201,
            content=ResourceOperationStatus(
                success=True,
                target=f"Resolution {testkit.model.name}",
                operation=CRUDOperation.CREATE,
            ).model_dump_json(),
        )
    )

    # Resolution data can be inserted
    insert_results_route = matchbox_api.post(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}/data"
    ).mock(
        return_value=Response(
            202,
            json=ResourceOperationStatus(
                success=True, target="", operation=CRUDOperation.CREATE
            ).model_dump(),
        )
    )

    # Later, resolution can be updated
    update_route = matchbox_api.put(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}"
    ).mock(
        return_value=Response(
            200,
            content=ResourceOperationStatus(
                success=True,
                target=f"Resolution {testkit.model.name}",
                operation=CRUDOperation.UPDATE,
            ).model_dump_json(),
        )
    )

    # Later, resolution can be deleted and recreated
    delete_route = matchbox_api.delete(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}"
    ).mock(
        return_value=Response(
            200,
            content=ResourceOperationStatus(
                success=True,
                target=f"Resolution {testkit.model.name}",
                operation=CRUDOperation.DELETE,
            ).model_dump_json(),
        )
    )

    # -- ERRORS --

    # Can't sync before running
    with pytest.raises(RuntimeError, match="must be run"):
        testkit.model.sync()

    # We now run, but test that upload failure is handled
    testkit.fake_run()
    # Before and after upload, server accepts data - the second time reporting error
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}/data/status"
    ).mock(
        side_effect=[
            Response(200, json=UploadInfo(stage=UploadStage.READY).model_dump()),
            Response(
                200,
                json=UploadInfo(stage=UploadStage.READY, error="error").model_dump(),
            ),
        ]
    )
    with pytest.raises(MatchboxServerFileError, match="error"):
        testkit.model.sync()

    # -- FIRST TIME INSERTION --

    # Before upload, resolution is ready for data, after it is complete
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}/data/status"
    ).mock(
        side_effect=[
            Response(200, json=UploadInfo(stage=UploadStage.READY).model_dump()),
            Response(200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()),
        ]
    )

    # Sync the source, successfully
    testkit.model.sync()
    # Source was created, not updated or deleted
    assert insert_config_route.called
    assert not update_route.called
    assert not delete_route.called
    # Resolution metadata was correct
    resolution_call = Resolution.model_validate_json(
        insert_config_route.calls.last.request.content.decode("utf-8")
    )
    assert resolution_call == testkit.model.to_resolution()
    # Resolution data was correct
    assert (
        b"Content-Disposition: form-data;"
        in insert_results_route.calls.last.request.content
    )
    assert b"PAR1" in insert_results_route.calls.last.request.content

    # -- SOFT UPDATE --

    insert_results_route.reset()
    # Mock endpoint now returns existing resolution
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}"
    ).mock(return_value=Response(200, json=testkit.model.to_resolution().model_dump()))

    # Mock endpoint now declares data is present already
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}/data/status"
    ).mock(
        return_value=Response(
            200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()
        )
    )

    testkit.model.sync()
    # Resolution was compatible: ensure it was updated, not deleted
    assert update_route.called
    assert not delete_route.called
    # The data did not need to be updated
    assert not insert_results_route.called

    # -- HARD UPDATE --

    insert_results_route.reset()
    # Changing data requires deletion and re-insertion
    testkit.probabilities = testkit.probabilities.slice(1, 3)
    testkit.fake_run()
    # Resolution data is first ready to upload, and then uploaded
    matchbox_api.get(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}/data/status"
    ).mock(
        side_effect=[
            Response(200, json=UploadInfo(stage=UploadStage.READY).model_dump()),
            Response(200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()),
        ]
    )

    testkit.model.sync()
    assert delete_route.called
    assert insert_results_route.called


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
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}",
        params={"certain": True},
    ).mock(
        return_value=Response(
            200,
            json=ResourceOperationStatus(
                success=True,
                target=f"Resolution {testkit.model.name}",
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
    route = matchbox_api.delete(
        f"/collections/{testkit.model.dag.name}/runs/{testkit.model.dag.run}/resolutions/{testkit.model.name}"
    ).mock(
        return_value=Response(
            409,
            json=ResourceOperationStatus(
                success=False,
                target=f"Resolution {testkit.model.name}",
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
