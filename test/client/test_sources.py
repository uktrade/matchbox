from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from respx import MockRouter
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.locations import RelationalDBLocation
from matchbox.client.sources import (
    Source,
)
from matchbox.common.dtos import (
    BackendResourceType,
    CRUDOperation,
    DataTypes,
    NotFoundError,
    Resolution,
    ResourceOperationStatus,
    SourceField,
    UploadInfo,
    UploadStage,
)
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.factories.sources import (
    source_factory,
)


def test_source_infers_type(sqlite_warehouse: Engine):
    """Creating a source with type inference works."""
    # Create test data
    source_testkit = source_factory(
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "datatype": DataTypes.STRING},
        ],
        engine=sqlite_warehouse,
    ).write_to_location()

    location = RelationalDBLocation(name="dbname").set_client(sqlite_warehouse)
    source = Source(
        dag=source_testkit.source.dag,
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name"],
    )

    assert source.config.key_field == SourceField(name="key", type=DataTypes.STRING)
    assert source.config.index_fields == tuple(
        [SourceField(name="name", type=DataTypes.STRING)]
    )


def test_source_sampling_preserves_original_sql(sqlite_warehouse: Engine):
    """SQL on RelationalDBLocation is preserved.

    SQLGlot transpiles INSTR() to STR_POSITION() in its default dialect.
    """
    # Create test data
    source_testkit = source_factory(
        n_true_entities=3,
        features=[
            {
                "name": "text_col",
                "base_generator": "word",
                "datatype": DataTypes.STRING,
            },
        ],
        engine=sqlite_warehouse,
    ).write_to_location()

    location = RelationalDBLocation(name="dbname").set_client(sqlite_warehouse)

    # Use SQLite's INSTR function (returns position of substring)
    # Other databases use CHARINDEX, POSITION, etc.
    extract_transform = f"""
        SELECT
            key,
            text_col,
            INSTR(text_col, 'a') as position_of_a
        FROM
            "{source_testkit.name}"
    """

    # This should work since INSTR is valid SQLite
    # Would fail if validation transpiles INSTR to POSITION() or similar
    source = Source(
        dag=source_testkit.source.dag,
        location=location,
        name="test_source",
        extract_transform=extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["text_col", "position_of_a"],
    )

    assert source.config.key_field == SourceField(name="key", type=DataTypes.STRING)
    assert len(source.config.index_fields) == 2

    # This should work if the SQL is preserved exactly
    df = next(source.fetch())
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3


def test_source_fetch(sqlite_warehouse: Engine):
    """Test the query method with default parameters."""
    # Create test data
    source_testkit = source_factory(
        n_true_entities=5,
        features=[
            {"name": "name", "base_generator": "word", "datatype": DataTypes.STRING},
        ],
        engine=sqlite_warehouse,
    ).write_to_location()

    # Create location and source
    location = RelationalDBLocation(name="dbname").set_client(sqlite_warehouse)
    source = Source(
        dag=source_testkit.source.dag,
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name"],
    )

    # Execute query
    result = next(source.fetch())

    # Verify result
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5
    assert "key" in result.columns
    assert "name" in result.columns

    # Try applying key filter
    key_subset = result[source.config.key_field.name][:2].to_list()
    result = next(source.fetch(keys=key_subset))
    assert len(result) == 2

    # Key filter ineffective with empty list
    result = next(source.fetch(keys=[]))
    assert len(result) == 5


@pytest.mark.parametrize(
    "qualify_names",
    [
        pytest.param(False, id="no_name_qualification"),
        pytest.param(True, id="with_name_qualification"),
    ],
)
@patch("matchbox.client.locations.RelationalDBLocation.execute")
def test_source_fetch_name_qualification(
    mock_execute: Mock, qualify_names: bool, sqlite_in_memory_warehouse: Engine
):
    """Test that column names are qualified when requested."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(name="sqlite").set_client(
        sqlite_in_memory_warehouse
    )

    # Create source
    source = Source(
        dag=DAG("collection"),
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
    )

    # Call query with qualification parameter
    next(source.fetch(qualify_names=qualify_names))

    # Verify the rename parameter passed to execute
    _, kwargs = mock_execute.call_args
    rename_param = kwargs.get("rename")

    if qualify_names:
        assert rename_param is not None
        assert callable(rename_param)
        # Test the rename function
        sample_col = "test_col"
        assert "test_source_" in source.name + "_" + sample_col
    else:
        assert rename_param is None


@pytest.mark.parametrize(
    ("batch_size", "expected_call_kwargs"),
    [
        pytest.param(
            None,
            {"batch_size": None},
            id="single_return",
        ),
        pytest.param(3, {"batch_size": 3}, id="multiple_batches"),
    ],
)
@patch("matchbox.client.locations.RelationalDBLocation.execute")
def test_source_fetch_batching(
    mock_execute: Mock,
    batch_size: int,
    expected_call_kwargs: dict,
    sqlite_in_memory_warehouse: Engine,
):
    """Test query with batching options."""
    # Mock the location execute method to verify parameters
    mock_execute.return_value = (x for x in [None])  # execute needs to be a generator
    location = RelationalDBLocation(name="sqlite").set_client(
        sqlite_in_memory_warehouse
    )

    # Create source
    source = Source(
        dag=DAG("collection"),
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
    )

    # Call query with batching parameters
    next(source.fetch(batch_size=batch_size))

    # Verify parameters passed to execute
    _, kwargs = mock_execute.call_args
    for key, value in expected_call_kwargs.items():
        assert kwargs.get(key) == value


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(None, id="no_batching"),
        pytest.param(2, id="with_batching"),
    ],
)
def test_source_run(sqlite_warehouse: Engine, batch_size: int):
    """Test the run method produces expected hash format."""
    # Create test data with unique values
    n_true_entities = 3
    source_testkit = source_factory(
        n_true_entities=n_true_entities,
        features=[
            {"name": "name", "base_generator": "name", "datatype": DataTypes.STRING},
            {
                "name": "age",
                "base_generator": "random_int",
                "datatype": DataTypes.INT64,
            },
        ],
        engine=sqlite_warehouse,
    ).write_to_location()

    # Create location and source
    location = RelationalDBLocation(name="dbname").set_client(sqlite_warehouse)
    source = Source(
        dag=source_testkit.source.dag,
        location=location,
        name="test_source",
        extract_transform=source_testkit.source_config.extract_transform,
        infer_types=True,
        key_field="key",
        index_fields=["name", "age"],
    )

    # Execute run with different batching parameters
    result = source.run(batch_size=batch_size) if batch_size else source.run()

    # Verify result
    assert isinstance(result, pa.Table)
    assert "hash" in result.column_names
    assert "keys" in result.column_names
    assert len(result) == n_true_entities

    source.run()


@patch("matchbox.client.sources.Source.fetch")
def test_source_run_null_identifier(
    mock_fetch: Mock, sqlite_in_memory_warehouse: Engine
):
    """Test hashing data raises an error when source primary keys contain nulls."""
    # Create a source
    location = RelationalDBLocation(name="sqlite").set_client(
        sqlite_in_memory_warehouse
    )
    source = Source(
        dag=DAG("collection"),
        location=location,
        name="test_source",
        extract_transform="SELECT key, name FROM users",
        key_field=SourceField(name="key", type=DataTypes.STRING),
        index_fields=[SourceField(name="name", type=DataTypes.STRING)],
    )

    # Mock query to return data with null keys
    mock_df = pl.DataFrame({"key": ["1", None], "name": ["a", "b"]})
    mock_fetch.return_value = (x for x in [mock_df])

    # hashing data should raise ValueErrors for null keys
    with pytest.raises(ValueError, match="keys column contains null values"):
        source.run()


def test_source_sync(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test source syncing flow through the API."""
    # Mock source
    testkit = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}],
        engine=sqlite_warehouse,
    ).write_to_location()

    # Mock the routes:
    # Resolution doesn't yet exist
    matchbox_api.get(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}"
    ).mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Source not found", entity=BackendResourceType.RESOLUTION
            ).model_dump(),
        )
    )
    # Resolution can be inserted
    insert_config_route = matchbox_api.post(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}"
    ).mock(
        return_value=Response(
            201,
            content=ResourceOperationStatus(
                success=True,
                target=f"Resolution {testkit.source.name}",
                operation=CRUDOperation.CREATE,
            ).model_dump_json(),
        )
    )
    # Resolution data can be inserted
    insert_hashes_route = matchbox_api.post(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}/data"
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
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}"
    ).mock(
        return_value=Response(
            200,
            content=ResourceOperationStatus(
                success=True,
                target=f"Resolution {testkit.source.name}",
                operation=CRUDOperation.UPDATE,
            ).model_dump_json(),
        )
    )

    # Later, resolution can be deleted and recreated
    delete_route = matchbox_api.delete(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}"
    ).mock(
        return_value=Response(
            200,
            content=ResourceOperationStatus(
                success=True,
                target=f"Resolution {testkit.source.name}",
                operation=CRUDOperation.DELETE,
            ).model_dump_json(),
        )
    )

    # -- ERRORS --

    # Can't sync before running
    with pytest.raises(RuntimeError, match="must be run"):
        testkit.source.sync()

    # We now run, but test that upload failure is handled
    testkit.fake_run()
    # Before and after upload, server accepts data - the second time reporting error
    matchbox_api.get(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}/data/status"
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
        testkit.source.sync()

    # -- FIRST TIME INSERTION --

    # Before upload, resolution is ready for data, after it is complete
    matchbox_api.get(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}/data/status"
    ).mock(
        side_effect=[
            Response(200, json=UploadInfo(stage=UploadStage.READY).model_dump()),
            Response(200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()),
        ]
    )

    # Sync the source, successfully
    testkit.source.run()
    testkit.source.sync()
    # Source was created, not updated or deleted
    assert insert_config_route.called
    assert not update_route.called
    assert not delete_route.called
    # Resolution metadata was correct
    resolution_call = Resolution.model_validate_json(
        insert_config_route.calls.last.request.content.decode("utf-8")
    )
    assert resolution_call == testkit.source.to_resolution()
    # Resolution data was correct
    assert (
        b"Content-Disposition: form-data;"
        in insert_hashes_route.calls.last.request.content
    )
    assert b"PAR1" in insert_hashes_route.calls.last.request.content

    # -- SOFT UPDATE --

    insert_hashes_route.reset()
    # Mock endpoint now returns existing resolution
    matchbox_api.get(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}"
    ).mock(return_value=Response(200, json=testkit.source.to_resolution().model_dump()))

    # Mock endpoint now declares data is present already
    matchbox_api.get(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}/data/status"
    ).mock(
        return_value=Response(
            200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()
        )
    )
    testkit.source.sync()
    # Resolution was compatible: ensure it was updated, not deleted
    assert update_route.called
    assert not delete_route.called
    # The data did not need to be updated
    assert not insert_hashes_route.called

    # -- HARD UPDATE --

    insert_hashes_route.reset()
    # Changing data requires deletion and re-insertion
    testkit.data_hashes = testkit.data_hashes.slice(1, 3)
    testkit.fake_run()
    # Resolution data is first ready to upload, and then uploaded
    matchbox_api.get(
        f"/collections/{testkit.source.dag.name}/runs/{testkit.source.dag.run}/resolutions/{testkit.source.name}/data/status"
    ).mock(
        side_effect=[
            Response(200, json=UploadInfo(stage=UploadStage.READY).model_dump()),
            Response(200, json=UploadInfo(stage=UploadStage.COMPLETE).model_dump()),
        ]
    )
    testkit.source.sync()

    assert delete_route.called
    assert insert_hashes_route.called
