import json
from datetime import datetime
from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.sources import Source
from matchbox.common.arrow import SCHEMA_QUERY, table_to_buffer
from matchbox.common.dtos import (
    BackendResourceType,
    Collection,
    CRUDOperation,
    Match,
    NotFoundError,
    Resolution,
    ResolutionName,
    ResolutionPath,
    ResourceOperationStatus,
    Run,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxEmptyServerResponse,
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import (
    linked_sources_factory,
    source_factory,
    source_from_tuple,
)


@patch.object(Source, "run")
@patch.object(Model, "run")
@patch.object(Source, "sync")
@patch.object(Model, "sync")
def test_dag_run_and_sync(
    model_sync_mock: Mock,
    source_sync_mock: Mock,
    model_run_mock: Mock,
    source_run_mock: Mock,
    sqlite_warehouse: Engine,
):
    """A legal DAG can be built and run."""
    # Set up constituents
    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse).write_to_location()
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse).write_to_location()
    baz_tkit = source_factory(name="baz", engine=sqlite_warehouse).write_to_location()

    dag = TestkitDAG().dag

    # Structure: sources can be added
    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    baz = dag.source(**baz_tkit.into_dag())

    assert set(dag.nodes.keys()) == {foo.name, bar.name, baz.name}

    # Structure: sources can be deduped
    d_foo = foo.query().deduper(
        name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )

    # Structure:
    # - sources can be passed directly to linkers
    # - or, linkers can take dedupers
    foo_bar = d_foo.query(foo).linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    # Structure: linkers can take other linkers
    foo_bar_baz = foo_bar.query(foo, bar, baz).linker(
        baz.query(),
        name="foo_bar_baz",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    assert set(dag.nodes.keys()) == {
        foo.name,
        bar.name,
        baz.name,
        d_foo.name,
        foo_bar.name,
        foo_bar_baz.name,
    }

    # Run DAG
    dag.run_and_sync()

    assert source_run_mock.call_count == 3
    assert source_sync_mock.call_count == 3
    assert model_run_mock.call_count == 3
    assert model_sync_mock.call_count == 3


def test_dags_missing_dependency(sqlite_warehouse: Engine):
    """Steps cannot be added before their dependencies."""
    dag = TestkitDAG().dag

    foo = source_factory(name="foo", engine=sqlite_warehouse, dag=dag).source

    with pytest.raises(ValueError, match="not added to DAG"):
        dag.model(
            left_query=foo.query(),
            name="d_foo",
            model_class=NaiveDeduper,
            model_settings={"unique_fields": []},
        )

    # Failure leads to no dags being added
    assert not len(dag.nodes)
    assert not len(dag.graph)


def test_mixing_dags_fails(sqlite_warehouse: Engine):
    """Cannot reference a different DAG when adding a step."""
    dag = TestkitDAG().dag
    dag2 = TestkitDAG().dag

    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse)
    foo = dag.source(**foo_tkit.into_dag())

    with pytest.raises(ValueError, match="mix DAGs"):
        # Different DAG in input
        dag2.model(
            left_query=foo.query(),
            name="d_foo",
            model_class=NaiveDeduper,
            model_settings={"unique_fields": []},
        )

    # Failure leads to no dags being added
    assert not len(dag2.nodes)
    assert not len(dag2.graph)


def test_dag_name_clash(sqlite_warehouse: Engine):
    """Names across sources and steps must be unique."""
    dag = TestkitDAG().dag

    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse)
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse)

    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())

    d_foo = foo.query().deduper(
        name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )

    with pytest.raises(ValueError, match="already taken"):
        bar.query().deduper(
            name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
        )

    # DAG is not modified by failed attempt
    assert dag.nodes["d_foo"] == d_foo
    # We didn't overwrite d_foo's dependencies
    assert dag.graph["d_foo"] == [foo.name]


@patch.object(Source, "run")
@patch.object(Model, "run")
@patch.object(Source, "sync")
@patch.object(Model, "sync")
def test_dag_disconnected(
    model_sync_mock: Mock,
    source_sync_mock: Mock,
    model_run_mock: Mock,
    source_run_mock: Mock,
    sqlite_warehouse: Engine,
):
    """Nodes cannot be disconnected."""
    dag = TestkitDAG().dag

    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse)
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse)

    dag.source(**foo_tkit.into_dag())
    dag.source(**bar_tkit.into_dag())

    with pytest.raises(ValueError, match="disconnected"):
        dag.run_and_sync()


def test_dag_draw(sqlite_warehouse: Engine):
    """Test that the draw method produces a correct string representation of the DAG."""
    # Set up a simple DAG
    foo_tkit = source_factory(name="foo", engine=sqlite_warehouse).write_to_location()
    bar_tkit = source_factory(name="bar", engine=sqlite_warehouse).write_to_location()
    baz_tkit = source_factory(name="baz", engine=sqlite_warehouse).write_to_location()

    dag = TestkitDAG().dag

    # Structure: sources can be added
    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    baz = dag.source(**baz_tkit.into_dag())

    d_foo = foo.query().deduper(
        name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )

    foo_bar = d_foo.query(foo).linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    # Structure: linkers can take other linkers
    foo_bar_baz = foo_bar.query(foo, bar, baz).linker(
        baz.query(),
        name="foo_bar_baz",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    # Prepare the DAG and draw it

    # Test 1: Drawing without timestamps (original behavior)
    tree_str = dag.draw()

    # Verify the structure
    lines = tree_str.strip().split("\n")

    # The root node should be first
    assert lines[0] == "foo_bar_baz"

    # Check that all nodes are present
    node_names = [
        foo.name,
        bar.name,
        baz.name,
        d_foo.name,
        foo_bar.name,
        foo_bar_baz.name,
    ]

    for node in node_names:
        # Either the node name is at the start of a line or after the tree characters
        node_present = any(line.endswith(node) for line in lines)
        assert node_present, f"Node {node} not found in the tree representation"

    # Check that tree has correct formatting with tree characters
    tree_chars = ["└──", "├──", "│"]
    has_tree_chars = any(char in tree_str for char in tree_chars)
    assert has_tree_chars, (
        "Tree representation doesn't use expected formatting characters"
    )

    # Test 2: Drawing with timestamps (status indicators)
    # Set d_foo as processing and foo_bar as completed
    start_time = datetime.now()
    doing = "d_foo"
    foo_bar.last_run = datetime.now()

    # Draw the DAG with status indicators
    tree_str_with_status = dag.draw(start_time=start_time, doing=doing)
    status_lines = tree_str_with_status.strip().split("\n")

    # Verify status indicators are present
    status_indicators = ["✅", "🔄", "⏸️"]
    assert any(indicator in tree_str_with_status for indicator in status_indicators)

    # Check specific statuses: foo_bar done, d_foo working, others awaiting
    for line in status_lines:
        name = line.split()[-1]
        if name == "foo_bar":
            assert "✅" in line
        elif name == "d_foo":
            assert "🔄" in line
        elif name in [foo.name, bar.name, baz.name]:
            assert "⏸️" in line

    # Test 3: Check that node names are still present with status indicators
    for node in node_names:
        node_present = any(node in line for line in status_lines)
        assert node_present, (
            f"Node {node} not found in the tree representation with status indicators"
        )

    # Test 4: Drawing with skipped nodes
    skipped_nodes = [foo.name, d_foo.name]
    tree_str_with_skipped = dag.draw(
        start_time=start_time, doing=doing, skipped=skipped_nodes
    )
    skipped_lines = tree_str_with_skipped.strip().split("\n")

    # Check that skipped nodes have the skipped indicator
    for line in skipped_lines:
        name = line.split()[-1]
        if any(name == skipped for skipped in skipped_nodes):
            assert "⏭️" in line

    # Test all status indicators together
    doing = "foo_bar_baz"
    tree_str_all_statuses = dag.draw(
        start_time=start_time, doing=doing, skipped=skipped_nodes
    )
    assert all(
        indicator in tree_str_all_statuses for indicator in ["✅", "🔄", "⏸️", "⏭️"]
    )


# Lookups


def test_extract_lookup(
    sqlite_warehouse: Engine,
    sqlite_in_memory_warehouse: Engine,
    matchbox_api: MockRouter,
):
    """Entire lookup can be extracted from DAG."""
    # Make dummy data
    foo = source_from_tuple(
        name="foo",
        location_name="sqlite",
        engine=sqlite_warehouse,
        data_keys=[1, 2, 3],
        data_tuple=({"col": 0}, {"col": 1}, {"col": 2}),
    ).write_to_location()
    bar = source_from_tuple(
        name="bar",
        location_name="sqlite_memory",
        engine=sqlite_in_memory_warehouse,
        data_keys=["a", "b", "c"],
        data_tuple=({"col": 10}, {"col": 11}, {"col": 12}),
    ).write_to_location()

    dag = DAG("companies")

    # Mock API
    # In the beginning, no run
    matchbox_api.get(f"/collections/{dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(
                name=dag.name,
                runs=[],
                default_run=None,
            ).model_dump(),
        )
    )

    matchbox_api.post(f"/collections/{dag.name}/runs").mock(
        return_value=Response(
            200,
            json=Run(run_id=1, resolutions={}).model_dump(),
        )
    )

    # Build dummy DAG
    dag.new_run().source(**foo.into_dag()).query().linker(
        dag.source(**bar.into_dag()).query(),
        name="root",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    # Then the new run
    matchbox_api.get(f"/collections/{dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(
                name=dag.name,
                runs=[1],
                default_run=1,
            ).model_dump(),
        )
    )

    matchbox_api.get(f"/collections/{dag.name}/runs/1").mock(
        return_value=Response(
            200,
            json=Run(
                run_id=1,
                resolutions={
                    foo.name: foo.source.to_resolution(),
                    bar.name: bar.source.to_resolution(),
                    "root": dag.get_model("root").to_resolution(),
                },
            ).model_dump(),
        )
    )

    matchbox_api.get(
        "/query", params={"source": "foo", "run_id": 1, "collection": dag.name}
    ).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"id": 1, "key": "1"},
                        {"id": 2, "key": "2"},
                        {"id": 3, "key": "3"},
                    ],
                    schema=SCHEMA_QUERY,
                )
            ).read(),
        )
    )

    matchbox_api.get(
        "/query", params={"source": "bar", "run_id": 1, "collection": dag.name}
    ).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.Table.from_pylist(
                    [
                        {"id": 1, "key": "a"},
                        {"id": 3, "key": "b"},
                        {"id": 3, "key": "c"},
                    ],
                    schema=SCHEMA_QUERY,
                )
            ).read(),
        )
    )

    # Because of FULL OUTER JOIN, we expect some values to be null, and some explosions
    expected_foo_bar_mapping = pl.DataFrame(
        [
            {"id": 1, "foo_key": "1", "bar_key": "a"},
            {"id": 2, "foo_key": "2", "bar_key": None},
            {"id": 3, "foo_key": "3", "bar_key": "b"},
            {"id": 3, "foo_key": "3", "bar_key": "c"},
        ]
    )

    # When selecting single source, we won't explode
    expected_foo_mapping = expected_foo_bar_mapping.select(["id", "foo_key"]).unique()

    # Case 0: No sources are found
    with pytest.raises(MatchboxResolutionNotFoundError):
        dag.extract_lookup(source_filter=["nonexistent"])

    with pytest.raises(MatchboxResolutionNotFoundError):
        dag.extract_lookup(location_names=["nonexistent"])

    # Case 1: Retrieve single table
    # With URI filter
    foo_mapping = dag.extract_lookup(location_names=["sqlite"])

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # With source filter
    foo_mapping = dag.extract_lookup(source_filter="foo")

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # With both filters
    foo_mapping = dag.extract_lookup(source_filter="foo", location_names="sqlite")

    assert_frame_equal(
        pl.from_arrow(foo_mapping),
        expected_foo_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # Case 2: Retrieve multiple tables
    # With no filter
    foo_bar_mapping = dag.extract_lookup()

    assert_frame_equal(
        pl.from_arrow(foo_bar_mapping),
        expected_foo_bar_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # With source filter
    foo_bar_mapping = dag.extract_lookup(source_filter=["foo", "bar"])

    assert_frame_equal(
        pl.from_arrow(foo_bar_mapping),
        expected_foo_bar_mapping,
        check_row_order=False,
        check_column_order=False,
    )

    # Case 3: Retrieve from reconstituted DAG
    reconstituted_dag = DAG("companies").load_default()
    assert reconstituted_dag.extract_lookup() == foo_bar_mapping


def test_lookup_key_ok(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """The DAG can map between single keys."""
    # Set up dummy data
    foo_testkit = source_factory(
        engine=sqlite_warehouse, name="foo"
    ).write_to_location()
    bar_testkit = source_factory(
        engine=sqlite_warehouse, name="bar"
    ).write_to_location()
    baz_testkit = source_factory(
        engine=sqlite_warehouse, name="baz"
    ).write_to_location()

    dag = TestkitDAG().dag

    foo = dag.source(**foo_testkit.into_dag())
    bar = dag.source(**bar_testkit.into_dag())
    baz = dag.source(**baz_testkit.into_dag())

    foo.query().linker(
        bar.query(),
        name="linker1",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    ).query(foo, bar).linker(
        baz.query(),
        name="linker2",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    foo_path = ResolutionPath(name="foo", collection=dag.name, run=dag.run)
    bar_path = ResolutionPath(name="bar", collection=dag.name, run=dag.run)
    baz_path = ResolutionPath(name="baz", collection=dag.name, run=dag.run)

    mock_match1 = Match(
        cluster=1, source=foo_path, source_id={"a"}, target=bar_path, target_id={"b"}
    )
    mock_match2 = Match(
        cluster=1, source=foo_path, source_id={"a"}, target=baz_path, target_id={"b"}
    )
    # The standard JSON serialiser does not handle Pydantic objects
    serialised_matches = json.dumps(
        [m.model_dump() for m in [mock_match1, mock_match2]]
    )

    matchbox_api.get("/match").mock(
        return_value=Response(200, content=serialised_matches)
    )

    # Use lookup function
    matches = dag.lookup_key(from_source="foo", to_sources=["bar", "baz"], key="pk1")

    # Verify results
    assert matches == {foo.name: ["a"], bar.name: ["b"], baz.name: ["b"]}


def test_lookup_key_404_source(matchbox_api: MockRouter):
    """Key lookup throws a resolution not found error."""
    # Set up dummy data
    source_testkit = source_factory(name="source")
    target_testkit = source_factory(name="target")

    dag = TestkitDAG().dag

    dag.source(**source_testkit.into_dag()).query().linker(
        dag.source(**target_testkit.into_dag()).query(),
        name="root",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    matchbox_api.get("/match").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendResourceType.RESOLUTION,
            ).model_dump(),
        )
    )

    # Use match function
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        dag.lookup_key(from_source="source", to_sources=["target"], key="pk1")


def test_lookup_key_no_matches(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Key lookup raises MatchboxEmptyServerResponse when no matches are found."""
    # Set up dummy data
    source_testkit = source_factory(
        engine=sqlite_warehouse, name="source"
    ).write_to_location()
    target_testkit = source_factory(
        engine=sqlite_warehouse, name="target"
    ).write_to_location()

    dag = TestkitDAG().dag

    dag.source(**source_testkit.into_dag()).query().linker(
        dag.source(**target_testkit.into_dag()).query(),
        name="root",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    # Mock empty match results
    matchbox_api.get("/match").mock(return_value=Response(200, content="[]"))

    # Test that empty match results raise MatchboxEmptyServerResponse
    with pytest.raises(
        MatchboxEmptyServerResponse, match="The match operation returned no data"
    ):
        dag.lookup_key(from_source="source", to_sources=["target"], key="pk1")


def test_from_resolution():
    """Test reconstructing Sources and Models from a Resolution."""
    # Create test data
    test_dag = TestkitDAG().dag

    # Create test sources and model
    linked_testkit = linked_sources_factory(dag=test_dag)
    crn_testkit = linked_testkit.sources["crn"]
    duns_testkit = linked_testkit.sources["duns"]

    deduper_model_testkit = model_factory(
        name="deduper",
        left_testkit=crn_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    )
    linker_model_testkit = model_factory(
        name="linker",
        left_testkit=deduper_model_testkit,
        right_testkit=duns_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    )

    # Add to DAG
    test_dag.source(**crn_testkit.into_dag())
    test_dag.source(**duns_testkit.into_dag())
    test_dag.model(**deduper_model_testkit.into_dag())
    test_dag.model(**linker_model_testkit.into_dag())

    # Test 1: Add all resolutions to the DAG in order
    t1_dag = TestkitDAG().dag

    for testkit in [crn_testkit, duns_testkit]:
        t1_dag.add_resolution(
            name=testkit.name, resolution=testkit.source.to_resolution()
        )
    for testkit in [deduper_model_testkit, linker_model_testkit]:
        t1_dag.add_resolution(
            name=testkit.name, resolution=testkit.model.to_resolution()
        )

    # Verify reconstruction matches original
    assert t1_dag.name == test_dag.name
    assert t1_dag.run == test_dag.run
    for name, resolution in t1_dag.nodes.items():
        assert resolution.config == test_dag.nodes[name].config
    assert t1_dag.graph == test_dag.graph

    # Test 2: Add resolutions out of order
    t2_dag = TestkitDAG().dag

    with pytest.raises(ValueError, match="not found in DAG"):
        t2_dag.add_resolution(
            name=linker_model_testkit.name,
            resolution=linker_model_testkit.model.to_resolution(),
        )


def test_dag_creates_new_collection(
    matchbox_api: MockRouter,
    sqlite_warehouse: Engine,
):
    """Connect creates a new collection when it doesn't exist."""
    dag = DAG(name="test_collection")

    # Mock collection not found, then found after creation
    matchbox_api.get("/collections/test_collection").mock(
        side_effect=[
            Response(
                404,
                json=NotFoundError(
                    details="Collection not found",
                    entity=BackendResourceType.COLLECTION,
                ).model_dump(),
            ),
            Response(
                200,
                json=Collection(
                    name="test_collection",
                    runs=[],
                    default_run=None,
                ).model_dump(),
            ),
        ]
    )

    # Mock collection creation
    matchbox_api.post("/collections/test_collection").mock(
        return_value=Response(
            200,
            json=ResourceOperationStatus(
                success=True,
                name="test_collection",
                operation=CRUDOperation.CREATE,
            ).model_dump(),
        )
    )

    # Mock run creation
    matchbox_api.post("/collections/test_collection/runs").mock(
        return_value=Response(
            200,
            json=Run(run_id=1, resolutions={}).model_dump(),
        )
    )

    # Connect the DAG
    result = dag.new_run()

    # Verify
    assert result == dag
    assert dag.run == 1


@pytest.mark.parametrize(
    ("has_existing_runs", "expected_run_id"),
    [
        pytest.param(False, 1, id="no_existing_runs"),
        pytest.param(True, 4, id="with_existing_runs"),
    ],
)
def test_dag_uses_existing_collection(
    matchbox_api: MockRouter, has_existing_runs: bool, expected_run_id: int
):
    """New runs can be started from existing collection."""
    dag = DAG(name="test_collection")

    # Mock existing collection
    existing_runs = [2, 3] if has_existing_runs else []
    matchbox_api.get("/collections/test_collection").mock(
        return_value=Response(
            200,
            json=Collection(
                name="test_collection",
                runs=existing_runs,
                default_run=None,
            ).model_dump(),
        )
    )

    # Mock deleting non-default runs
    if has_existing_runs:
        for run_id in existing_runs:
            matchbox_api.delete(f"/collections/test_collection/runs/{run_id}").mock(
                return_value=Response(
                    200,
                    json=ResourceOperationStatus(
                        success=True,
                        name=run_id,
                        operation=CRUDOperation.DELETE,
                    ).model_dump(),
                )
            )

    # Mock run creation
    matchbox_api.post("/collections/test_collection/runs").mock(
        return_value=Response(
            200,
            json=Run(run_id=expected_run_id, resolutions={}).model_dump(),
        )
    )

    # Connect the DAG
    result = dag.new_run()

    # Verify
    assert result == dag
    assert dag.run == expected_run_id


def test_dag_load_default_run(matchbox_api: MockRouter):
    """Can load default run with sources and optionally models."""
    # Create test data
    test_dag = TestkitDAG().dag

    # Create test sources and model
    linked_testkit = linked_sources_factory(dag=test_dag)
    crn_testkit = linked_testkit.sources["crn"]
    duns_testkit = linked_testkit.sources["duns"]

    deduper_model_testkit = model_factory(
        name="deduper",
        left_testkit=crn_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    )
    linker_model_testkit = model_factory(
        name="linker",
        left_testkit=deduper_model_testkit,
        right_testkit=duns_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    )

    # Add to DAG
    test_dag.source(**crn_testkit.into_dag())
    test_dag.source(**duns_testkit.into_dag())
    test_dag.model(**deduper_model_testkit.into_dag())
    test_dag.model(**linker_model_testkit.into_dag())

    # Create default Run
    resolutions: dict[ResolutionName, Resolution] = {
        crn_testkit.name: crn_testkit.source.to_resolution(),
        duns_testkit.name: duns_testkit.source.to_resolution(),
        deduper_model_testkit.name: deduper_model_testkit.model.to_resolution(),
        linker_model_testkit.name: linker_model_testkit.model.to_resolution(),
    }

    run = Run(run_id=1, resolutions=resolutions)

    # Mock existing collection with default run
    matchbox_api.get(f"/collections/{test_dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(
                name=test_dag.name,
                runs=[1, 2],
                default_run=1,
            ).model_dump(),
        )
    )

    # Mock getting default run
    matchbox_api.get(f"/collections/{test_dag.name}/runs/1").mock(
        return_value=Response(
            200,
            json=run.model_dump(),
        )
    )

    # Mock getting pending run
    matchbox_api.get(f"/collections/{test_dag.name}/runs/2").mock(
        return_value=Response(
            200,
            json=run.model_dump(),
        )
    )

    # Load default run
    default_dag = DAG(name=test_dag.name)
    default_dag = default_dag.load_default()

    # Verify reconstruction matches original
    assert default_dag.name == test_dag.name
    assert default_dag.run == 1
    assert set(default_dag.nodes.keys()) == set(test_dag.nodes.keys())
    assert default_dag.graph == test_dag.graph

    # Load pending run
    pending_dag = DAG(name=test_dag.name)
    pending_dag = pending_dag.load_pending()

    # Verify reconstruction matches original
    assert pending_dag.name == test_dag.name
    assert pending_dag.run == 2
    assert set(pending_dag.nodes.keys()) == set(test_dag.nodes.keys())
    assert pending_dag.graph == test_dag.graph

    # If the collection is not available, errors
    matchbox_api.get(f"/collections/{test_dag.name}/runs/1").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Collection not found",
                entity=BackendResourceType.COLLECTION,
            ).model_dump(),
        ),
    )

    with pytest.raises(MatchboxCollectionNotFoundError):
        DAG(name=test_dag.name).load_default()


def test_dag_set_client(sqlite_warehouse: Engine):
    """Client can be set for all sources at once."""
    # Create factory data
    foo_params = source_factory(name="foo").into_dag()
    bar_params = source_factory(name="bar").into_dag()

    # Create new DAG
    dag = DAG(name="dag")
    dag.source(**foo_params)
    dag.source(**bar_params)

    # Setting client re-assigns all clients
    assert dag.get_source("foo").location.client != sqlite_warehouse
    assert dag.get_source("bar").location.client != sqlite_warehouse
    dag.set_client(sqlite_warehouse)
    assert dag.get_source("foo").location.client == sqlite_warehouse
    assert dag.get_source("bar").location.client == sqlite_warehouse


def test_dag_set_default_ok(matchbox_api: MockRouter):
    """Set default makes run immutable and sets as default."""
    # Create test data
    dag = TestkitDAG().dag

    # Mock set mutable
    api_mutable = matchbox_api.patch(
        f"/collections/{dag.name}/runs/{dag.run}/mutable"
    ).mock(
        return_value=Response(
            200,
            json=ResourceOperationStatus(
                success=True,
                name=dag.run,
                operation=CRUDOperation.UPDATE,
            ).model_dump(),
        )
    )

    # Mock set default
    api_default = matchbox_api.patch(
        f"/collections/{dag.name}/runs/{dag.run}/default"
    ).mock(
        return_value=Response(
            200,
            json=ResourceOperationStatus(
                success=True,
                name=dag.run,
                operation=CRUDOperation.UPDATE,
            ).model_dump(),
        )
    )

    # Set as default
    dag.set_default()

    # Verify both endpoints were called
    assert api_mutable.called
    assert api_default.called


def test_dag_set_default_not_connected():
    """Set default raises error when DAG is not connected."""
    dag = DAG(name="test_collection")

    with pytest.raises(RuntimeError, match="has not been connected"):
        dag.set_default()
