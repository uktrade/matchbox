import json
from unittest.mock import Mock, patch

import polars as pl
import pyarrow as pa
import pytest
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine

from matchbox.client.dags import DAG, DAGNodeExecutionStatus
from matchbox.client.models import Model
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.sources import Source
from matchbox.common.arrow import SCHEMA_QUERY_WITH_LEAVES, table_to_buffer
from matchbox.common.dtos import (
    Collection,
    CRUDOperation,
    ErrorResponse,
    Match,
    Resolution,
    ResolutionName,
    ResolutionPath,
    ResolutionType,
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
)


def test_dag_list(matchbox_api: MockRouter) -> None:
    """Can retrieve list of DAG names."""
    dummy_names = ["companies", "contacts"]
    matchbox_api.get("/collections").mock(return_value=Response(200, json=dummy_names))
    assert DAG.list_all() == ["companies", "contacts"]


@patch.object(Source, "run")
@patch.object(Model, "run")
@patch.object(Source, "sync")
@patch.object(Model, "sync")
@patch.object(Source, "clear_data")
@patch.object(Model, "clear_data")
def test_dag_run_and_sync(
    model_clear_mock: Mock,
    source_clear_mock: Mock,
    model_sync_mock: Mock,
    source_sync_mock: Mock,
    model_run_mock: Mock,
    source_run_mock: Mock,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """A legal DAG can be built and run."""
    # Set up constituents
    foo_tkit = source_factory(
        name="foo", engine=sqla_sqlite_warehouse
    ).write_to_location()
    bar_tkit = source_factory(
        name="bar", engine=sqla_sqlite_warehouse
    ).write_to_location()
    baz_tkit = source_factory(
        name="baz", engine=sqla_sqlite_warehouse
    ).write_to_location()

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
    source_clear_mock.assert_not_called()
    model_clear_mock.assert_not_called()

    # Running DAG destroys intermediate results
    dag.run_and_sync(low_memory=True)
    assert source_clear_mock.call_count == 3
    assert model_clear_mock.call_count == 3


def test_dags_missing_dependency(sqla_sqlite_warehouse: Engine) -> None:
    """Steps cannot be added before their dependencies."""
    dag = TestkitDAG().dag

    foo = source_factory(name="foo", engine=sqla_sqlite_warehouse, dag=dag).source

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


def test_mixing_dags_fails(sqla_sqlite_warehouse: Engine) -> None:
    """Cannot reference a different DAG when adding a step."""
    dag = TestkitDAG().dag
    dag2 = TestkitDAG().dag

    foo_tkit = source_factory(name="foo", engine=sqla_sqlite_warehouse)
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


def test_dag_name_clash(sqla_sqlite_warehouse: Engine) -> None:
    """Under some conditions, nodes can be overwritten."""
    dag = TestkitDAG().dag

    foo_tkit = source_factory(name="foo", engine=sqla_sqlite_warehouse)
    bar_tkit = source_factory(name="bar", engine=sqla_sqlite_warehouse)

    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())

    d_foo = foo.query().deduper(
        name="d_foo",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [foo.f(foo.index_fields[0].name)]},
    )

    # Can re-define nodes, in whatever order
    d_foo = foo.query().deduper(
        name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )
    updated_foo_args = foo_tkit.into_dag()
    updated_foo_args["description"] = "new description"
    foo = dag.source(**updated_foo_args)

    assert dag.get_model("d_foo").model_settings.unique_fields == []
    assert dag.get_source(foo_tkit.name).description == "new description"

    # Cannot overwrite source with model
    with pytest.raises(ValueError, match="Cannot re-assign"):
        foo.query().deduper(
            name=foo_tkit.name,
            model_class=NaiveDeduper,
            model_settings={"unique_fields": []},
        )

    # Cannot overwrite model with source
    updated_foo_args = foo_tkit.into_dag()
    updated_foo_args["name"] = "d_foo"
    with pytest.raises(ValueError, match="Cannot re-assign"):
        dag.source(**updated_foo_args)

    # Cannot change inputs of model
    linker = foo.query().linker(
        bar.query(),
        name="linker",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    baz_tkit = source_factory(name="baz", engine=sqla_sqlite_warehouse)
    baz = dag.source(**baz_tkit.into_dag())
    with pytest.raises(ValueError, match="Cannot re-assign"):
        foo.query().linker(
            # Linking different source using linker whose name is taken
            baz.query(),
            name="linker",
            model_class=DeterministicLinker,
            model_settings={"comparisons": "l.field=r.field"},
        )

    # After failed attempts, DAG is as we expect
    assert dag.get_source(foo.name) == foo
    assert dag.get_source(bar.name) == bar
    assert dag.get_source(baz.name) == baz
    assert dag.get_model(d_foo.name) == d_foo
    assert dag.get_model(linker.name) == linker

    assert dag.graph[foo.name] == []
    assert dag.graph[bar.name] == []
    assert dag.graph[baz.name] == []
    assert dag.graph[d_foo.name] == [foo.name]
    assert dag.graph[linker.name] == [foo.name, bar.name]


def test_dag_final_steps(sqla_sqlite_warehouse: Engine) -> None:
    """Test final_steps property returns all apex nodes."""
    dag = TestkitDAG().dag

    # Empty DAG has no final steps
    assert dag.final_steps == []

    # Single apex
    foo_tkit = source_factory(name="foo", engine=sqla_sqlite_warehouse)
    foo = dag.source(**foo_tkit.into_dag())
    assert dag.final_steps == [foo]

    # Multiple apexes (disconnected)
    bar_tkit = source_factory(name="bar", engine=sqla_sqlite_warehouse)
    bar = dag.source(**bar_tkit.into_dag())
    assert set(dag.final_steps) == {foo, bar}

    # After linking, single apex again
    foo.query().linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    assert len(dag.final_steps) == 1
    assert dag.final_steps[0].name == "foo_bar"


def test_dag_draw(sqla_sqlite_warehouse: Engine) -> None:
    """Test that the draw method produces a correct string representation of the DAG."""
    # Set up a simple DAG
    foo_tkit = source_factory(
        name="foo", engine=sqla_sqlite_warehouse
    ).write_to_location()
    bar_tkit = source_factory(
        name="bar", engine=sqla_sqlite_warehouse
    ).write_to_location()
    baz_tkit = source_factory(
        name="baz", engine=sqla_sqlite_warehouse
    ).write_to_location()

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
    head_lines, tree_lines = lines[:3], lines[3:]
    assert "Collection" in head_lines[0]
    assert "Run" in head_lines[1]

    # The root node should be first
    assert tree_lines[0] == "foo_bar_baz"

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
        node_present = any(line.endswith(node) for line in tree_lines)
        assert node_present, f"Node {node} not found in the tree representation"

    # Check that tree has correct formatting with tree characters
    tree_chars = ["└──", "├──", "│"]
    has_tree_chars = any(char in tree_str for char in tree_chars)
    assert has_tree_chars, (
        "Tree representation doesn't use expected formatting characters"
    )

    # Test 2: Drawing with status indicators

    tree_str_with_status = dag.draw(
        status={
            foo_bar.name: DAGNodeExecutionStatus.DOING,
            d_foo.name: DAGNodeExecutionStatus.DONE,
        }
    )
    status_lines = tree_str_with_status.strip().split("\n")[3:]

    # Verify status indicators are present
    status_indicators = ["✅", "🔄", "⏸️"]
    assert any(indicator in tree_str_with_status for indicator in status_indicators)

    # Check specific statuses: foo_bar done, d_foo working, others awaiting
    for line in status_lines:
        name = line.split()[-1]
        if name == d_foo.name:
            assert "✅" in line
        elif name == foo_bar.name:
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
        status={node: DAGNodeExecutionStatus.SKIPPED for node in skipped_nodes}
    )
    skipped_lines = tree_str_with_skipped.strip().split("\n")[3:]

    # Check that skipped nodes have the skipped indicator
    for line in skipped_lines:
        name = line.split()[-1]
        if any(name == skipped for skipped in skipped_nodes):
            assert "⏭️" in line

    # Test all status indicators together
    tree_str_all_statuses = dag.draw(
        status={
            foo_bar.name: DAGNodeExecutionStatus.DOING,
            d_foo.name: DAGNodeExecutionStatus.DONE,
            foo.name: DAGNodeExecutionStatus.SKIPPED,
        }
    )
    assert all(
        indicator in tree_str_all_statuses for indicator in ["✅", "🔄", "⏸️", "⏭️"]
    )

    # Test 5: Empty DAG
    empty_dag = TestkitDAG().dag
    assert empty_dag.draw() == "Empty DAG"

    # Test 6: __str__ method
    assert str(dag) == dag.draw()

    # Test 7: Multiple disconnected components
    disconnected_dag = TestkitDAG().dag
    qux_tkit = source_factory(name="qux", engine=sqla_sqlite_warehouse)
    quux_tkit = source_factory(name="quux", engine=sqla_sqlite_warehouse)
    _ = disconnected_dag.source(**qux_tkit.into_dag())
    _ = disconnected_dag.source(**quux_tkit.into_dag())

    tree_str = disconnected_dag.draw()
    # Both apex nodes should be present
    assert "qux" in tree_str
    assert "quux" in tree_str
    # Should have blank line between components
    assert "\n\n" in tree_str


# Lookups


def test_resolve(matchbox_api: MockRouter) -> None:
    """Resolved data can be generated from DAG."""
    # Make dummy data
    foo = source_factory(name="foo", location_name="sqlite")
    bar = source_factory(name="bar", location_name="postgres")
    baz = source_factory(name="baz", location_name="postgres")

    # Nothing is merged - that's OK for this test
    foo_data = pa.Table.from_pylist(
        [
            {"id": 1, "leaf_id": 1, "key": "1"},
            {"id": 2, "leaf_id": 2, "key": "2"},
        ],
        schema=SCHEMA_QUERY_WITH_LEAVES,
    )

    bar_data = pa.Table.from_pylist(
        [
            {"id": 3, "leaf_id": 3, "key": "a"},
            {"id": 4, "leaf_id": 4, "key": "b"},
        ],
        schema=SCHEMA_QUERY_WITH_LEAVES,
    )

    baz_data = pa.Table.from_pylist(
        [
            {"id": 5, "leaf_id": 5, "key": "x"},
            {"id": 6, "leaf_id": 6, "key": "y"},
        ],
        schema=SCHEMA_QUERY_WITH_LEAVES,
    )

    dag = DAG("companies")

    # Mock API
    # In the beginning, no run
    matchbox_api.get(f"/collections/{dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(name=dag.name, runs=[], default_run=None).model_dump(),
        )
    )

    matchbox_api.post(f"/collections/{dag.name}/runs").mock(
        return_value=Response(
            200,
            json=Run(run_id=1, resolutions={}).model_dump(),
        )
    )

    # Build dummy DAG
    foo_source = dag.source(**foo.into_dag())
    bar_source = dag.source(**bar.into_dag())
    baz_source = dag.source(**baz.into_dag())
    foo_bar = foo_source.query().linker(
        bar_source.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    foo_bar_baz = foo_bar.query(foo, bar).linker(
        baz_source.query(),
        name="foo_bar_baz",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    dag.new_run()

    # Then the new run
    matchbox_api.get(f"/collections/{dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(name=dag.name, runs=[1], default_run=1).model_dump(),
        )
    )

    matchbox_api.get(f"/collections/{dag.name}/runs/1").mock(
        return_value=Response(
            200,
            json=Run(
                run_id=1,
                resolutions={
                    foo.name: foo.fake_run().source.to_resolution(),
                    bar.name: bar.fake_run().source.to_resolution(),
                    baz.name: baz.fake_run().source.to_resolution(),
                    foo_bar.name: Resolution(
                        fingerprint=b"mock",
                        truth=1,
                        resolution_type=ResolutionType.MODEL,
                        config=foo_bar.config,
                    ),
                    foo_bar_baz.name: Resolution(
                        fingerprint=b"mock",
                        truth=1,
                        resolution_type=ResolutionType.MODEL,
                        config=foo_bar_baz.config,
                    ),
                },
            ).model_dump(),
        )
    )

    # Return dummy query data for all sources
    matchbox_api.get(
        "/query", params={"source": "foo", "run_id": 1, "collection": dag.name}
    ).mock(return_value=Response(200, content=table_to_buffer(foo_data).read()))

    matchbox_api.get(
        "/query", params={"source": "bar", "run_id": 1, "collection": dag.name}
    ).mock(return_value=Response(200, content=table_to_buffer(bar_data).read()))

    matchbox_api.get(
        "/query", params={"source": "baz", "run_id": 1, "collection": dag.name}
    ).mock(return_value=Response(200, content=table_to_buffer(baz_data).read()))

    # No sources are found
    with pytest.raises(MatchboxResolutionNotFoundError):
        dag.resolve(source_filter=["nonexistent"])

    with pytest.raises(MatchboxResolutionNotFoundError):
        dag.resolve(location_names=["nonexistent"])

    # With URI filter
    uri_filter_resolved = dag.resolve(location_names=["sqlite"])

    assert len(uri_filter_resolved.sources) == 1
    assert uri_filter_resolved.sources[0].name == "foo"

    # With source filter
    source_filter_resolved = dag.resolve(source_filter="foo")

    assert len(source_filter_resolved.sources) == 1
    assert source_filter_resolved.sources[0].name == "foo"

    # Select intermediate model
    intermediate_res = dag.resolve(node=foo_bar.name)
    assert len(intermediate_res.sources) == 2
    assert set([s.name for s in intermediate_res.sources]) == {foo.name, bar.name}

    # With no filter
    full_resolved = dag.resolve()
    assert len(full_resolved.sources) == 3
    assert len(full_resolved.query_results) == 3
    source_names = [
        full_resolved.sources[0].name,
        full_resolved.sources[1].name,
        full_resolved.sources[2].name,
    ]
    foo_index = source_names.index("foo")
    bar_index = source_names.index("bar")
    baz_index = source_names.index("baz")

    assert_frame_equal(full_resolved.query_results[foo_index], pl.from_arrow(foo_data))
    assert_frame_equal(full_resolved.query_results[bar_index], pl.from_arrow(bar_data))
    assert_frame_equal(full_resolved.query_results[baz_index], pl.from_arrow(baz_data))

    # Retrieve from reconstituted DAG
    loaded_res = DAG("companies").load_default().resolve()
    assert len(loaded_res.sources) == 3
    assert len(loaded_res.query_results) == 3
    source_names = [
        loaded_res.sources[0].name,
        loaded_res.sources[1].name,
        loaded_res.sources[2].name,
    ]
    foo_index = source_names.index("foo")
    bar_index = source_names.index("bar")
    baz_index = source_names.index("baz")

    assert_frame_equal(loaded_res.query_results[foo_index], pl.from_arrow(foo_data))
    assert_frame_equal(loaded_res.query_results[bar_index], pl.from_arrow(bar_data))
    assert_frame_equal(loaded_res.query_results[baz_index], pl.from_arrow(baz_data))


def test_lookup_key_ok(matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine) -> None:
    """The DAG can map between single keys."""
    # Set up dummy data
    foo_testkit = source_factory(
        engine=sqla_sqlite_warehouse, name="foo"
    ).write_to_location()
    bar_testkit = source_factory(
        engine=sqla_sqlite_warehouse, name="bar"
    ).write_to_location()
    baz_testkit = source_factory(
        engine=sqla_sqlite_warehouse, name="baz"
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


def test_lookup_key_404_source(matchbox_api: MockRouter) -> None:
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
            json=ErrorResponse(
                exception_type="MatchboxResolutionNotFoundError",
                message="Resolution 42 not found",
            ).model_dump(),
        )
    )

    # Use match function
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        dag.lookup_key(from_source="source", to_sources=["target"], key="pk1")


def test_lookup_key_no_matches(
    matchbox_api: MockRouter, sqla_sqlite_warehouse: Engine
) -> None:
    """Key lookup raises MatchboxEmptyServerResponse when no matches are found."""
    # Set up dummy data
    source_testkit = source_factory(
        engine=sqla_sqlite_warehouse, name="source"
    ).write_to_location()
    target_testkit = source_factory(
        engine=sqla_sqlite_warehouse, name="target"
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


def test_from_resolution() -> None:
    """Test reconstructing Sources and Models from a Resolution."""
    # Create test data
    test_dag = TestkitDAG().dag

    # Create test sources and model
    linked_testkit = linked_sources_factory(dag=test_dag)
    crn_testkit = linked_testkit.sources["crn"]
    dh_testkit = linked_testkit.sources["dh"]

    deduper_model_testkit = model_factory(
        name="deduper",
        left_testkit=crn_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    )
    linker_model_testkit = model_factory(
        name="linker",
        left_testkit=deduper_model_testkit,
        right_testkit=dh_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    )

    # Add to DAG
    test_dag.source(**crn_testkit.into_dag())
    test_dag.source(**dh_testkit.into_dag())
    test_dag.model(**deduper_model_testkit.into_dag())
    test_dag.model(**linker_model_testkit.into_dag())

    # Test 1: Add all resolutions to the DAG in order
    t1_dag = TestkitDAG().dag

    for testkit in [crn_testkit, dh_testkit]:
        t1_dag.add_resolution(
            name=testkit.name, resolution=testkit.fake_run().source.to_resolution()
        )
    for testkit in [deduper_model_testkit, linker_model_testkit]:
        t1_dag.add_resolution(
            name=testkit.name, resolution=testkit.fake_run().model.to_resolution()
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
            resolution=linker_model_testkit.fake_run().model.to_resolution(),
        )


def test_dag_creates_new_collection(
    matchbox_api: MockRouter,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Connect creates a new collection when it doesn't exist."""
    dag = DAG(name="test_collection")

    # Mock collection not found, then found after creation
    matchbox_api.get("/collections/test_collection").mock(
        side_effect=[
            Response(
                404,
                json=ErrorResponse(
                    exception_type="MatchboxCollectionNotFoundError",
                    message="Collection not found",
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
                target="Collection test_collection",
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
) -> None:
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
                        target=f"Run {run_id}",
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


def test_dag_load_server_run(matchbox_api: MockRouter) -> None:
    """Can retrieve serialised DAG from the server."""
    # Create test data
    test_dag = TestkitDAG().dag

    # Create test sources and model
    linked_testkit = linked_sources_factory(dag=test_dag)
    crn_testkit = linked_testkit.sources["crn"].fake_run()
    dh_testkit = linked_testkit.sources["dh"].fake_run()

    deduper_model_testkit = model_factory(
        name="deduper",
        left_testkit=crn_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    ).fake_run()
    linker_model_testkit = model_factory(
        name="linker",
        left_testkit=deduper_model_testkit,
        right_testkit=dh_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    ).fake_run()

    # Add to DAG
    test_dag.source(**crn_testkit.into_dag())
    test_dag.source(**dh_testkit.into_dag())
    test_dag.model(**deduper_model_testkit.into_dag())
    test_dag.model(**linker_model_testkit.into_dag())

    # Create default Run
    resolutions: dict[ResolutionName, Resolution] = {
        crn_testkit.name: crn_testkit.source.to_resolution(),
        dh_testkit.name: dh_testkit.source.to_resolution(),
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

    # Load default run
    default_dag = DAG(name=test_dag.name)
    default_dag = default_dag.load_default()

    # Verify reconstruction matches original
    assert default_dag.name == test_dag.name
    assert default_dag.run == 1
    assert set(default_dag.nodes.keys()) == set(test_dag.nodes.keys())
    assert default_dag.graph == test_dag.graph

    # Mock getting pending run
    matchbox_api.get(f"/collections/{test_dag.name}/runs/2").mock(
        return_value=Response(
            200,
            json=run.model_dump(),
        )
    )

    pending_dag = DAG(name=test_dag.name)
    pending_dag = pending_dag.load_pending()

    # Verify reconstruction matches original
    assert pending_dag.name == test_dag.name
    assert pending_dag.run == 2
    assert set(pending_dag.nodes.keys()) == set(test_dag.nodes.keys())
    assert pending_dag.graph == test_dag.graph

    # Compatible nodes are updated when loading pending
    overwritten_dag = DAG(name=test_dag.name)
    overwritten_source = overwritten_dag.source(**crn_testkit.into_dag())
    overwritten_source.description = "new description"
    overwritten_dag.load_pending()
    # Description is overwritten
    assert (
        overwritten_dag.get_source(crn_testkit.name).description
        == crn_testkit.source.description
    )

    # Cannot load pending into local DAG that alters the graph
    clashing_dag = DAG(name=test_dag.name)
    crn = clashing_dag.source(**crn_testkit.into_dag())
    dh = clashing_dag.source(**dh_testkit.into_dag())

    # This linker skips the deduper defined above
    crn.query().linker(
        dh.query(),
        name=linker_model_testkit.name,
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    with pytest.raises(ValueError, match="Cannot re-assign"):
        clashing_dag.load_pending()

    # If the collection is not available, errors
    matchbox_api.get(f"/collections/{test_dag.name}/runs/1").mock(
        return_value=Response(
            404,
            json=ErrorResponse(
                exception_type="MatchboxCollectionNotFoundError",
                message="Collection not found",
            ).model_dump(),
        ),
    )

    with pytest.raises(MatchboxCollectionNotFoundError):
        DAG(name=test_dag.name).load_default()


def test_dag_load_run_complex_dependencies(matchbox_api: MockRouter) -> None:
    """Test that _load_run handles nodes with complex dependencies.

    In particular, tests for DAGs where nodes have the same dependency count but
    inter-dependencies between them.
    """
    test_dag = TestkitDAG().dag

    linked_testkit = linked_sources_factory(dag=test_dag)
    crn_testkit = linked_testkit.sources["crn"].fake_run()
    dh_testkit = linked_testkit.sources["dh"].fake_run()
    cdms_testkit = linked_testkit.sources["cdms"].fake_run()

    # Model_inner depends on 2 sources (count=2)
    model_inner_testkit = model_factory(
        name="model_inner",
        left_testkit=crn_testkit,
        right_testkit=dh_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    ).fake_run()

    # Model_outer ALSO depends on 2 things: model_inner + a source (count=2)
    # Same count as model_inner, but MUST be loaded AFTER model_inner
    model_outer_testkit = model_factory(
        name="model_outer",
        left_testkit=model_inner_testkit,  # depends on model_inner!
        right_testkit=cdms_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    ).fake_run()

    # Add to original DAG
    test_dag.source(**crn_testkit.into_dag())
    test_dag.source(**dh_testkit.into_dag())
    test_dag.source(**cdms_testkit.into_dag())
    test_dag.model(**model_inner_testkit.into_dag())
    test_dag.model(**model_outer_testkit.into_dag())

    # Create resolutions with model_outer FIRST in dict
    resolutions: dict[ResolutionName, Resolution] = {
        model_outer_testkit.name: model_outer_testkit.model.to_resolution(),  # 2 deps
        crn_testkit.name: crn_testkit.source.to_resolution(),  # 0 deps
        dh_testkit.name: dh_testkit.source.to_resolution(),  # 0 deps
        cdms_testkit.name: cdms_testkit.source.to_resolution(),  # 0 deps
        model_inner_testkit.name: model_inner_testkit.model.to_resolution(),  # 2 deps
    }

    run = Run(run_id=1, resolutions=resolutions)

    # Mock API
    matchbox_api.get(f"/collections/{test_dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(
                name=test_dag.name,
                runs=[1],
                default_run=1,
            ).model_dump(),
        )
    )

    matchbox_api.get(f"/collections/{test_dag.name}/runs/1").mock(
        return_value=Response(
            200,
            json=run.model_dump(),
        )
    )

    # Load the run
    loaded_dag = DAG(name=test_dag.name)
    loaded_dag = loaded_dag.load_default()

    # Verify all nodes were loaded correctly
    assert loaded_dag.run == 1
    assert set(loaded_dag.nodes.keys()) == set(test_dag.nodes.keys())
    assert loaded_dag.graph == test_dag.graph


def test_dag_set_client(sqla_sqlite_warehouse: Engine) -> None:
    """Client can be set for all sources at once."""
    # Create factory data
    foo_params = source_factory(name="foo").into_dag()
    bar_params = source_factory(name="bar").into_dag()

    # Create new DAG
    dag = DAG(name="dag")
    dag.source(**foo_params)
    dag.source(**bar_params)

    # Setting client re-assigns all clients
    assert dag.get_source("foo").location.client != sqla_sqlite_warehouse
    assert dag.get_source("bar").location.client != sqla_sqlite_warehouse
    dag.set_client(sqla_sqlite_warehouse)
    assert dag.get_source("foo").location.client == sqla_sqlite_warehouse
    assert dag.get_source("bar").location.client == sqla_sqlite_warehouse


def test_dag_set_default_ok(
    matchbox_api: MockRouter,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Set default makes run immutable and sets as default."""
    # Create test data
    dag = TestkitDAG().dag

    dag.source(**source_factory().into_dag())

    # Mock set mutable
    api_mutable = matchbox_api.patch(
        f"/collections/{dag.name}/runs/{dag.run}/mutable"
    ).mock(
        return_value=Response(
            200,
            json=ResourceOperationStatus(
                success=True,
                target=f"Run {dag.run}",
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
                target=f"Run {dag.run}",
                operation=CRUDOperation.UPDATE,
            ).model_dump(),
        )
    )

    # Set as default
    dag.set_default()

    # Verify both endpoints were called
    assert api_mutable.called
    assert api_default.called


def test_dag_set_default_not_connected() -> None:
    """Set default raises error when DAG is not connected to server."""
    dag = DAG(name="test_collection")
    dag.source(**source_factory().into_dag())

    with pytest.raises(RuntimeError, match="has not been connected"):
        dag.set_default()


def test_dag_set_default_unreachable_nodes(sqla_sqlite_warehouse: Engine) -> None:
    """Nodes cannot be unreachable from root when setting a default run."""
    dag = TestkitDAG().dag

    foo_tkit = source_factory(name="foo", engine=sqla_sqlite_warehouse)
    bar_tkit = source_factory(name="bar", engine=sqla_sqlite_warehouse)

    dag.source(**foo_tkit.into_dag())
    dag.source(**bar_tkit.into_dag())

    with pytest.raises(ValueError, match="unreachable"):
        dag.set_default()
