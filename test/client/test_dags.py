"""Tests for DAG construction, orchestration, and lookup behaviour in the client."""

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
from matchbox.client.resolvers import Components, Resolver
from matchbox.client.sources import Source
from matchbox.common.arrow import SCHEMA_QUERY_WITH_LEAVES, table_to_buffer
from matchbox.common.dtos import (
    Collection,
    CRUDOperation,
    DefaultGroup,
    ErrorResponse,
    Match,
    PermissionGrant,
    PermissionType,
    ResourceOperationStatus,
    Run,
    Step,
    StepName,
    StepPath,
    StepType,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxEmptyServerResponse,
    MatchboxStepNotFoundError,
    MatchboxStepTypeError,
)
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.resolvers import resolver_factory
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
@patch.object(Resolver, "run")
@patch.object(Source, "sync")
@patch.object(Model, "sync")
@patch.object(Resolver, "sync")
@patch.object(Source, "clear_data")
@patch.object(Model, "clear_data")
@patch.object(Resolver, "clear_data")
def test_dag_run_and_sync(
    resolver_clear_mock: Mock,
    model_clear_mock: Mock,
    source_clear_mock: Mock,
    resolver_sync_mock: Mock,
    model_sync_mock: Mock,
    source_sync_mock: Mock,
    resolver_run_mock: Mock,
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
    foo_bar = foo.query().linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    # Additional independent model
    foo_baz = foo.query().linker(
        baz.query(),
        name="foo_baz",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )

    root = foo_bar.resolver(d_foo, foo_baz, name="root", resolver_class=Components)

    assert set(dag.nodes.keys()) == {
        foo.name,
        bar.name,
        baz.name,
        d_foo.name,
        foo_bar.name,
        foo_baz.name,
        root.name,
    }

    # Run DAG
    dag.run_and_sync()

    assert source_run_mock.call_count == 3
    assert source_sync_mock.call_count == 3
    assert model_run_mock.call_count == 3
    assert model_sync_mock.call_count == 3
    assert resolver_run_mock.call_count == 1
    assert resolver_sync_mock.call_count == 1
    source_clear_mock.assert_not_called()
    model_clear_mock.assert_not_called()
    resolver_clear_mock.assert_not_called()

    # Running DAG destroys intermediate results
    dag.run_and_sync(low_memory=True)
    assert source_clear_mock.call_count == 3
    assert model_clear_mock.call_count == 3
    assert resolver_clear_mock.call_count == 1


@patch.object(Source, "sync")
@patch.object(Model, "sync")
@patch.object(Resolver, "sync")
@patch.object(Source, "run", autospec=True)
@patch.object(Model, "run", autospec=True)
@patch.object(Resolver, "run", autospec=True)
def test_dag_run_and_sync_orders_shared_parents_before_children(
    resolver_run_mock: Mock,
    model_run_mock: Mock,
    source_run_mock: Mock,
    resolver_sync_mock: Mock,
    model_sync_mock: Mock,
    source_sync_mock: Mock,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Execution order ensures all structural parents run before their children."""
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

    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    baz = dag.source(**baz_tkit.into_dag())

    d_foo = foo.query().deduper(
        name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )
    foo_bar = foo.query().linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    foo_baz = foo.query().linker(
        baz.query(),
        name="foo_baz",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    _ = foo_bar.resolver(
        d_foo,
        foo_baz,
        name="root",
        resolver_class=Components,
        resolver_settings={
            "thresholds": {d_foo.name: 0, foo_bar.name: 0, foo_baz.name: 0}
        },
    )

    run_order: list[str] = []

    def _record_run(
        node: Source | Model | Resolver, *args: object, **kwargs: object
    ) -> pl.DataFrame:
        run_order.append(node.name)
        node._local_data = pl.DataFrame()
        return node._local_data

    source_run_mock.side_effect = _record_run
    model_run_mock.side_effect = _record_run
    resolver_run_mock.side_effect = _record_run

    dag.run_and_sync()

    assert len(run_order) == len(dag.nodes)
    assert set(run_order) == set(dag.nodes.keys())

    run_index = {name: index for index, name in enumerate(run_order)}
    for child, parents in dag.graph.items():
        for parent in parents:
            assert run_index[parent] < run_index[child]


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

    d_foo = model_factory(name="d_foo", dag=dag).model

    with pytest.raises(ValueError, match="not added to DAG"):
        dag.resolver(
            inputs=[d_foo],
            name="r_foo",
            resolver_class=Components,
            resolver_settings={},
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

    # Create a valid resolver, then verify clashes
    root = linker.resolver(d_foo, name="root", resolver_class=Components)

    # Cannot overwrite resolver with source
    updated_root_args = foo_tkit.into_dag()
    updated_root_args["name"] = "root"
    with pytest.raises(ValueError, match="Cannot re-assign"):
        dag.source(**updated_root_args)

    # Cannot overwrite resolver with model
    with pytest.raises(ValueError, match="Cannot re-assign"):
        foo.query().deduper(
            name="root",
            model_class=NaiveDeduper,
            model_settings={"unique_fields": []},
        )

    # Cannot change inputs of resolver
    with pytest.raises(ValueError, match="Cannot re-assign"):
        linker.resolver(name="root", resolver_class=Components)

    # After failed attempts, DAG is as we expect
    assert dag.get_source(foo.name) == foo
    assert dag.get_source(bar.name) == bar
    assert dag.get_source(baz.name) == baz
    assert dag.get_model(d_foo.name) == d_foo
    assert dag.get_model(linker.name) == linker
    assert dag.get_resolver(root.name) == root

    assert dag.graph[foo.name] == []
    assert dag.graph[bar.name] == []
    assert dag.graph[baz.name] == []
    assert dag.graph[d_foo.name] == [foo.name]
    assert dag.graph[linker.name] == [foo.name, bar.name]
    assert set(dag.graph[root.name]) == {linker.name, d_foo.name}


def test_dag_default_resolvers(sqla_sqlite_warehouse: Engine) -> None:
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


def test_dag_draw_tree(sqla_sqlite_warehouse: Engine) -> None:
    """Test tree mode rendering of the DAG."""
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

    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    baz = dag.source(**baz_tkit.into_dag())
    d_foo = foo.query().deduper(
        name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )
    foo_bar = foo.query().linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    foo_baz = foo.query().linker(
        baz.query(),
        name="foo_baz",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    root = foo_bar.resolver(d_foo, foo_baz, name="root", resolver_class=Components)

    node_names = [
        foo.name,
        bar.name,
        baz.name,
        d_foo.name,
        foo_bar.name,
        foo_baz.name,
        root.name,
    ]

    tree_str = dag.draw()
    lines = tree_str.strip().split("\n")
    head_lines, tree_lines = lines[:3], lines[3:]

    # Header
    assert "Collection" in head_lines[0]
    assert "Run" in head_lines[1]

    # Root node is first
    assert root.name in tree_lines[0]

    # All nodes present
    for node in node_names:
        assert any(node in line for line in tree_lines)

    # Type indicators
    assert "💎" in tree_str
    assert "⚙️" in tree_str
    assert "📄" in tree_str

    # Step number badges
    assert "[1]" in tree_str

    # Shared parents shown in each branch, not globally suppressed
    shared_foo_count = sum(1 for line in tree_lines if f" {foo.name} [" in line)
    assert shared_foo_count == 3

    # Tree formatting characters
    assert any(char in tree_str for char in ["└──", "├──", "│"])

    # __str__
    assert str(dag) == dag.draw()

    # Empty DAG
    assert TestkitDAG().dag.draw() == "Empty DAG"

    # Disconnected components
    disconnected_dag = TestkitDAG().dag
    _ = disconnected_dag.source(
        **source_factory(name="qux", engine=sqla_sqlite_warehouse)
        .write_to_location()
        .into_dag()
    )
    _ = disconnected_dag.source(
        **source_factory(name="quux", engine=sqla_sqlite_warehouse)
        .write_to_location()
        .into_dag()
    )
    disconnected_str = disconnected_dag.draw()
    head_lines, tree_lines = disconnected_str[:3], disconnected_str[3:]
    assert "qux" in tree_lines
    assert "quux" in tree_lines
    assert "\n\n" in tree_lines


def test_dag_draw_list(sqla_sqlite_warehouse: Engine) -> None:
    """Test list mode rendering of the DAG."""
    foo_tkit = source_factory(
        name="foo", engine=sqla_sqlite_warehouse
    ).write_to_location()
    bar_tkit = source_factory(
        name="bar", engine=sqla_sqlite_warehouse
    ).write_to_location()

    dag = TestkitDAG().dag

    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    foo_bar = foo.query().linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    root = foo_bar.resolver(name="root", resolver_class=Components)

    list_str = dag.draw(mode="list")
    lines = list_str.strip().split("\n")
    head_lines, list_lines = lines[:3], lines[3:]

    # Same header as tree mode
    assert "Collection" in head_lines[0]
    assert "Run" in head_lines[1]

    # All nodes present
    for node in [foo.name, bar.name, foo_bar.name, root.name]:
        assert any(node in line for line in list_lines)

    # Step numbers prefix each line
    for i, line in enumerate(list_lines, start=1):
        assert line.startswith(f"{i}.")

    # Sources appear before linker, linker before resolver
    foo_bar_idx = next(
        i for i, line in enumerate(list_lines) if line.endswith(foo_bar.name)
    )
    foo_idx = next(i for i, line in enumerate(list_lines) if line.endswith(foo.name))
    bar_idx = next(i for i, line in enumerate(list_lines) if line.endswith(bar.name))
    root_idx = next(i for i, line in enumerate(list_lines) if line.endswith(root.name))
    assert foo_bar_idx > foo_idx
    assert foo_bar_idx > bar_idx
    assert root_idx > foo_bar_idx

    # No tree formatting characters in content lines
    assert not any(char in line for line in list_lines for char in ["└──", "├──", "│"])


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("tree", id="tree"),
        pytest.param("list", id="list"),
    ],
)
def test_dag_draw_status(sqla_sqlite_warehouse: Engine, mode: str) -> None:
    """Test that status indicators render correctly in both draw modes."""
    foo_tkit = source_factory(
        name="foo", engine=sqla_sqlite_warehouse
    ).write_to_location()
    bar_tkit = source_factory(
        name="bar", engine=sqla_sqlite_warehouse
    ).write_to_location()

    dag = TestkitDAG().dag

    foo = dag.source(**foo_tkit.into_dag())
    bar = dag.source(**bar_tkit.into_dag())
    d_foo = foo.query().deduper(
        name="d_foo", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )
    foo_bar = foo.query().linker(
        bar.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    root = foo_bar.resolver(d_foo, name="root", resolver_class=Components)

    draw_str = dag.draw(
        mode=mode,
        status={
            foo_bar.name: DAGNodeExecutionStatus.DOING,
            d_foo.name: DAGNodeExecutionStatus.DONE,
            foo.name: DAGNodeExecutionStatus.SKIPPED,
            root.name: DAGNodeExecutionStatus.DONE,
        },
    )
    content_lines = draw_str.strip().split("\n")[3:]

    # All four indicators present (bar is awaiting)
    assert "✅" in draw_str
    assert "🔄" in draw_str
    assert "⏸️" in draw_str
    assert "⏭️" in draw_str

    # Indicators are directly adjacent to type indicators
    assert "✅⚙️" in draw_str
    assert "🔄⚙️" in draw_str
    assert "⏸️📄" in draw_str
    assert "⏭️📄" in draw_str
    assert "✅💎" in draw_str

    # Correct indicator per node
    for line in content_lines:
        if foo_bar.name in line:
            assert "🔄" in line
        elif d_foo.name in line:
            assert "✅" in line
        elif f" {foo.name}" in line:
            assert "⏭️" in line
        elif bar.name in line:
            assert "⏸️" in line
        elif root.name in line:
            assert "✅" in line


# Lookups


def test_get_matches(matchbox_api: MockRouter) -> None:
    """Match lookup can be generated from DAG."""
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
            json=Run(run_id=1, steps={}).model_dump(),
        )
    )

    # Build dummy DAG
    foo_source = dag.source(**foo.into_dag())
    bar_source = dag.source(**bar.into_dag())
    baz_source = dag.source(**baz.into_dag())
    foo_dedupe = foo_source.query().deduper(
        name="foo_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    foo_bar = foo_source.query().linker(
        bar_source.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    bar_baz = bar_source.query().linker(
        baz_source.query(),
        name="bar_baz",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    foo_bar_baz = foo_bar.resolver(
        foo_dedupe, bar_baz, name="foo_bar_baz", resolver_class=Components
    )
    dag.new_run()

    # Start with a single resolver apex and exercise implicit apex selection.
    matchbox_api.get(
        "/query",
        params={
            "source": "foo",
            "run_id": 1,
            "collection": dag.name,
            "resolver": foo_bar_baz.name,
            "return_leaf_id": "True",
        },
    ).mock(return_value=Response(200, content=table_to_buffer(foo_data).read()))

    matchbox_api.get(
        "/query",
        params={
            "source": "bar",
            "run_id": 1,
            "collection": dag.name,
            "resolver": foo_bar_baz.name,
            "return_leaf_id": "True",
        },
    ).mock(return_value=Response(200, content=table_to_buffer(bar_data).read()))

    matchbox_api.get(
        "/query",
        params={
            "source": "baz",
            "run_id": 1,
            "collection": dag.name,
            "resolver": foo_bar_baz.name,
            "return_leaf_id": "True",
        },
    ).mock(return_value=Response(200, content=table_to_buffer(baz_data).read()))
    apex_resolved = dag.get_matches()
    assert len(apex_resolved.sources) == 3
    assert {source.name for source in apex_resolved.sources} == {"foo", "bar", "baz"}

    # Add a second resolver after the apex assertion.
    foo_bar_resolver = foo_bar.resolver(
        foo_dedupe, name="foo_bar_resolver", resolver_class=Components
    )

    # Intermediate resolver has a narrower source set (foo + bar only).
    matchbox_api.get(
        "/query",
        params={
            "source": "foo",
            "run_id": 1,
            "collection": dag.name,
            "resolver": foo_bar_resolver.name,
            "return_leaf_id": "True",
        },
    ).mock(return_value=Response(200, content=table_to_buffer(foo_data).read()))
    matchbox_api.get(
        "/query",
        params={
            "source": "bar",
            "run_id": 1,
            "collection": dag.name,
            "resolver": foo_bar_resolver.name,
            "return_leaf_id": "True",
        },
    ).mock(return_value=Response(200, content=table_to_buffer(bar_data).read()))

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
                steps={
                    foo.name: foo.fake_run().source.to_dto(),
                    bar.name: bar.fake_run().source.to_dto(),
                    baz.name: baz.fake_run().source.to_dto(),
                    foo_dedupe.name: Step(
                        fingerprint=b"mock_dedupe",
                        step_type=StepType.MODEL,
                        config=foo_dedupe.config,
                    ),
                    foo_bar.name: Step(
                        fingerprint=b"mock_model_1",
                        step_type=StepType.MODEL,
                        config=foo_bar.config,
                    ),
                    bar_baz.name: Step(
                        fingerprint=b"mock_model_2",
                        step_type=StepType.MODEL,
                        config=bar_baz.config,
                    ),
                    foo_bar_resolver.name: Step(
                        fingerprint=b"mock_resolver_1",
                        step_type=StepType.RESOLVER,
                        config=foo_bar_resolver.config,
                    ),
                    foo_bar_baz.name: Step(
                        fingerprint=b"mock_resolver_2",
                        step_type=StepType.RESOLVER,
                        config=foo_bar_baz.config,
                    ),
                },
            ).model_dump(),
        )
    )

    # No sources are found
    with pytest.raises(MatchboxStepNotFoundError):
        dag.get_matches(resolver=foo_bar_baz.name, source_filter=["nonexistent"])

    with pytest.raises(MatchboxStepNotFoundError):
        dag.get_matches(resolver=foo_bar_baz.name, location_names=["nonexistent"])

    # With URI filter
    uri_filter_resolved = dag.get_matches(
        resolver=foo_bar_baz.name, location_names=["sqlite"]
    )

    assert len(uri_filter_resolved.sources) == 1
    assert uri_filter_resolved.sources[0].name == "foo"

    # With source filter
    source_filter_resolved = dag.get_matches(
        resolver=foo_bar_baz.name, source_filter=["foo"]
    )

    assert len(source_filter_resolved.sources) == 1
    assert source_filter_resolved.sources[0].name == "foo"

    # Select intermediate resolver
    intermediate_res = dag.get_matches(resolver=foo_bar_resolver.name)
    assert len(intermediate_res.sources) == 2
    assert set([s.name for s in intermediate_res.sources]) == {foo.name, bar.name}

    # With no filter
    full_resolved = dag.get_matches(resolver=foo_bar_baz.name)
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
    loaded_res = DAG("companies").load_default().get_matches(resolver=foo_bar_baz.name)
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

    linker_foo_bar = foo.query().linker(
        bar.query(),
        name="linker1",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    linker_bar_baz = bar.query().linker(
        baz.query(),
        name="linker2",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    linker_foo_bar.resolver(linker_bar_baz, name="root", resolver_class=Components)

    foo_path = StepPath(name="foo", collection=dag.name, run=dag.run)
    bar_path = StepPath(name="bar", collection=dag.name, run=dag.run)
    baz_path = StepPath(name="baz", collection=dag.name, run=dag.run)

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

    get_route = matchbox_api.get("/match").mock(
        return_value=Response(200, content=serialised_matches)
    )

    # Use lookup function
    matches = dag.lookup_key(from_source="foo", to_sources=["bar", "baz"], key="pk1")

    # Verify results
    assert matches == {foo.name: ["a"], bar.name: ["b"], baz.name: ["b"]}
    request = get_route.calls.last.request
    assert request.url.params["collection"] == dag.name
    assert request.url.params["run_id"] == str(dag.run)
    assert request.url.params.get_list("targets") == ["bar", "baz"]
    assert request.url.params["source"] == "foo"
    assert request.url.params["key"] == "pk1"
    assert request.url.params["resolver"] == "root"


def test_resolver_rejects_resolver_inputs(sqla_sqlite_warehouse: Engine) -> None:
    dag = TestkitDAG().dag
    source_testkit = source_factory(
        engine=sqla_sqlite_warehouse,
        name="source",
    ).write_to_location()
    source = dag.source(**source_testkit.into_dag())

    dedupe = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    first_resolver = dedupe.resolver(name="resolver_1", resolver_class=Components)

    with pytest.raises(MatchboxStepTypeError, match="Expected one of: model"):
        dedupe.resolver(first_resolver, name="resolver_2", resolver_class=Components)


def test_lookup_key_404_source(matchbox_api: MockRouter) -> None:
    """Key lookup throws a step not found error."""
    # Set up dummy data
    source_testkit = source_factory(name="source")
    target_testkit = source_factory(name="target")

    dag = TestkitDAG().dag

    source = dag.source(**source_testkit.into_dag())
    target = dag.source(**target_testkit.into_dag())
    linker = source.query().linker(
        target.query(),
        name="root",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    source_dedupe = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    linker.resolver(source_dedupe, name="root_resolver", resolver_class=Components)

    matchbox_api.get("/match").mock(
        return_value=Response(
            404,
            json=ErrorResponse(
                exception_type="MatchboxStepNotFoundError",
                message="Step 42 not found",
            ).model_dump(),
        )
    )

    # Use match function
    with pytest.raises(MatchboxStepNotFoundError, match="42"):
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

    source = dag.source(**source_testkit.into_dag())
    target = dag.source(**target_testkit.into_dag())
    linker = source.query().linker(
        target.query(),
        name="root",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    source_dedupe = source.query().deduper(
        name="source_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    linker.resolver(source_dedupe, name="root_resolver", resolver_class=Components)

    # Mock empty match results
    matchbox_api.get("/match").mock(return_value=Response(200, content="[]"))

    # Test that empty match results raise MatchboxEmptyServerResponse
    with pytest.raises(
        MatchboxEmptyServerResponse, match="The match operation returned no data"
    ):
        dag.lookup_key(from_source="source", to_sources=["target"], key="pk1")


def test_from_dto() -> None:
    """Test reconstructing Sources, Models and Resolvers from a Step."""
    # Setup
    dag_testkit = TestkitDAG()
    linked_testkit = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked_testkit)
    true_entities = tuple(linked_testkit.true_entities)

    crn_testkit = linked_testkit.sources["crn"].fake_run()
    dh_testkit = linked_testkit.sources["dh"].fake_run()
    cdms_testkit = linked_testkit.sources["cdms"].fake_run()

    crn_dh_model = model_factory(
        name="link_crn_dh",
        left_testkit=crn_testkit,
        right_testkit=dh_testkit,
        true_entities=true_entities,
    ).fake_run()
    crn_cdms_model = model_factory(
        name="link_crn_cdms",
        left_testkit=crn_testkit,
        right_testkit=cdms_testkit,
        true_entities=true_entities,
    ).fake_run()
    dag_testkit.add_model(crn_dh_model)
    dag_testkit.add_model(crn_cdms_model)

    resolver_testkit = resolver_factory(
        dag=dag_testkit.dag,
        inputs=[crn_dh_model, crn_cdms_model],
        true_entities=true_entities,
    ).fake_run()
    dag_testkit.add_resolver(resolver_testkit)

    # Test 1: Add all steps in dependency order
    t1_dag = TestkitDAG().dag
    for testkit in [crn_testkit, dh_testkit, cdms_testkit]:
        t1_dag.add_step(name=testkit.name, step=testkit.source.to_dto())
    for testkit in [crn_dh_model, crn_cdms_model]:
        t1_dag.add_step(name=testkit.name, step=testkit.model.to_dto())
    t1_dag.add_step(
        name=resolver_testkit.name,
        step=resolver_testkit.resolver.to_dto(),
    )

    assert t1_dag.name == dag_testkit.dag.name
    assert t1_dag.run == dag_testkit.dag.run
    for name, step in t1_dag.nodes.items():
        assert step.config == dag_testkit.dag.nodes[name].config
    assert t1_dag.graph == dag_testkit.dag.graph

    # Test 2: Resolver added before its model dependencies raises
    t2_dag = TestkitDAG().dag
    with pytest.raises(ValueError, match="not found in DAG"):
        t2_dag.add_step(
            name=resolver_testkit.name,
            step=resolver_testkit.resolver.to_dto(),
        )

    # Test 3: Model added before its source dependencies raises
    t3_dag = TestkitDAG().dag
    with pytest.raises(ValueError, match="not found in DAG"):
        t3_dag.add_step(
            name=crn_dh_model.name,
            step=crn_dh_model.model.to_dto(),
        )


def test_dag_creates_new_collection(matchbox_api: MockRouter) -> None:
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
    create_route = matchbox_api.post("/collections/test_collection").mock(
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
            json=Run(run_id=1, steps={}).model_dump(),
        )
    )

    # Connect the DAG
    result = dag.new_run()

    # Verify
    assert result == dag
    assert dag.run == 1

    assert create_route.called
    assert json.loads(create_route.calls.last.request.content) == [
        PermissionGrant(
            group_name=DefaultGroup.PUBLIC, permission=PermissionType.ADMIN
        ).model_dump()
    ]


def test_dag_creates_new_collection_with_custom_admin(
    matchbox_api: MockRouter,
) -> None:
    """DAG can be configured with a custom admin group."""
    dag = DAG(name="test_collection", admin_group="custom_admins")

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
    create_route = matchbox_api.post("/collections/test_collection").mock(
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
            json=Run(run_id=1, steps={}).model_dump(),
        )
    )

    dag.new_run()

    assert create_route.called
    assert json.loads(create_route.calls.last.request.content) == [
        PermissionGrant(
            group_name="custom_admins", permission=PermissionType.ADMIN
        ).model_dump()
    ]


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
            json=Run(run_id=expected_run_id, steps={}).model_dump(),
        )
    )

    # Connect the DAG
    result = dag.new_run()

    # Verify
    assert result == dag
    assert dag.run == expected_run_id


def test_dag_load_server_run(matchbox_api: MockRouter) -> None:
    """Can retrieve serialised DAG from the server."""
    # Setup
    dag_testkit = TestkitDAG()
    linked_testkit = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked_testkit)
    true_entities = tuple(linked_testkit.true_entities)

    crn_testkit = linked_testkit.sources["crn"].fake_run()
    dh_testkit = linked_testkit.sources["dh"].fake_run()
    cdms_testkit = linked_testkit.sources["cdms"].fake_run()

    crn_dh_model = model_factory(
        name="link_crn_dh",
        left_testkit=crn_testkit,
        right_testkit=dh_testkit,
        true_entities=true_entities,
    ).fake_run()
    crn_cdms_model = model_factory(
        name="link_crn_cdms",
        left_testkit=crn_testkit,
        right_testkit=cdms_testkit,
        true_entities=true_entities,
    ).fake_run()
    dag_testkit.add_model(crn_dh_model)
    dag_testkit.add_model(crn_cdms_model)

    resolver_testkit = resolver_factory(
        dag=dag_testkit.dag,
        inputs=[crn_dh_model, crn_cdms_model],
        true_entities=true_entities,
    ).fake_run()
    dag_testkit.add_resolver(resolver_testkit)

    # Full pipeline: 3 sources, 2 linkers, 1 resolver
    steps: dict[StepName, Step] = {
        crn_testkit.name: crn_testkit.source.to_dto(),
        dh_testkit.name: dh_testkit.source.to_dto(),
        cdms_testkit.name: cdms_testkit.source.to_dto(),
        crn_dh_model.name: crn_dh_model.model.to_dto(),
        crn_cdms_model.name: crn_cdms_model.model.to_dto(),
        resolver_testkit.name: resolver_testkit.resolver.to_dto(),
    }

    run = Run(run_id=1, steps=steps)

    matchbox_api.get(f"/collections/{dag_testkit.dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(
                name=dag_testkit.dag.name,
                runs=[1, 2],
                default_run=1,
            ).model_dump(),
        )
    )
    matchbox_api.get(f"/collections/{dag_testkit.dag.name}/runs/1").mock(
        return_value=Response(200, json=run.model_dump())
    )

    # Load default run and verify reconstruction matches original DAG
    default_dag = DAG(name=dag_testkit.dag.name)
    default_dag = default_dag.load_default()

    assert default_dag.name == dag_testkit.dag.name
    assert default_dag.run == 1
    assert set(default_dag.nodes.keys()) == set(dag_testkit.dag.nodes.keys())
    assert default_dag.graph == dag_testkit.dag.graph

    matchbox_api.get(f"/collections/{dag_testkit.dag.name}/runs/2").mock(
        return_value=Response(200, json=run.model_dump())
    )

    # Load pending run and verify it gets run=2
    pending_dag = DAG(name=dag_testkit.dag.name)
    pending_dag = pending_dag.load_pending()

    assert pending_dag.name == dag_testkit.dag.name
    assert pending_dag.run == 2
    assert set(pending_dag.nodes.keys()) == set(dag_testkit.dag.nodes.keys())
    assert pending_dag.graph == dag_testkit.dag.graph

    # Compatible local nodes are overwritten by server state
    overwritten_dag = DAG(name=dag_testkit.dag.name)
    overwritten_source = overwritten_dag.source(**crn_testkit.into_dag())
    overwritten_source.description = "new description"
    overwritten_dag.load_pending()
    assert (
        overwritten_dag.get_source(crn_testkit.name).description
        == crn_testkit.source.description
    )

    # A local node with the same name but different parents cannot be reconciled
    # This mirrors a script that defines a same-name node with changed dependencies.
    clashing_dag = DAG(name=dag_testkit.dag.name)
    crn = clashing_dag.source(**crn_testkit.into_dag())
    clashing_dag.source(**dh_testkit.into_dag())
    crn.query().deduper(
        name=crn_dh_model.name,
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    with pytest.raises(ValueError, match="Cannot re-assign"):
        clashing_dag.load_pending()

    # Missing collection raises
    matchbox_api.get(f"/collections/{dag_testkit.dag.name}/runs/1").mock(
        return_value=Response(
            404,
            json=ErrorResponse(
                exception_type="MatchboxCollectionNotFoundError",
                message="Collection not found",
            ).model_dump(),
        ),
    )
    with pytest.raises(MatchboxCollectionNotFoundError):
        DAG(name=dag_testkit.dag.name).load_default()


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

    # model_inner depends on two sources.
    model_inner_testkit = model_factory(
        name="model_inner",
        left_testkit=crn_testkit,
        right_testkit=dh_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    ).fake_run()

    model_side_testkit = model_factory(
        name="model_side",
        left_testkit=dh_testkit,
        right_testkit=cdms_testkit,
        true_entities=linked_testkit.true_entities,
        dag=test_dag,
    ).fake_run()

    # Add sources and models to the reference DAG.
    test_dag.source(**crn_testkit.into_dag())
    test_dag.source(**dh_testkit.into_dag())
    test_dag.source(**cdms_testkit.into_dag())
    test_dag.model(**model_inner_testkit.into_dag())
    test_dag.model(**model_side_testkit.into_dag())

    resolver_inner_testkit = resolver_factory(
        dag=test_dag,
        name="resolver_inner",
        inputs=[model_inner_testkit, model_side_testkit],
        true_entities=tuple(linked_testkit.true_entities),
    ).fake_run()
    test_dag.add_step(
        name=resolver_inner_testkit.name,
        step=resolver_inner_testkit.resolver.to_dto(),
    )
    resolver_inner = test_dag.get_resolver(resolver_inner_testkit.name)

    # model_outer also has dependency count=2, but one dependency is resolver_inner.
    # This must still load after resolver_inner, even if dict order puts it first.
    model_outer = test_dag.model(
        name="model_outer",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.key=r.key"},
        left_query=resolver_inner.query(),
        right_query=test_dag.get_source(cdms_testkit.name).query(),
    )

    # Intentionally place model_outer first: dependency count ties with model_inner,
    # but model_outer depends on resolver_inner and must therefore be delayed.
    steps: dict[StepName, Step] = {
        model_outer.name: Step(
            fingerprint=b"model_outer",
            step_type=StepType.MODEL,
            config=model_outer.config,
        ),
        crn_testkit.name: crn_testkit.source.to_dto(),
        dh_testkit.name: dh_testkit.source.to_dto(),
        cdms_testkit.name: cdms_testkit.source.to_dto(),
        model_inner_testkit.name: model_inner_testkit.model.to_dto(),
        model_side_testkit.name: model_side_testkit.model.to_dto(),
        resolver_inner_testkit.name: resolver_inner_testkit.resolver.to_dto(),
    }

    run = Run(run_id=1, steps=steps)

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

    crn = dag.source(**source_factory().into_dag())
    dd = crn.query().deduper(
        name="dd", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )
    dd.resolver(name="r", resolver_class=Components)

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
    crn = dag.source(**source_factory().into_dag())
    dd = crn.query().deduper(
        name="dd", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )
    dd.resolver(name="r", resolver_class=Components)

    with pytest.raises(RuntimeError, match="has not been connected"):
        dag.set_default()


def test_dag_set_default_unreachable_nodes(sqla_sqlite_warehouse: Engine) -> None:
    """Nodes cannot be unreachable from root when setting a default run."""
    dag = TestkitDAG().dag

    foo_tkit = source_factory(name="foo", engine=sqla_sqlite_warehouse)
    bar_tkit = source_factory(name="bar", engine=sqla_sqlite_warehouse)

    foo = dag.source(**foo_tkit.into_dag())
    _ = dag.source(**bar_tkit.into_dag())

    dd = foo.query().deduper(
        name="dd", model_class=NaiveDeduper, model_settings={"unique_fields": []}
    )
    dd.resolver(name="r", resolver_class=Components)

    with pytest.raises(ValueError, match="unreachable"):
        dag.set_default()


def test_run_and_sync_records_stats(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """run_and_sync resets stats and records dag_run_seconds.

    When run/sync are mocked the decorated method bodies are bypassed, so
    individual operation timings may be empty.  We assert the wall-clock
    field and that every node gets an entry (empty or not).
    """
    foo_tkit = source_factory(
        name="foo", engine=sqla_sqlite_warehouse
    ).write_to_location()
    bar_tkit = source_factory(
        name="bar", engine=sqla_sqlite_warehouse
    ).write_to_location()

    dag = TestkitDAG().dag
    foo = dag.source(**foo_tkit.into_dag())
    dag.source(**bar_tkit.into_dag())
    foo_dedupe = foo.query().deduper(
        name="foo_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    foo_dedupe.resolver(name="root", resolver_class=Components)

    with (
        patch.object(Source, "run"),
        patch.object(Source, "sync"),
        patch.object(Model, "run"),
        patch.object(Model, "sync"),
        patch.object(Resolver, "run"),
        patch.object(Resolver, "sync"),
    ):
        dag.run_and_sync()

    # Wall clock is always recorded
    assert dag.stats.dag_run_seconds is not None
    assert dag.stats.dag_run_seconds > 0


def test_get_matches_records_stats(
    matchbox_api: MockRouter,
) -> None:
    """get_matches records 'query' timing per source."""
    foo = source_factory(name="foo", location_name="sqlite")
    bar = source_factory(name="bar", location_name="postgres")

    foo_data = pa.Table.from_pylist(
        [{"id": 1, "leaf_id": 1, "key": "1"}],
        schema=SCHEMA_QUERY_WITH_LEAVES,
    )
    bar_data = pa.Table.from_pylist(
        [{"id": 2, "leaf_id": 2, "key": "a"}],
        schema=SCHEMA_QUERY_WITH_LEAVES,
    )

    dag = DAG("companies")

    matchbox_api.get(f"/collections/{dag.name}").mock(
        return_value=Response(
            200,
            json=Collection(name=dag.name, runs=[], default_run=None).model_dump(),
        )
    )
    matchbox_api.post(f"/collections/{dag.name}/runs").mock(
        return_value=Response(200, json=Run(run_id=1, steps={}).model_dump()),
    )

    foo_source = dag.source(**foo.into_dag())
    bar_source = dag.source(**bar.into_dag())
    foo_dedupe = foo_source.query().deduper(
        name="foo_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    )
    foo_bar = foo_source.query().linker(
        bar_source.query(),
        name="foo_bar",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.field=r.field"},
    )
    foo_bar_resolver = foo_bar.resolver(
        foo_dedupe, name="foo_bar_resolver", resolver_class=Components
    )
    dag.new_run()

    matchbox_api.get(
        "/query",
        params={
            "source": "foo",
            "run_id": 1,
            "collection": dag.name,
            "resolver": foo_bar_resolver.name,
            "return_leaf_id": "True",
        },
    ).mock(return_value=Response(200, content=table_to_buffer(foo_data).read()))

    matchbox_api.get(
        "/query",
        params={
            "source": "bar",
            "run_id": 1,
            "collection": dag.name,
            "resolver": foo_bar_resolver.name,
            "return_leaf_id": "True",
        },
    ).mock(return_value=Response(200, content=table_to_buffer(bar_data).read()))

    dag.get_matches()

    assert "foo" in dag.stats.timings
    assert "query" in dag.stats.timings["foo"]
    assert "bar" in dag.stats.timings
    assert "query" in dag.stats.timings["bar"]


def test_stats_reset_between_calls(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Stats are reset between run_and_sync calls."""
    foo_tkit = source_factory(
        name="foo", engine=sqla_sqlite_warehouse
    ).write_to_location()

    dag = TestkitDAG().dag
    foo = dag.source(**foo_tkit.into_dag())
    foo.query().deduper(
        name="foo_dedupe",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": []},
    ).resolver(name="root", resolver_class=Components)

    with (
        patch.object(Source, "run"),
        patch.object(Source, "sync"),
        patch.object(Model, "run"),
        patch.object(Model, "sync"),
        patch.object(Resolver, "run"),
        patch.object(Resolver, "sync"),
    ):
        dag.run_and_sync()
    assert dag.stats.dag_run_seconds is not None
    first_run_seconds = dag.stats.dag_run_seconds

    with (
        patch.object(Source, "run"),
        patch.object(Source, "sync"),
        patch.object(Model, "run"),
        patch.object(Model, "sync"),
        patch.object(Resolver, "run"),
        patch.object(Resolver, "sync"),
    ):
        dag.run_and_sync()

    # dag_run_seconds should reflect only the second run (reset worked)
    assert dag.stats.dag_run_seconds is not None
    assert dag.stats.dag_run_seconds != first_run_seconds
