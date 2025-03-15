from unittest.mock import Mock, call, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.dags import Dag, DedupeStep, LinkStep, StepInput
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import source_factory


def test_dedupe_step_run(
    sqlite_warehouse: Engine,
):
    """Tests that a dedupe step orchestrates lower-level API correctly"""
    with (
        patch("matchbox.client.dags.make_model") as make_model_mock,
        patch("matchbox.client.dags.process") as process_mock,
        patch("matchbox.client.dags.query") as query_mock,
        patch("matchbox.client.dags.select") as select_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run deduper
        foo = source_factory(full_name="foo").source
        foo_select = {foo.address.full_name: []}
        d_foo = DedupeStep(
            name="d_foo",
            description="",
            left=StepInput(origin=foo, select=foo_select, threshold=0.5),
            model_class=NaiveDeduper,
            settings={"id": "id", "unique_fields": []},
        )

        d_foo.run(engine=sqlite_warehouse)

        # Right data is queried
        select_mock.assert_called_once_with(foo_select, engine=sqlite_warehouse)
        query_mock.assert_called_once_with(
            select_mock.return_value,
            return_type="pandas",
            threshold=d_foo.left.threshold,
            resolution_name=d_foo.left.name,
        )
        # Data is pre-processed
        process_mock.assert_called_once_with(
            query_mock.return_value, d_foo.left.cleaners
        )

        # Model is created and run
        make_model_mock.assert_called_once_with(
            model_name=d_foo.name,
            description=d_foo.description,
            model_class=d_foo.model_class,
            model_settings=d_foo.settings,
            left_data=process_mock.return_value,
            left_resolution=d_foo.left.name,
        )
        model_mock.run.assert_called_once()

        # Results are stored
        model_mock.run().to_matchbox.assert_called_once()


def test_link_step_run(
    sqlite_warehouse: Engine,
):
    """Tests that a link step orchestrates lower-level API correctly"""
    with (
        patch("matchbox.client.dags.make_model") as make_model_mock,
        patch("matchbox.client.dags.process") as process_mock,
        patch("matchbox.client.dags.query") as query_mock,
        patch("matchbox.client.dags.select") as select_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run deduper
        foo = source_factory(full_name="foo").source
        bar = source_factory(full_name="bar").source
        foo_select = {foo.address.full_name: []}
        bar_select = {bar.address.full_name: []}
        foo_bar = LinkStep(
            name="foo_bar",
            description="",
            left=StepInput(origin=foo, select=foo_select, threshold=0.5),
            right=StepInput(origin=bar, select=bar_select, threshold=0.7),
            model_class=DeterministicLinker,
            settings={"left_id": "id", "right_id": "id", "comparisons": ""},
        )

        foo_bar.run(engine=sqlite_warehouse)

        # Right data is queried
        assert select_mock.call_count == 2
        assert select_mock.call_args_list[0] == call(
            foo_select,
            engine=sqlite_warehouse,
        )
        assert select_mock.call_args_list[1] == call(
            bar_select,
            engine=sqlite_warehouse,
        )
        assert query_mock.call_count == 2
        assert query_mock.call_args_list[0] == call(
            select_mock.return_value,
            return_type="pandas",
            threshold=foo_bar.left.threshold,
            resolution_name=foo_bar.left.name,
        )
        assert query_mock.call_args_list[1] == call(
            select_mock.return_value,
            return_type="pandas",
            threshold=foo_bar.right.threshold,
            resolution_name=foo_bar.right.name,
        )

        # Data is pre-processed
        assert process_mock.call_count == 2
        assert process_mock.call_args_list[0] == call(
            query_mock.return_value, foo_bar.left.cleaners
        )
        assert process_mock.call_args_list[1] == call(
            query_mock.return_value, foo_bar.right.cleaners
        )

        # Model is created and run
        make_model_mock.assert_called_once_with(
            model_name=foo_bar.name,
            description=foo_bar.description,
            model_class=foo_bar.model_class,
            model_settings=foo_bar.settings,
            left_data=process_mock.return_value,
            left_resolution=foo_bar.left.name,
            right_data=process_mock.return_value,
            right_resolution=foo_bar.right.name,
        )
        model_mock.run.assert_called_once()

        # Results are stored
        model_mock.run().to_matchbox.assert_called_once()


@patch("matchbox.client.dags._handler.index")
@patch.object(DedupeStep, "run")
@patch.object(LinkStep, "run")
def test_dag_runs(
    link_run: Mock, dedupe_run: Mock, handler_index: Mock, sqlite_warehouse: Engine
):
    """A legal DAG can be built and run"""
    # Set up constituents
    foo = source_factory(full_name="foo").source
    bar = source_factory(full_name="bar").source

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(origin=foo, select={foo.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    d_bar = DedupeStep(
        name="d_bar",
        description="",
        left=StepInput(origin=bar, select={bar.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    foo_bar = LinkStep(
        left=StepInput(
            origin=d_foo,
            select={foo.address.full_name: []},
            cleaners={},
        ),
        right=StepInput(
            origin=d_bar,
            select={bar.address.full_name: []},
            cleaners={},
        ),
        name="foo_bar",
        description="",
        model_class=DeterministicLinker,
        settings={},
    )

    # Assemble DAG
    dag = Dag(engine=sqlite_warehouse)

    dag.add_sources(foo, bar)
    assert set(dag.nodes.keys()) == {"foo", "bar"}

    dag.add_steps(d_foo, d_bar, foo_bar)
    assert set(dag.nodes.keys()) == {"foo", "bar", "d_foo", "d_bar", "foo_bar"}
    assert d_foo.sources == {"foo"}
    assert d_bar.sources == {"bar"}
    assert foo_bar.sources == {"foo", "bar"}

    # Prepare DAG
    dag.prepare()
    s_foo, s_bar, s_d_foo, s_d_bar, s_foo_bar = (
        dag.sequence.index("foo"),
        dag.sequence.index("bar"),
        dag.sequence.index("d_foo"),
        dag.sequence.index("d_bar"),
        dag.sequence.index("foo_bar"),
    )
    assert s_foo < s_d_foo < s_foo_bar
    assert s_bar < s_d_bar < s_foo_bar

    # Run DAG
    dag.run()

    assert handler_index.call_count == 2
    assert {
        handler_index.call_args_list[0].kwargs["source"].address.full_name,
        handler_index.call_args_list[1].kwargs["source"].address.full_name,
    } == {"foo", "bar"}

    assert dedupe_run.call_count == 2
    dedupe_run.assert_called_with(engine=sqlite_warehouse)

    link_run.assert_called_once_with(engine=sqlite_warehouse)


def test_dag_missing_dependency(sqlite_warehouse: Engine):
    """Steps cannot be added before their dependencies"""
    foo = source_factory(full_name="foo").source
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(origin=foo, select={foo.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )

    dag = Dag(engine=sqlite_warehouse)
    with pytest.raises(ValueError, match="Dependency"):
        dag.add_steps(d_foo)
    # Sources are not added
    assert not d_foo.sources


def test_dag_name_clash(sqlite_warehouse: Engine):
    """Names across sources and steps must be unique"""
    foo = source_factory(full_name="foo").source
    d_foo = DedupeStep(
        # Name clash!
        name="foo",
        description="",
        left=StepInput(origin=foo, select={foo.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )
    dag = Dag(engine=sqlite_warehouse)
    dag.add_sources(foo)
    with pytest.raises(ValueError, match="already taken"):
        dag.add_steps(d_foo)
    # DAG is not modified by failed attempt
    assert dag.nodes["foo"] == foo
    assert not len(dag.graph["foo"])
    # Sources are not added
    assert not d_foo.sources


def test_dag_source_unavailable(sqlite_warehouse: Engine):
    """Cannot select sources not available to a step"""
    # CASE 1: Reading from Source
    foo = source_factory(full_name="foo").source
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(origin=foo, select={"bar": []}),
        model_class=NaiveDeduper,
        settings={},
    )

    dag = Dag(engine=sqlite_warehouse)
    dag.add_sources(foo)
    with pytest.raises(ValueError, match="only select"):
        dag.add_steps(d_foo)
    # DAG is not modified by failed attempt
    assert d_foo.name not in dag.graph and d_foo.name not in dag.nodes

    # CASE 2: Reading from previous step
    bar = source_factory(full_name="bar").source
    # Re-define d_foo, this time correctly
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(origin=foo, select={foo.address.full_name: []}),
        model_class=NaiveDeduper,
        settings={},
    )
    bar_foo = LinkStep(
        name="foo_bar",
        description="",
        left=StepInput(origin=bar, select={bar.address.full_name: []}),
        # Notice "typo"
        right=StepInput(origin=d_foo, select={"typo": []}),
        model_class=DeterministicLinker,
        settings={},
    )

    dag = Dag(engine=sqlite_warehouse)
    dag.add_sources(foo, bar)
    dag.add_steps(d_foo)
    with pytest.raises(ValueError, match="Cannot select"):
        dag.add_steps(bar_foo)
    # DAG is not modified by failed attempt
    assert bar_foo.name not in dag.graph and bar_foo.name not in dag.nodes
    assert not bar_foo.sources


def test_dag_disconnected(sqlite_warehouse: Engine):
    """Nodes cannot be disconnected"""
    foo = source_factory(full_name="foo").source
    bar = source_factory(full_name="bar").source

    dag = Dag(engine=sqlite_warehouse)
    dag.add_sources(foo, bar)

    with pytest.raises(ValueError, match="disconnected"):
        dag.prepare()
