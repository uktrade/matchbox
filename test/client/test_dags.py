from unittest.mock import Mock, call, patch

import pytest
from sqlalchemy import Engine

from matchbox.client.dags import DAG, DedupeStep, LinkStep, StepInput
from matchbox.client.helpers.selector import Selector
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import source_factory


def test_step_input_validation(sqlite_warehouse: Engine):
    """Cannot select sources not available to a step."""
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source

    d_foo_right = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    # CASE 1: Reading from Source
    with pytest.raises(ValueError, match="only select"):
        StepInput(prev_node=foo, select={bar: []})

    # CASE 2: Reading from previous step
    with pytest.raises(ValueError, match="Cannot select"):
        StepInput(prev_node=d_foo_right, select={bar: []})


def test_model_step_validation(sqlite_warehouse: Engine):
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source
    baz = source_factory(full_name="baz", engine=sqlite_warehouse).source

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    foo_bar = LinkStep(
        name="foo_bar",
        description="",
        left=StepInput(prev_node=d_foo, select={foo: []}, threshold=0.5),
        right=StepInput(prev_node=bar, select={bar: []}, threshold=0.7),
        model_class=DeterministicLinker,
        settings={"left_id": "id", "right_id": "id", "comparisons": ""},
        truth=1,
    )

    foo_bar_baz = LinkStep(
        name="foo_bar_baz",
        description="",
        left=StepInput(prev_node=foo_bar, select={foo: [], bar: []}),
        right=StepInput(prev_node=baz, select={baz: []}),
        model_class=DeterministicLinker,
        settings={"left_id": "id", "right_id": "id", "comparisons": ""},
        truth=1,
    )

    # Inherit from a source directly
    assert d_foo.sources == {str(foo.address)}

    # Inherit one source from a previous step
    assert foo_bar.sources == {str(foo.address), str(bar.address)}

    # Inherit multiple sources from a previous step
    assert foo_bar_baz.sources == {str(foo.address), str(bar.address), str(baz.address)}


@pytest.mark.parametrize(
    "batched",
    [
        pytest.param(False, id="not batched"),
        pytest.param(True, id="batched"),
    ],
)
def test_dedupe_step_run(
    batched: bool,
    sqlite_warehouse: Engine,
):
    """Tests that a dedupe step orchestrates lower-level API correctly."""
    with (
        patch("matchbox.client.dags.make_model") as make_model_mock,
        patch("matchbox.client.dags.process") as process_mock,
        patch("matchbox.client.dags.query") as query_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run deduper
        foo = source_factory(
            full_name="foo", engine=sqlite_warehouse
        ).source.set_engine(sqlite_warehouse)
        d_foo = DedupeStep(
            name="d_foo",
            description="",
            left=StepInput(
                prev_node=foo,
                select={foo: []},
                threshold=0.5,
                batch_size=100 if batched else None,
            ),
            model_class=NaiveDeduper,
            settings={"id": "id", "unique_fields": []},
            truth=1,
        )

        d_foo.run()

        # Right data is queried
        query_mock.assert_called_once_with(
            [Selector(engine=sqlite_warehouse, address=foo.address, fields=[])],
            return_type="pandas",
            threshold=d_foo.left.threshold,
            resolution_name=d_foo.left.name,
            only_indexed=True,
            batch_size=100 if batched else None,
            return_batches=False,
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
        assert model_mock.truth == 1


@pytest.mark.parametrize(
    "batched",
    [
        pytest.param(False, id="not batched"),
        pytest.param(True, id="batched"),
    ],
)
def test_link_step_run(
    batched: bool,
    sqlite_warehouse: Engine,
):
    """Tests that a link step orchestrates lower-level API correctly."""
    with (
        patch("matchbox.client.dags.make_model") as make_model_mock,
        patch("matchbox.client.dags.process") as process_mock,
        patch("matchbox.client.dags.query") as query_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run deduper
        foo = source_factory(
            full_name="foo", engine=sqlite_warehouse
        ).source.set_engine(sqlite_warehouse)
        bar = source_factory(
            full_name="bar", engine=sqlite_warehouse
        ).source.set_engine(sqlite_warehouse)
        foo_bar = LinkStep(
            name="foo_bar",
            description="",
            left=StepInput(
                prev_node=foo,
                select={foo: []},
                threshold=0.5,
                batch_size=100 if batched else None,
            ),
            right=StepInput(
                prev_node=bar,
                select={bar: []},
                threshold=0.7,
                batch_size=100 if batched else None,
            ),
            model_class=DeterministicLinker,
            settings={"left_id": "id", "right_id": "id", "comparisons": ""},
            truth=1,
        )

        foo_bar.run()

        # Right data is queried
        assert query_mock.call_count == 2
        assert query_mock.call_args_list[0] == call(
            [Selector(engine=sqlite_warehouse, address=foo.address, fields=[])],
            return_type="pandas",
            threshold=foo_bar.left.threshold,
            resolution_name=foo_bar.left.name,
            only_indexed=True,
            batch_size=100 if batched else None,
            return_batches=False,
        )
        assert query_mock.call_args_list[1] == call(
            [Selector(engine=sqlite_warehouse, address=bar.address, fields=[])],
            return_type="pandas",
            threshold=foo_bar.right.threshold,
            resolution_name=foo_bar.right.name,
            only_indexed=True,
            batch_size=100 if batched else None,
            return_batches=False,
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
        assert model_mock.truth == 1


@patch("matchbox.client.dags._handler.index")
@patch.object(DedupeStep, "run")
@patch.object(LinkStep, "run")
def test_dag_runs(
    link_run: Mock, dedupe_run: Mock, handler_index: Mock, sqlite_warehouse: Engine
):
    """A legal DAG can be built and run."""
    # Set up constituents
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source
    baz = source_factory(full_name="baz", engine=sqlite_warehouse).source

    # Structure: Sources can be deduped
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    # Structure:
    # - Sources can be passed directly to linkers
    # - Or, linkers can take dedupers
    foo_bar = LinkStep(
        left=StepInput(
            prev_node=d_foo,
            select={foo: []},
        ),
        right=StepInput(
            prev_node=bar,
            select={bar: []},
        ),
        name="foo_bar",
        description="",
        model_class=DeterministicLinker,
        settings={},
        truth=1,
    )

    # Structure: Linkers can take other linkers
    foo_bar_baz = LinkStep(
        left=StepInput(
            prev_node=foo_bar,
            select={foo: [], bar: []},
            cleaners={},
        ),
        right=StepInput(
            prev_node=baz,
            select={baz: []},
            cleaners={},
        ),
        name="foo_bar_baz",
        description="",
        model_class=DeterministicLinker,
        settings={},
        truth=1,
    )

    # Assemble DAG with batch sizes
    dag = DAG()

    dag.add_sources(foo, batch_size=100)
    dag.add_sources(bar, baz, batch_size=200)

    # Verify batch sizes are stored correctly
    assert dag._index_batch_sizes[str(foo.address)] == 100
    assert dag._index_batch_sizes[str(bar.address)] == 200
    assert dag._index_batch_sizes[str(baz.address)] == 200

    assert set(dag.nodes.keys()) == {
        str(foo.address),
        str(bar.address),
        str(baz.address),
    }

    dag.add_steps(d_foo, foo_bar, foo_bar_baz)
    assert set(dag.nodes.keys()) == {
        str(foo.address),
        str(bar.address),
        str(baz.address),
        d_foo.name,
        foo_bar.name,
        foo_bar_baz.name,
    }

    # Prepare DAG
    dag.prepare()
    s_foo, s_bar, s_d_foo, s_foo_bar, s_foo_bar_baz = (
        dag.sequence.index(str(foo.address)),
        dag.sequence.index(str(bar.address)),
        dag.sequence.index(d_foo.name),
        dag.sequence.index(foo_bar.name),
        dag.sequence.index(foo_bar_baz.name),
    )
    assert s_foo < s_d_foo < s_foo_bar < s_foo_bar_baz
    assert s_bar < s_foo_bar < s_foo_bar_baz

    # Run DAG
    dag.run()

    assert handler_index.call_count == 3

    # Verify sources and batch sizes passed to handler.index
    calls = {
        call.kwargs["source"]: call.kwargs["batch_size"]
        for call in handler_index.call_args_list
    }

    assert calls[foo] == 100
    assert calls[bar] == 200
    assert calls[baz] == 200

    # Verify the right sources were sent to index
    assert {
        handler_index.call_args_list[0].kwargs["source"],
        handler_index.call_args_list[1].kwargs["source"],
        handler_index.call_args_list[2].kwargs["source"],
    } == {foo, bar, baz}

    assert dedupe_run.call_count == 1
    assert link_run.call_count == 2

    # Reset mocks to test the start argument
    handler_index.reset_mock()
    dedupe_run.reset_mock()
    link_run.reset_mock()

    # Run DAG again, starting from foo_bar step
    dag.run(start="foo_bar")

    # Verify only steps from foo_bar onward were executed
    assert handler_index.call_count == 0
    assert dedupe_run.call_count == 0
    assert link_run.call_count == 2


def test_dag_missing_dependency(sqlite_warehouse: Engine):
    """Steps cannot be added before their dependencies."""
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    dag = DAG()
    with pytest.raises(ValueError, match="Dependency"):
        dag.add_steps(d_foo)


def test_dag_name_clash(sqlite_warehouse: Engine):
    """Names across sources and steps must be unique."""
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )
    d_bar_wrong = DedupeStep(
        # Name clash!
        name="d_foo",
        description="",
        left=StepInput(prev_node=bar, select={bar: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )
    dag = DAG()
    dag.add_sources(foo)
    dag.add_steps(d_foo)
    with pytest.raises(ValueError, match="already taken"):
        dag.add_steps(d_bar_wrong)
    # DAG is not modified by failed attempt
    assert dag.nodes["d_foo"] == d_foo
    # We didn't overwrite d_foo's dependencies
    assert dag.graph["d_foo"] == [str(foo.address)]


def test_dag_disconnected(sqlite_warehouse: Engine):
    """Nodes cannot be disconnected."""
    foo = source_factory(full_name="foo", engine=sqlite_warehouse).source
    bar = source_factory(full_name="bar", engine=sqlite_warehouse).source

    dag = DAG()
    dag.add_sources(foo, bar)

    with pytest.raises(ValueError, match="disconnected"):
        dag.prepare()
