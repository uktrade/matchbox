import datetime
from unittest.mock import Mock, call, patch

import polars as pl
import pytest
from sqlalchemy import Engine

from matchbox.client.dags import (
    DAG,
    DAGDebugOptions,
    DedupeStep,
    IndexStep,
    LinkStep,
    StepInput,
)
from matchbox.client.helpers.selector import Selector
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.sources import Source
from matchbox.common.factories.sources import source_factory


def test_step_input_validation(sqlite_warehouse: Engine):
    """Cannot select sources not available to a step."""
    foo = source_factory(name="foo", engine=sqlite_warehouse).source
    bar = source_factory(name="bar", engine=sqlite_warehouse).source

    i_foo = IndexStep(source=foo)

    d_foo_right = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=i_foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    # CASE 1: Reading from SourceConfig
    with pytest.raises(ValueError, match="only select"):
        StepInput(prev_node=i_foo, select={bar: []})

    # CASE 2: Reading from previous step
    with pytest.raises(ValueError, match="Cannot select"):
        StepInput(prev_node=d_foo_right, select={bar: []})


def test_step_input_select_fields(sqlite_warehouse: Engine):
    """Test that StepInput correctly handles field selection in select attribute."""
    foo = source_factory(name="foo", engine=sqlite_warehouse).source

    # Create some mock source fields
    field1 = foo.config.index_fields[0]

    i_foo = IndexStep(source=foo)

    # Test selecting specific fields
    step_input = StepInput(prev_node=i_foo, select={foo: [field1.name]})

    # Verify the select attribute contains the expected fields
    assert step_input.select[foo] == [field1.name]
    assert len(step_input.select) == 1

    # Test selecting empty field list (all fields)
    step_input_all = StepInput(prev_node=i_foo, select={foo: []})

    # Verify empty list selection works
    assert step_input_all.select[foo] == []
    assert len(step_input_all.select) == 1


def test_cleaning_dict(sqlite_warehouse: Engine):
    """Test that cleaning works in a StepInput."""
    foo = source_factory(name="foo", engine=sqlite_warehouse).source
    i_foo = IndexStep(source=foo)

    test_data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "foo_name": ["A", "B", "C"],
            "foo_status": ["active", "inactive", "active"],
        }
    )

    step_input = StepInput(
        prev_node=i_foo,
        select={foo: ["name", "status"]},
        cleaning_dict={
            "name": "lower(foo_name)",
        },
    )

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=step_input,
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    result = d_foo.clean(test_data, step_input)
    assert len(result) == 3
    assert result["name"].to_list() == ["a", "b", "c"]
    assert set(result.columns) == {"id", "name", "foo_status"}


@pytest.mark.parametrize(
    "combine_type",
    ["concat", "explode", "set_agg"],
)
def test_step_input_combine_type_in_query(combine_type: str, sqlite_warehouse: Engine):
    """Test that StepInput's combine_type parameter is passed to query function."""
    with patch("matchbox.client.dags.query") as query_mock:
        query_mock.return_value = pl.DataFrame({"id": [1, 2, 3]})

        foo_testkit = source_factory(
            name="foo", engine=sqlite_warehouse
        ).write_to_location()
        foo = foo_testkit.source
        i_foo = IndexStep(source=foo)

        step_input = StepInput(
            prev_node=i_foo, select={foo: []}, combine_type=combine_type
        )

        # Create a mock model step to test the query method
        model_step = DedupeStep(
            name="test_step",
            description="Test step",
            left=step_input,
            model_class=NaiveDeduper,
            settings={},
            truth=1.0,
        )

        # Call the query method
        model_step.query(step_input)

        # Verify query was called with the correct combine_type
        query_mock.assert_called_once()
        call_args = query_mock.call_args
        assert call_args.kwargs["combine_type"] == combine_type


def test_model_step_validation(sqlite_warehouse: Engine):
    foo = source_factory(name="foo", engine=sqlite_warehouse).source
    bar = source_factory(name="bar", engine=sqlite_warehouse).source
    baz = source_factory(name="baz", engine=sqlite_warehouse).source

    i_foo = IndexStep(source=foo)
    i_bar = IndexStep(source=bar)
    i_baz = IndexStep(source=baz)

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=i_foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    foo_bar = LinkStep(
        name="foo_bar",
        description="",
        left=StepInput(prev_node=d_foo, select={foo: []}, threshold=0.5),
        right=StepInput(prev_node=i_bar, select={bar: []}, threshold=0.7),
        model_class=DeterministicLinker,
        settings={"left_id": "id", "right_id": "id", "comparisons": ""},
        truth=1,
    )

    foo_bar_baz = LinkStep(
        name="foo_bar_baz",
        description="",
        left=StepInput(prev_node=foo_bar, select={foo: [], bar: []}),
        right=StepInput(prev_node=i_baz, select={baz: []}),
        model_class=DeterministicLinker,
        settings={"left_id": "id", "right_id": "id", "comparisons": ""},
        truth=1,
    )

    # Inherit from a source directly
    assert d_foo.sources == {foo.name}

    # Inherit one source from a previous step
    assert foo_bar.sources == {foo.name, bar.name}

    # Inherit multiple sources from a previous step
    assert foo_bar_baz.sources == {foo.name, bar.name, baz.name}


@patch("matchbox.client.dags._handler.index")
@patch.object(Source, "hash_data")
def test_index_step_run(
    hash_data: Mock, handler_index_mock: Mock, sqlite_warehouse: Engine
):
    """Tests that an index step correctly calls the index handler."""
    foo = source_factory(name="foo", engine=sqlite_warehouse).source

    # Test with batch size
    batch_size = 100

    i_foo = IndexStep(source=foo, batch_size=batch_size)
    i_foo.run()

    hash_data.assert_called_once_with(batch_size=batch_size)
    assert (
        handler_index_mock.call_args_list[0].kwargs["source_config"].model_dump()
        == foo.config.model_dump()
    )


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
        patch("matchbox.client.dags.query") as query_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run deduper
        foo_testkit = source_factory(
            name="foo", engine=sqlite_warehouse
        ).write_to_location()
        foo = foo_testkit.source

        i_foo = IndexStep(source=foo)

        d_foo = DedupeStep(
            name="d_foo",
            description="",
            left=StepInput(
                prev_node=i_foo,
                select={foo: []},
                cleaning_dict=None,
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
            [Selector(source=foo, fields=[])],
            return_type="polars",
            return_leaf_id=False,
            threshold=d_foo.left.threshold,
            resolution=d_foo.left.name,
            batch_size=100 if batched else None,
            combine_type="concat",
        )

        # Model is created and run
        make_model_mock.assert_called_once_with(
            name=d_foo.name,
            description=d_foo.description,
            model_class=d_foo.model_class,
            model_settings=d_foo.settings,
            left_data=query_mock.return_value,
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
        patch("matchbox.client.dags.query") as query_mock,
    ):
        # Complete mock set up
        model_mock = Mock()
        make_model_mock.return_value = model_mock

        results_mock = Mock()
        results_mock.to_matchbox = Mock()

        model_mock.run = Mock(return_value=results_mock)

        # Set up and run linker
        foo_testkit = source_factory(
            name="foo", engine=sqlite_warehouse
        ).write_to_location()
        foo = foo_testkit.source

        bar_testkit = source_factory(
            name="bar", engine=sqlite_warehouse
        ).write_to_location()
        bar = bar_testkit.source

        i_foo = IndexStep(source=foo)
        i_bar = IndexStep(source=bar)

        foo_bar = LinkStep(
            name="foo_bar",
            description="",
            left=StepInput(
                prev_node=i_foo,
                select={foo: ["company_name", "crn"]},
                cleaning_dict=None,
                threshold=0.5,
                batch_size=100 if batched else None,
            ),
            right=StepInput(
                prev_node=i_bar,
                select={bar: []},
                cleaning_dict=None,
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
            [Selector(source=foo, fields=foo.config.index_fields)],
            return_type="polars",
            return_leaf_id=False,
            threshold=foo_bar.left.threshold,
            resolution=foo_bar.left.name,
            batch_size=100 if batched else None,
            combine_type="concat",
        )
        assert query_mock.call_args_list[1] == call(
            [Selector(source=bar, fields=[])],
            return_type="polars",
            return_leaf_id=False,
            threshold=foo_bar.right.threshold,
            resolution=foo_bar.right.name,
            batch_size=100 if batched else None,
            combine_type="concat",
        )

        # Model is created and run
        make_model_mock.assert_called_once_with(
            name=foo_bar.name,
            description=foo_bar.description,
            model_class=foo_bar.model_class,
            model_settings=foo_bar.settings,
            left_data=query_mock.return_value,
            left_resolution=foo_bar.left.name,
            right_data=query_mock.return_value,
            right_resolution=foo_bar.right.name,
        )
        model_mock.run.assert_called_once()

        # Results are stored
        model_mock.run().to_matchbox.assert_called_once()
        assert model_mock.truth == 1


@patch("matchbox.client.dags._handler.index")
@patch.object(Source, "hash_data", autospec=True, wraps=Source.hash_data)
@patch.object(DedupeStep, "run")
@patch.object(LinkStep, "run")
def test_dag_runs(
    link_run: Mock,
    dedupe_run: Mock,
    hash_data_spy: Mock,
    handler_index: Mock,
    sqlite_warehouse: Engine,
):
    """A legal DAG can be built and run."""
    # Assemble DAG
    dag = DAG()

    # Set up constituents
    foo_testkit = source_factory(
        name="foo", engine=sqlite_warehouse
    ).write_to_location()

    foo = foo_testkit.source
    bar = source_factory(name="bar", engine=sqlite_warehouse).write_to_location().source
    baz = source_factory(name="baz", engine=sqlite_warehouse).write_to_location().source

    # Structure: SourceConfigs can be added directly, with and without IndexStep
    i_foo = IndexStep(source=foo, batch_size=100)
    dag.add_steps(i_foo)

    i_bar, i_baz = dag.add_sources(bar, baz, batch_size=200)

    assert set(dag.nodes.keys()) == {foo.name, bar.name, baz.name}

    # Structure: IndexSteps can be deduped
    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=i_foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    # Structure:
    # - IndexSteps can be passed directly to linkers
    # - Or, linkers can take dedupers
    foo_bar = LinkStep(
        left=StepInput(
            prev_node=d_foo,
            select={foo: []},
        ),
        right=StepInput(
            prev_node=i_bar,
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
            cleaning_dict=None,
        ),
        right=StepInput(
            prev_node=i_baz,
            select={baz: []},
            cleaning_dict=None,
        ),
        name="foo_bar_baz",
        description="",
        model_class=DeterministicLinker,
        settings={},
        truth=1,
    )

    dag.add_steps(d_foo, foo_bar, foo_bar_baz)
    assert set(dag.nodes.keys()) == {
        foo.name,
        bar.name,
        baz.name,
        d_foo.name,
        foo_bar.name,
        foo_bar_baz.name,
    }

    # Prepare DAG
    dag.prepare()
    s_foo, s_bar, s_d_foo, s_foo_bar, s_foo_bar_baz = (
        dag.sequence.index(foo.name),
        dag.sequence.index(bar.name),
        dag.sequence.index(d_foo.name),
        dag.sequence.index(foo_bar.name),
        dag.sequence.index(foo_bar_baz.name),
    )
    assert s_foo < s_d_foo < s_foo_bar < s_foo_bar_baz
    assert s_bar < s_foo_bar < s_foo_bar_baz

    # Run DAG
    dag.run()

    # By default outputs are discarded
    assert not dag.debug_outputs

    assert handler_index.call_count == 3

    # Verify batch sizes passed to source_config.hash_data
    assert {
        hash_data_spy.call_args_list[0].kwargs["batch_size"],
        hash_data_spy.call_args_list[1].kwargs["batch_size"],
        hash_data_spy.call_args_list[2].kwargs["batch_size"],
    } == {100, 200}

    # Verify the right sources were sent to index
    assert {
        handler_index.call_args_list[0].kwargs["source_config"],
        handler_index.call_args_list[1].kwargs["source_config"],
        handler_index.call_args_list[2].kwargs["source_config"],
    } == {foo.config, bar.config, baz.config}

    assert dedupe_run.call_count == 1
    assert link_run.call_count == 2

    # Real sources can be overridden for debugging
    handler_index.reset_mock()

    dag.run(
        DAGDebugOptions(
            override_sources={foo.name: pl.from_arrow(foo_testkit.data)[:2]}
        )
    )

    overridden = handler_index.call_args.kwargs["data_hashes"]
    assert len(overridden) == 2

    # Outputs can be kept for debugging
    dag.run(DAGDebugOptions(keep_outputs=True))
    assert len(dag.debug_outputs.keys()) == 6
    dag.run()
    # Re-running DAG drops debug outputs
    assert not dag.debug_outputs

    # Reset mocks to test the start argument
    handler_index.reset_mock()
    dedupe_run.reset_mock()
    link_run.reset_mock()

    # Can specify a start
    dag.run(DAGDebugOptions(start="foo_bar"))

    # Verify only steps from foo_bar onward were executed
    assert handler_index.call_count == 0
    assert dedupe_run.call_count == 0
    assert link_run.call_count == 2

    # Reset mocks to test the finish argument
    handler_index.reset_mock()
    dedupe_run.reset_mock()
    link_run.reset_mock()

    # Run DAG with finish at foo_bar step (should not execute foo_bar_baz)
    dag.run(DAGDebugOptions(finish="foo_bar"))

    # Verify steps up to foo_bar were executed but foo_bar_baz was not
    assert handler_index.call_count == 3
    assert dedupe_run.call_count == 1
    assert link_run.call_count == 1

    # Reset mocks again to test combining start and finish
    handler_index.reset_mock()
    dedupe_run.reset_mock()
    link_run.reset_mock()

    # Run from d_foo to foo_bar (skipping sources at start and foo_bar_baz at end)
    dag.run(DAGDebugOptions(start="d_foo", finish="foo_bar"))

    # Verify only the specified segment was executed
    assert handler_index.call_count == 0
    assert dedupe_run.call_count == 1
    assert link_run.call_count == 1


def test_dag_missing_dependency(sqlite_warehouse: Engine):
    """Steps cannot be added before their dependencies."""
    foo = source_factory(name="foo", engine=sqlite_warehouse).source

    i_foo = IndexStep(source=foo)

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=i_foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    dag = DAG()
    with pytest.raises(ValueError, match="Dependency"):
        dag.add_steps(d_foo)


def test_dag_name_clash(sqlite_warehouse: Engine):
    """Names across sources and steps must be unique."""
    foo = source_factory(name="foo", engine=sqlite_warehouse).source
    bar = source_factory(name="bar", engine=sqlite_warehouse).source

    i_foo = IndexStep(source=foo)
    i_bar = IndexStep(source=bar)

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=i_foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )
    d_bar_wrong = DedupeStep(
        # Name clash!
        name="d_foo",
        description="",
        left=StepInput(prev_node=i_bar, select={bar: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    dag = DAG()
    dag.add_steps(i_foo, d_foo)

    with pytest.raises(ValueError, match="already taken"):
        dag.add_steps(d_bar_wrong)

    # DAG is not modified by failed attempt
    assert dag.nodes["d_foo"] == d_foo
    # We didn't overwrite d_foo's dependencies
    assert dag.graph["d_foo"] == [foo.name]


def test_dag_disconnected(sqlite_warehouse: Engine):
    """Nodes cannot be disconnected."""
    foo = source_factory(name="foo", engine=sqlite_warehouse).source
    bar = source_factory(name="bar", engine=sqlite_warehouse).source

    dag = DAG()
    _ = dag.add_sources(foo, bar)

    with pytest.raises(ValueError, match="disconnected"):
        dag.prepare()


def test_dag_draw(sqlite_warehouse: Engine):
    """Test that the draw method produces a correct string representation of the DAG."""
    # Set up a simple DAG
    dag = DAG()

    foo = source_factory(name="foo", engine=sqlite_warehouse).source
    bar = source_factory(name="bar", engine=sqlite_warehouse).source
    baz = source_factory(name="baz", engine=sqlite_warehouse).source

    i_foo, i_bar, i_baz = dag.add_sources(foo, bar, baz)

    d_foo = DedupeStep(
        name="d_foo",
        description="",
        left=StepInput(prev_node=i_foo, select={foo: []}),
        model_class=NaiveDeduper,
        settings={},
        truth=1,
    )

    foo_bar = LinkStep(
        left=StepInput(prev_node=d_foo, select={foo: []}),
        right=StepInput(prev_node=i_bar, select={bar: []}),
        name="foo_bar",
        description="",
        model_class=DeterministicLinker,
        settings={},
        truth=1,
    )

    foo_bar_baz = LinkStep(
        left=StepInput(prev_node=foo_bar, select={foo: [], bar: []}),
        right=StepInput(prev_node=i_baz, select={baz: []}),
        name="foo_bar_baz",
        description="",
        model_class=DeterministicLinker,
        settings={},
        truth=1,
    )

    dag.add_steps(d_foo, foo_bar, foo_bar_baz)

    # Prepare the DAG and draw it
    dag.prepare()
    tree_str = dag.draw()

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
    start_time = datetime.datetime.now()
    doing = "d_foo"
    foo_bar.last_run = datetime.datetime.now()

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
