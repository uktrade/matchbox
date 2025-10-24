from collections.abc import Callable

import polars as pl
import pytest
from httpx import Response
from polars.testing import assert_frame_equal
from pyarrow import Table
from respx import MockRouter
from sqlalchemy import Engine

from matchbox.client.dags import DAG
from matchbox.client.eval import get_samples
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.arrow import SCHEMA_EVAL_SAMPLES, table_to_buffer
from matchbox.common.dtos import Collection, Run
from matchbox.common.exceptions import MatchboxSourceTableError
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.sources import source_from_tuple


def test_get_samples(
    matchbox_api: MockRouter,
    sqlite_warehouse: Engine,
    sqlite_in_memory_warehouse: Engine,
    env_setter: Callable[[str, str], None],
):
    # Make dummmy data
    user_id = 12

    # Foo has two identical rows
    foo_testkit = source_from_tuple(
        data_tuple=({"col": 1}, {"col": 1}, {"col": 2}, {"col": 3}, {"col": 4}),
        data_keys=["1", "1bis", "2", "3", "4"],
        name="foo",
        location_name="db",
        engine=sqlite_warehouse,
    ).write_to_location()

    bar_testkit = source_from_tuple(
        data_tuple=({"col": 1}, {"col": 2}, {"col": 3}, {"col": 4}),
        data_keys=["a", "b", "c", "d"],
        name="bar",
        location_name="db",
        engine=sqlite_warehouse,
    ).write_to_location()

    # This will be excluded as the location name differs
    baz_testkit = source_from_tuple(
        data_tuple=({"col": 1},),
        data_keys=["x"],
        name="baz",
        location_name="db_other",
        engine=sqlite_warehouse,
    ).write_to_location()

    dag = TestkitDAG().dag

    foo = dag.source(**foo_testkit.into_dag())
    bar = dag.source(**bar_testkit.into_dag())
    baz = dag.source(**baz_testkit.into_dag())
    foo_bar = foo.query().linker(
        bar.query(),
        name="linker1",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.key=r.key"},
    )
    foo_bar_baz = foo_bar.query(foo, bar).linker(
        baz.query(),
        name="linker2",
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.key=r.key"},
    )

    # Mock the collection and run endpoint that load_pending() calls
    collection_data = Collection(runs=[dag.run])
    run_data = Run(
        run_id=dag.run,
        mutable=True,
        resolutions={
            "foo": foo_testkit.source.to_resolution(),
            "bar": bar_testkit.source.to_resolution(),
            "baz": baz_testkit.source.to_resolution(),
            "linker1": foo_bar.to_resolution(),
            "linker2": foo_bar_baz.to_resolution(),
        },
    )

    # Mock API endpoints for load_pending
    matchbox_api.get(f"/collections/{dag.name}").mock(
        return_value=Response(200, content=collection_data.model_dump_json())
    )
    matchbox_api.get(f"/collections/{dag.name}/runs/{dag.run}").mock(
        return_value=Response(200, content=run_data.model_dump_json())
    )

    # Mock samples
    samples = Table.from_pylist(
        [
            # Source foo - with two keys for one leaf
            {"root": 10, "leaf": 1, "key": "1", "source": "foo"},
            {"root": 10, "leaf": 1, "key": "1bis", "source": "foo"},
            {"root": 10, "leaf": 2, "key": "2", "source": "foo"},
            {"root": 11, "leaf": 3, "key": "3", "source": "foo"},
            {"root": 11, "leaf": 4, "key": "4", "source": "foo"},
            # Source bar
            {"root": 10, "leaf": 5, "key": "a", "source": "bar"},
            {"root": 10, "leaf": 6, "key": "b", "source": "bar"},
            {"root": 11, "leaf": 7, "key": "c", "source": "bar"},
            {"root": 11, "leaf": 8, "key": "d", "source": "bar"},
            # Source baz
            {"root": 10, "leaf": 1, "key": "x", "source": "baz"},
        ],
        schema=SCHEMA_EVAL_SAMPLES,
    )
    # There will be nulls in case of a schema mismatch
    assert len(samples.drop_null()) == len(samples)

    matchbox_api.get("/eval/samples").mock(
        return_value=Response(200, content=table_to_buffer(samples).read())
    )

    # Create a fresh DAG and load it with warehouse location
    # (can't reuse the existing dag as it already has sources added)

    loaded_dag: DAG = (
        DAG(name=str(dag.name)).load_pending().set_client(sqlite_warehouse)
    )

    # Check results - test with samples that include all three sources
    # All three sources (foo, bar, baz) are in loaded_dag with the warehouse location
    samples_all = get_samples(
        n=10,
        resolution=dag.final_step.resolution_path.name,
        user_id=user_id,
        dag=loaded_dag,
    )

    assert sorted(samples_all.keys()) == [10, 11]

    # Now test with samples that only include foo and bar (not baz)
    samples_no_baz = Table.from_pylist(
        [
            # Source foo - with two keys for one leaf
            {"root": 10, "leaf": 1, "key": "1", "source": "foo"},
            {"root": 10, "leaf": 1, "key": "1bis", "source": "foo"},
            {"root": 10, "leaf": 2, "key": "2", "source": "foo"},
            {"root": 11, "leaf": 3, "key": "3", "source": "foo"},
            {"root": 11, "leaf": 4, "key": "4", "source": "foo"},
            # Source bar
            {"root": 10, "leaf": 5, "key": "a", "source": "bar"},
            {"root": 10, "leaf": 6, "key": "b", "source": "bar"},
            {"root": 11, "leaf": 7, "key": "c", "source": "bar"},
            {"root": 11, "leaf": 8, "key": "d", "source": "bar"},
        ],
        schema=SCHEMA_EVAL_SAMPLES,
    )
    matchbox_api.get("/eval/samples").mock(
        return_value=Response(200, content=table_to_buffer(samples_no_baz).read())
    )

    samples = get_samples(
        n=10,
        resolution=dag.final_step.resolution_path.name,
        user_id=user_id,
        dag=loaded_dag,
    )

    assert sorted(samples.keys()) == [10, 11]

    expected_sample_10 = pl.DataFrame(
        {
            "leaf": [1, 1, 2, 5, 6],
            "key": ["1", "1bis", "2", "a", "b"],
            "foo_col": [1, 1, 2, None, None],
            "bar_col": [None, None, None, 1, 2],
        }
    )

    expected_sample_11 = pl.DataFrame(
        {
            "leaf": [3, 4, 7, 8],
            "key": ["3", "4", "c", "d"],
            "foo_col": [3, 4, None, None],
            "bar_col": [None, None, 3, 4],
        }
    )

    # EvaluationItems.dataframe contains the data
    assert_frame_equal(
        samples[10].dataframe,
        expected_sample_10,
        check_column_order=False,
        check_row_order=False,
        check_dtypes=False,
    )
    assert_frame_equal(
        samples[11].dataframe,
        expected_sample_11,
        check_column_order=False,
        check_row_order=False,
        check_dtypes=False,
    )

    # What happens if no samples are available?
    matchbox_api.get("/eval/samples").mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                Table.from_pylist([], schema=SCHEMA_EVAL_SAMPLES)
            ).read(),
        )
    )

    no_samples = get_samples(
        n=10,
        resolution=dag.final_step.resolution_path.name,
        user_id=user_id,
        dag=loaded_dag,
    )
    assert no_samples == {}

    # What happens if source cannot be queried using client?
    # Create new DAG with wrong warehouse (in-memory, no tables)
    bad_dag: DAG = (
        DAG(name=str(dag.name)).load_pending().set_client(sqlite_in_memory_warehouse)
    )

    matchbox_api.get("/eval/samples").mock(
        return_value=Response(200, content=table_to_buffer(samples_no_baz).read())
    )

    with pytest.raises(MatchboxSourceTableError, match="Could not query source"):
        get_samples(
            n=10,
            resolution=dag.final_step.resolution_path.name,
            user_id=user_id,
            dag=bad_dag,
        )
