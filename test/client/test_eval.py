import tempfile
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
from matchbox.client.results import ResolvedMatches
from matchbox.common.arrow import (
    SCHEMA_EVAL_SAMPLES,
    SCHEMA_QUERY_WITH_LEAVES,
    table_to_buffer,
)
from matchbox.common.dtos import Collection, Resolution, ResolutionType, Run, User
from matchbox.common.exceptions import MatchboxSourceTableError
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.sources import source_from_tuple


def test_get_samples_local(sqlite_in_memory_warehouse: Engine) -> None:
    """We can sample from a parquet dump."""
    dag = DAG("companies")
    foo = dag.source(
        **source_from_tuple(
            name="foo",
            engine=sqlite_in_memory_warehouse,
            data_keys=["1", "2", "2b", "3"],
            data_tuple=(
                {"field_a": 10},
                {"field_a": 20},
                {"field_a": 20},
                {"field_a": 30},
            ),
        )
        .write_to_location()
        .into_dag()
    )
    bar = dag.source(
        **source_from_tuple(
            name="bar",
            engine=sqlite_in_memory_warehouse,
            data_keys=["a", "b", "c", "d"],
            data_tuple=(
                {"field_a": "1x", "field_b": "1y"},
                {"field_a": "2x", "field_b": "2y"},
                {"field_a": "3x", "field_b": "3y"},
                {"field_a": "4x", "field_b": "4y"},
            ),
        )
        .write_to_location()
        .into_dag()
    )

    # Both foo and bar have a record that's not linked
    # Foo has two keys for one leaf ID
    # Foo and bar have links across; bar also has link within
    foo_query_data = pl.DataFrame(
        [
            {"id": 14, "leaf_id": 1, "key": "1"},
            {"id": 2, "leaf_id": 2, "key": "2"},
            {"id": 2, "leaf_id": 2, "key": "2b"},
            {"id": 356, "leaf_id": 3, "key": "3"},
        ],
        schema=pl.Schema(SCHEMA_QUERY_WITH_LEAVES),
    )

    bar_query_data = pl.DataFrame(
        [
            {"id": 14, "leaf_id": 4, "key": "a"},
            {"id": 356, "leaf_id": 5, "key": "b"},
            {"id": 356, "leaf_id": 6, "key": "c"},
            {"id": 7, "leaf_id": 7, "key": "d"},
        ],
        schema=pl.Schema(SCHEMA_QUERY_WITH_LEAVES),
    )

    rm = ResolvedMatches(
        sources=[foo, bar], query_results=[foo_query_data, bar_query_data]
    )

    # Create a temporary file with .pq suffix
    with tempfile.NamedTemporaryFile(suffix=".pq") as tmp_file:
        # Write the parquet data to the temporary file
        rm.as_dump().write_parquet(tmp_file.name)

        # Use the temporary file in get_samples
        samples = get_samples(
            n=2, dag=dag, user_name="alice", sample_file=tmp_file.name
        )
        assert len(samples) == 2
        possible_clusters = set(foo_query_data["id"]) | set(bar_query_data["id"])
        assert set(samples.keys()) <= possible_clusters


def test_get_samples_remote(
    matchbox_api: MockRouter,
    sqla_sqlite_warehouse: Engine,
    sqlite_in_memory_warehouse: Engine,
    env_setter: Callable[[str, str], None],
) -> None:
    """We can sample from a resolution on the server."""
    # Make dummmy data
    user = User(user_name="alice", email="alice@example.com")

    # Foo has two identical rows
    foo_testkit = source_from_tuple(
        data_tuple=({"col": 1}, {"col": 1}, {"col": 2}, {"col": 3}, {"col": 4}),
        data_keys=["1", "1bis", "2", "3", "4"],
        name="foo",
        location_name="db",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()

    bar_testkit = source_from_tuple(
        data_tuple=({"col": 1}, {"col": 2}, {"col": 3}, {"col": 4}),
        data_keys=["a", "b", "c", "d"],
        name="bar",
        location_name="db",
        engine=sqla_sqlite_warehouse,
    ).write_to_location()

    # This will be excluded as the location name differs
    baz_testkit = source_from_tuple(
        data_tuple=({"col": 1},),
        data_keys=["x"],
        name="baz",
        location_name="db_other",
        engine=sqla_sqlite_warehouse,
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
            "foo": foo_testkit.fake_run().source.to_resolution(),
            "bar": bar_testkit.fake_run().source.to_resolution(),
            "baz": baz_testkit.fake_run().source.to_resolution(),
            "linker1": Resolution(
                fingerprint=b"mock",
                truth=1,
                resolution_type=ResolutionType.MODEL,
                config=foo_bar.config,
            ),
            "linker2": Resolution(
                fingerprint=b"mock2",
                truth=1,
                resolution_type=ResolutionType.MODEL,
                config=foo_bar_baz.config,
            ),
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
        DAG(name=str(dag.name)).load_pending().set_client(sqla_sqlite_warehouse)
    )

    # Check results - test with samples that include all three sources
    # All three sources (foo, bar, baz) are in loaded_dag with the warehouse location
    samples_all = get_samples(
        n=10,
        resolution=dag.final_step.resolution_path.name,
        user_name=user.user_name,
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
        user_name=user.user_name,
        dag=loaded_dag,
    )

    assert sorted(samples.keys()) == [10, 11]

    expected_sample_10 = pl.DataFrame(
        {
            "leaf": [1, 1, 2, 5, 6],
            "foo_col": [1, 1, 2, None, None],
            "bar_col": [None, None, None, 1, 2],
        }
    )

    expected_sample_11 = pl.DataFrame(
        {
            "leaf": [3, 4, 7, 8],
            "foo_col": [3, 4, None, None],
            "bar_col": [None, None, 3, 4],
        }
    )

    # EvaluationItems.records contains the data with qualified column names
    assert_frame_equal(
        samples[10].records,
        expected_sample_10,
        check_column_order=False,
        check_row_order=False,
        check_dtypes=False,
    )
    assert_frame_equal(
        samples[11].records,
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
        user_name=user.user_name,
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
            user_name=user.user_name,
            dag=bad_dag,
        )
