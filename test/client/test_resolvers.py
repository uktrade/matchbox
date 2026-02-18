import polars as pl
import pyarrow as pa
from httpx import Response
from polars.testing import assert_frame_equal
from respx import MockRouter
from sqlalchemy import Engine

from matchbox.common.arrow import table_to_buffer
from matchbox.common.factories.dags import add_components_resolver
from matchbox.common.factories.sources import source_factory


def test_resolver_download_results_uses_resolution_data_endpoint(
    matchbox_api: MockRouter,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Resolver downloads should call resolution `/data`, not per-source `/query`."""
    source_testkit = source_factory(name="foo", engine=sqla_sqlite_warehouse)
    source = source_testkit.source.dag.source(**source_testkit.into_dag())
    source.dag.run = 1
    model = source.query().deduper(
        name="foo_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    resolver = add_components_resolver(
        dag=source.dag,
        name="resolver",
        inputs=[model],
        thresholds={model.name: 0},
    )

    data_route = matchbox_api.get(
        f"/collections/{resolver.dag.name}/runs/{resolver.dag.run}/resolutions/{resolver.name}/data"
    ).mock(
        return_value=Response(
            200,
            content=table_to_buffer(
                pa.table(
                    {
                        "cluster_id": pa.array([1, 1], type=pa.uint64()),
                        "node_id": pa.array([1, 2], type=pa.uint64()),
                    }
                )
            ).read(),
        )
    )

    assignments = resolver.download_results()

    assert data_route.called
    expected = pl.DataFrame(
        {"cluster_id": [1, 1], "node_id": [1, 2]},
        schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64},
    )
    assert_frame_equal(assignments, expected)
