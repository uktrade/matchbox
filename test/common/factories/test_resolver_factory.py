"""Tests for resolver testkit factory helpers."""

from unittest.mock import Mock

import polars as pl
from pyarrow import Table

from matchbox.common.arrow import SCHEMA_RESOLVER_MAPPING
from matchbox.common.dtos import ResolutionPath
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.resolvers import (
    ResolverTestkit,
    resolver_factory,
    resolver_name_for_model,
)
from matchbox.common.factories.sources import source_factory


def test_resolver_factory_defaults_to_components_thresholds() -> None:
    dag_testkit = TestkitDAG()
    source_testkit = source_factory(name="foo", dag=dag_testkit.dag)
    dag_testkit.add_source(source_testkit)
    model_testkit = model_factory(
        name="dedupe_foo",
        left_testkit=source_testkit,
        true_entities=tuple(source_testkit.entities),
    )
    dag_testkit.add_model(model_testkit)
    model_testkit.fake_run()

    resolver_testkit = resolver_factory(
        dag=model_testkit.model.dag,
        name="resolver_dedupe_foo",
        inputs=[model_testkit.model],
    )

    assert isinstance(resolver_testkit, ResolverTestkit)
    assert resolver_testkit.resolver.resolver_settings.thresholds == {
        model_testkit.model.name: 0
    }


def test_resolver_testkit_materialise_uploads_and_hydrates_backend_ids() -> None:
    dag_testkit = TestkitDAG()
    source_testkit = source_factory(name="foo", dag=dag_testkit.dag)
    dag_testkit.add_source(source_testkit)
    model_testkit = model_factory(
        name="dedupe_foo",
        left_testkit=source_testkit,
        true_entities=tuple(source_testkit.entities),
    )
    dag_testkit.add_model(model_testkit)
    model_testkit.fake_run()

    resolver_testkit = resolver_factory(
        dag=model_testkit.model.dag,
        name=resolver_name_for_model(model_testkit.model.name),
        inputs=[model_testkit.model],
        thresholds={model_testkit.model.name: 0},
    )

    backend = Mock()

    def _insert_resolver_data(*, path: ResolutionPath, data: Table) -> Table:
        client_ids = (
            pl.from_arrow(data)
            .select(pl.col("client_cluster_id").unique().sort())
            .to_series()
            .cast(pl.UInt64)
        )
        return (
            pl.DataFrame(
                {
                    "client_cluster_id": client_ids,
                    "cluster_id": client_ids + 100,
                }
            )
            .to_arrow()
            .cast(SCHEMA_RESOLVER_MAPPING)
        )

    backend.insert_resolver_data.side_effect = _insert_resolver_data

    resolver_testkit.materialise(backend=backend)
    backend.create_resolution.assert_called_once()
    backend.insert_resolver_data.assert_called_once()
    assert resolver_testkit.resolver.results is not None
    assert resolver_testkit.resolver.results.columns == ["cluster_id", "node_id"]
    assert resolver_testkit.resolver.results.schema == {
        "cluster_id": pl.UInt64,
        "node_id": pl.UInt64,
    }
