"""Tests for resolver testkit factory helpers."""

import polars as pl
import pytest

from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.entities import FeatureConfig, SourceEntity
from matchbox.common.factories.models import ModelTestkit, model_factory
from matchbox.common.factories.resolvers import (
    ResolverTestkit,
    resolver_factory,
)
from matchbox.common.factories.sources import (
    SourceTestkitParameters,
    linked_sources_factory,
)


def _build_model_testkit(
    dag_testkit: TestkitDAG, name: str
) -> tuple[ModelTestkit, tuple[SourceEntity, ...]]:
    source_name = f"{name}_source"
    linked_testkit = linked_sources_factory(
        source_parameters=(
            SourceTestkitParameters(
                name=source_name,
                features=(FeatureConfig(name="company", base_generator="company"),),
            ),
        ),
        n_true_entities=10,
        dag=dag_testkit.dag,
    )
    dag_testkit.add_linked_sources(linked_testkit)

    source_testkit = linked_testkit.sources[source_name]
    model_testkit = model_factory(
        name=name,
        left_testkit=source_testkit,
        true_entities=tuple(linked_testkit.true_entities),
    )
    dag_testkit.add_model(model_testkit)
    return model_testkit.fake_run(), tuple(linked_testkit.true_entities)


def test_resolver_factory_is_detached_with_default_thresholds() -> None:
    dag_testkit = TestkitDAG()
    model_testkit, true_entities = _build_model_testkit(dag_testkit, "dedupe_foo")

    resolver_testkit = resolver_factory(
        dag=dag_testkit.dag,
        name=f"resolver_{model_testkit.name}",
        inputs=[model_testkit],
        true_entities=true_entities,
    )

    assert isinstance(resolver_testkit, ResolverTestkit)
    assert resolver_testkit.name not in dag_testkit.dag.nodes
    assert resolver_testkit.resolver.resolver_settings.thresholds == {
        model_testkit.name: 0
    }
    assert resolver_testkit.assignments.schema == {
        "cluster_id": pl.UInt64,
        "node_id": pl.UInt64,
    }
    assert resolver_testkit.into_dag()["inputs"] == [model_testkit.name]


def test_resolver_testkit_into_dag_can_be_attached() -> None:
    dag_testkit = TestkitDAG()
    model_testkit, true_entities = _build_model_testkit(dag_testkit, "dedupe_foo")
    resolver_testkit = resolver_factory(
        dag=dag_testkit.dag,
        name=f"resolver_{model_testkit.name}",
        inputs=[model_testkit],
        true_entities=true_entities,
        thresholds={model_testkit.name: 0},
    )

    dag_testkit.add_resolver(resolver_testkit)
    attached = dag_testkit.dag.get_resolver(resolver_testkit.name)
    assert dag_testkit.resolvers[resolver_testkit.name].resolver is attached


def test_resolver_factory_requires_testkit_inputs() -> None:
    dag_testkit = TestkitDAG()
    model_testkit, true_entities = _build_model_testkit(dag_testkit, "dedupe_foo")

    with pytest.raises(
        TypeError,
        match="resolver_factory inputs must be ModelTestkit or ResolverTestkit",
    ):
        resolver_factory(
            dag=dag_testkit.dag,
            name="resolver_invalid",
            inputs=[model_testkit.model],  # type: ignore[list-item]
            true_entities=true_entities,
        )


def test_resolver_factory_supports_resolver_chaining_with_entities() -> None:
    dag_testkit = TestkitDAG()
    foo_model, foo_truth = _build_model_testkit(dag_testkit, "dedupe_foo")
    bar_model, bar_truth = _build_model_testkit(dag_testkit, "dedupe_bar")
    all_truth = tuple({entity.id: entity for entity in foo_truth + bar_truth}.values())

    resolver_inner = resolver_factory(
        dag=dag_testkit.dag,
        name="resolver_inner",
        inputs=[foo_model, bar_model],
        true_entities=all_truth,
        thresholds={foo_model.name: 0, bar_model.name: 0},
    )
    resolver_outer = resolver_factory(
        dag=dag_testkit.dag,
        name="resolver_outer",
        inputs=[resolver_inner, foo_model],
        true_entities=all_truth,
        thresholds={resolver_inner.name: 0, foo_model.name: 0},
    )

    assert resolver_outer.assignments.schema == {
        "cluster_id": pl.UInt64,
        "node_id": pl.UInt64,
    }
    assert resolver_outer.resolver.resolver_settings.thresholds == {
        resolver_inner.name: 0,
        foo_model.name: 0,
    }

    assert isinstance(resolver_outer.entities, tuple)
