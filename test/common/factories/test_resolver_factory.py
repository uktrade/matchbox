"""Tests for resolver testkit factory helpers."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from matchbox.common.arrow import SCHEMA_CLUSTERS
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.resolvers import (
    MockResolver,
    ResolverTestkit,
    resolver_factory,
)
from matchbox.common.factories.sources import linked_sources_factory


def test_resolver_factory_is_detached() -> None:
    """Test resolver factory isn't attached to the DAG by default.

    This must be done explicitly.
    """
    dag_testkit = TestkitDAG()
    linked = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)

    model_testkit = model_factory(
        dag=dag_testkit.dag,
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    ).fake_run()
    dag_testkit.add_model(model_testkit)

    resolver_testkit = resolver_factory(dag=dag_testkit.dag, inputs=[model_testkit])

    assert isinstance(resolver_testkit, ResolverTestkit)
    assert resolver_testkit.name not in dag_testkit.dag.nodes
    assert resolver_testkit.resolver.results is None
    assert resolver_testkit.resolver.resolver_class is MockResolver
    assert resolver_testkit.assignments.schema == pl.Schema(SCHEMA_CLUSTERS)
    assert resolver_testkit.into_dag()["inputs"] == [model_testkit.name]
    assert resolver_testkit.into_dag()["resolver_class"] == "MockResolver"


def test_resolver_testkit_into_dag_can_be_attached() -> None:
    """Test the resolver can be attached to the DAG."""
    dag_testkit = TestkitDAG()
    linked = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)

    model_testkit = model_factory(
        dag=dag_testkit.dag,
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    ).fake_run()
    dag_testkit.add_model(model_testkit)

    resolver_testkit = resolver_factory(dag=dag_testkit.dag, inputs=[model_testkit])
    dag_testkit.add_resolver(resolver_testkit)

    attached = dag_testkit.dag.get_resolver(resolver_testkit.name)
    assert dag_testkit.resolvers[resolver_testkit.name].resolver is attached


def test_resolver_testkit_fake_run_materialises_results() -> None:
    """Test the resolver testkit's fake run works."""
    dag_testkit = TestkitDAG()
    linked = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)

    model_testkit = model_factory(
        dag=dag_testkit.dag,
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    ).fake_run()

    resolver_testkit = resolver_factory(dag=dag_testkit.dag, inputs=[model_testkit])

    with pytest.raises(RuntimeError, match="must be run"):
        resolver_testkit.resolver.to_resolution()

    resolver_testkit.fake_run()
    assert_frame_equal(resolver_testkit.resolver.results, resolver_testkit.assignments)
    resolution = resolver_testkit.resolver.to_resolution()
    assert resolution.config.resolver_class == "MockResolver"


def test_resolver_factory_requires_testkit_inputs() -> None:
    """Test that resolver_factory rejects non-ModelTestkit inputs."""
    dag_testkit = TestkitDAG()
    linked = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)
    crn_model = model_factory(
        name="dedupe_crn",
        dag=dag_testkit.dag,
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    ).fake_run()
    dag_testkit.add_model(crn_model)
    dh_model = model_factory(
        name="dedupe_dh",
        dag=dag_testkit.dag,
        left_testkit=linked.sources["dh"],
        true_entities=tuple(linked.true_entities),
    ).fake_run()
    dag_testkit.add_model(dh_model)
    resolver_inner = resolver_factory(dag=dag_testkit.dag, inputs=[crn_model, dh_model])

    with pytest.raises(TypeError, match="resolver_factory inputs must be ModelTestkit"):
        resolver_factory(dag=dag_testkit.dag, inputs=[crn_model.model])

    with pytest.raises(TypeError, match="resolver_factory inputs must be ModelTestkit"):
        resolver_factory(dag=dag_testkit.dag, inputs=[resolver_inner, crn_model])


def test_resolver_factory_can_autobuild() -> None:
    """Tests that the resolver factory can stand on its own."""
    resolver_testkit = resolver_factory()

    assert isinstance(resolver_testkit, ResolverTestkit)
    assert resolver_testkit.resolver.resolver_class is MockResolver
    assert resolver_testkit.resolver.results is None
    assert len(resolver_testkit.resolver.inputs) == 1
    input_name = resolver_testkit.resolver.inputs[0].name
    assert resolver_testkit.resolver.resolver_settings.thresholds == {input_name: 0.0}


def test_resolver_factory_honours_explicit_thresholds() -> None:
    """Explicit thresholds should influence generated assignments."""
    dag_testkit = TestkitDAG()
    linked = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)

    model_testkit = model_factory(
        dag=dag_testkit.dag,
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
        prob_range=(0.5, 0.99),
    ).fake_run()
    assert model_testkit.probabilities.height > 0

    low_threshold = resolver_factory(
        dag=dag_testkit.dag,
        inputs=[model_testkit],
        thresholds={model_testkit.name: 0.0},
    )
    high_threshold = resolver_factory(
        dag=dag_testkit.dag,
        inputs=[model_testkit],
        thresholds={model_testkit.name: 1.0},
    )

    assert low_threshold.assignments.height > 0
    assert high_threshold.assignments.height == 0
