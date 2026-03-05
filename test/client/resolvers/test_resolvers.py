"""Tests for resolver ABC and API contracts."""

from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from sqlalchemy import Engine

from matchbox.client.resolvers import Components, ComponentsSettings
from matchbox.client.resolvers.base import ResolverMethod
from matchbox.common.arrow import SCHEMA_CLUSTERS
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.models import ModelTestkit, model_factory
from matchbox.common.factories.sources import linked_sources_factory, source_factory

ResolverConfigurator = Callable[[list[ModelTestkit]], dict[str, Any]]


# Methodology configuration adapters


def configure_components_resolver(model_testkits: list) -> dict[str, Any]:
    """Configure settings for the Components resolver.

    Args:
        model_testkits: List of ModelTestkit objects that need resolver thresholds

    Returns:
        A dictionary with validated settings for ComponentsSettings
    """
    return ComponentsSettings(
        thresholds={testkit.name: 0.0 for testkit in model_testkits}
    ).model_dump()


RESOLVERS: list[pytest.param] = [
    pytest.param(Components, configure_components_resolver, id="Components"),
    # Add more resolver classes and configuration functions here
]


# Test cases


@pytest.mark.parametrize(("ResolverClass", "configure_resolver"), RESOLVERS)
def test_resolver_output_conforms_to_schema(
    ResolverClass: type[ResolverMethod],
    configure_resolver: ResolverConfigurator,
) -> None:
    """Resolver output should conform to SCHEMA_CLUSTERS."""
    dag_testkit = TestkitDAG()
    linked = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)
    true_entities = tuple(linked.true_entities)

    model_testkit = model_factory(
        name="link_crn_dh",
        left_testkit=linked.sources["crn"],
        right_testkit=linked.sources["dh"],
        true_entities=true_entities,
    ).fake_run()
    dag_testkit.add_model(model_testkit)

    settings = configure_resolver([model_testkit])
    resolver_instance = ResolverClass(
        settings=ResolverClass.model_fields["settings"].annotation(**settings)
    )

    model_edges = {model_testkit.name: model_testkit.model.results}
    result = resolver_instance.compute_clusters(model_edges=model_edges)

    assert result.schema == pl.Schema(SCHEMA_CLUSTERS)


@pytest.mark.parametrize(("ResolverClass", "configure_resolver"), RESOLVERS)
def test_resolver_happy_path_matches_factory(
    ResolverClass: type[ResolverMethod],
    configure_resolver: ResolverConfigurator,
) -> None:
    """Resolver assignments should match factory ground truth."""
    dag_testkit = TestkitDAG()
    linked = linked_sources_factory(dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)
    true_entities = tuple(linked.true_entities)

    crn_dh_model = model_factory(
        name="link_crn_dh",
        left_testkit=linked.sources["crn"],
        right_testkit=linked.sources["dh"],
        true_entities=true_entities,
    ).fake_run()
    crn_cdms_model = model_factory(
        name="link_crn_cdms",
        left_testkit=linked.sources["crn"],
        right_testkit=linked.sources["cdms"],
        true_entities=true_entities,
    ).fake_run()
    dag_testkit.add_model(crn_dh_model)
    dag_testkit.add_model(crn_cdms_model)

    settings = configure_resolver([crn_dh_model, crn_cdms_model])
    resolver_instance = ResolverClass(
        settings=ResolverClass.model_fields["settings"].annotation(**settings)
    )
    model_edges = {
        crn_dh_model.name: crn_dh_model.model.results,
        crn_cdms_model.name: crn_cdms_model.model.results,
    }
    result = resolver_instance.compute_clusters(model_edges=model_edges)

    identical, report = linked.diff_clusters(
        assignments=result,
        sources=["crn", "dh", "cdms"],
        input_clusters={
            crn_dh_model.name: tuple(crn_dh_model.left_clusters.values())
            + tuple(crn_dh_model.right_clusters.values()),
            crn_cdms_model.name: tuple(crn_cdms_model.left_clusters.values())
            + tuple(crn_cdms_model.right_clusters.values()),
        },
    )

    assert identical, f"Expected perfect clusters but got: {report}"


# API contract


def test_resolver_run_requires_materialised_inputs(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Resolver run should require explicit upstream materialisation."""
    source_testkit = source_factory(name="foo", engine=sqla_sqlite_warehouse)
    source = source_testkit.source.dag.source(**source_testkit.into_dag())
    model = source.query().deduper(
        name="foo_dedupe",
        model_class="NaiveDeduper",
        model_settings={"unique_fields": []},
    )
    resolver = source.dag.resolver(
        name="resolver",
        inputs=[model],
        resolver_class=Components,
        resolver_settings={"thresholds": {model.name: 0}},
    )

    with pytest.raises(ValueError, match="has no local results"):
        resolver.run()
