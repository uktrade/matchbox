"""Factory helpers for resolver testkits."""

import json
from collections.abc import Iterable
from typing import Any

import polars as pl
from faker import Faker
from pydantic import BaseModel, ConfigDict

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.resolvers import (
    Components,
    ComponentsSettings,
    Resolver,
    ResolverMethod,
    ResolverSettings,
)
from matchbox.common.dtos import ResolverResolutionName
from matchbox.common.factories.entities import (
    ClusterEntity,
    SourceEntity,
)
from matchbox.common.factories.models import ModelTestkit

_ASSIGNMENT_SCHEMA = {"cluster_id": pl.UInt64, "node_id": pl.UInt64}


class ResolverTestkit(BaseModel):
    """Resolver plus local expected data for tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolver: Resolver
    assignments: pl.DataFrame
    entities: tuple[ClusterEntity, ...]

    @property
    def name(self) -> str:
        """Return resolver name."""
        return self.resolver.name

    def into_dag(self) -> dict[str, Any]:
        """Return kwargs for explicit DAG insertion."""
        config = self.resolver.config
        return {
            "name": self.resolver.name,
            "inputs": list(config.inputs),
            "resolver_class": config.resolver_class,
            "resolver_settings": json.loads(config.resolver_settings),
            "description": self.resolver.description,
        }


def _as_assignments(data: pl.DataFrame) -> pl.DataFrame:
    """Coerce client_cluster_id outputs as if they were mapped server-side."""
    if data.height == 0:
        return pl.DataFrame(schema=_ASSIGNMENT_SCHEMA)
    return (
        data.select("cluster_id", "node_id")
        .cast(_ASSIGNMENT_SCHEMA)
        .unique()
        .sort(["cluster_id", "node_id"])
    )


def resolver_factory(
    *,
    dag: DAG,
    inputs: Iterable[ModelTestkit | ResolverTestkit],
    true_entities: Iterable[SourceEntity],
    name: ResolverResolutionName | None = None,
    resolver_class: type[ResolverMethod] | str = Components,
    resolver_settings: ResolverSettings | dict[str, Any] | None = None,
    thresholds: dict[str, int] | None = None,
    description: str | None = None,
    seed: int = 42,
) -> ResolverTestkit:
    """Build a detached resolver testkit and local expected entities."""
    input_map: dict[str, ModelTestkit | ResolverTestkit] = {}
    for testkit in inputs:
        if not isinstance(testkit, (ModelTestkit, ResolverTestkit)):
            raise TypeError(
                "resolver_factory inputs must be ModelTestkit or ResolverTestkit."
            )
        input_map.setdefault(testkit.name, testkit)

    if resolver_settings is None:
        if resolver_class not in {Components, "Components"}:
            raise ValueError(
                "resolver_settings must be provided for non-Components resolvers."
            )
        resolver_settings = ComponentsSettings(
            thresholds=thresholds or {input_name: 0 for input_name in input_map}
        )
    elif thresholds is not None:
        raise ValueError("Cannot set both resolver_settings and thresholds.")

    resolver_inputs: list[Model | Resolver] = []
    for testkit in input_map.values():
        if isinstance(testkit, ModelTestkit):
            if testkit.model.dag != dag:
                raise ValueError("Cannot mix DAGs when building a resolver testkit.")
            if testkit.model.results is None:
                testkit.fake_run()
            resolver_inputs.append(testkit.model)
            continue

        if testkit.resolver.dag != dag:
            raise ValueError("Cannot mix DAGs when building a resolver testkit.")
        testkit.resolver.results = _as_assignments(
            testkit.resolver.results
            if testkit.resolver.results is not None
            else testkit.assignments
        )
        resolver_inputs.append(testkit.resolver)

    generator = Faker()
    generator.seed_instance(seed)
    resolver = Resolver(
        dag=dag,
        name=name or generator.unique.word(),
        inputs=resolver_inputs,
        resolver_class=resolver_class,
        resolver_settings=resolver_settings,
        description=description,
    )
    assignments = _as_assignments(resolver.run())

    source_names = tuple(sorted(resolver.sources))
    entities = tuple(
        projected
        for entity in true_entities
        if (projected := entity.to_cluster_entity(*source_names)) is not None
    )

    return ResolverTestkit(
        resolver=resolver,
        assignments=assignments,
        entities=entities,
    )
