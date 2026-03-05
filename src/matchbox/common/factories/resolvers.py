"""Factory helpers for resolver testkits."""

import json
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, ClassVar, Self

import polars as pl
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.resolvers import (
    Resolver,
    ResolverMethod,
    ResolverSettings,
    add_resolver_class,
)
from matchbox.common.arrow import SCHEMA_CLUSTERS
from matchbox.common.dtos import (
    ModelResolutionName,
    ResolverResolutionName,
    ResolverType,
    SourceResolutionName,
)
from matchbox.common.factories.entities import ClusterEntity, SourceEntity
from matchbox.common.factories.models import ModelTestkit, model_factory
from matchbox.common.factories.sources import linked_sources_factory
from matchbox.common.transform import (
    DisjointSet,
    threshold_float_to_int,
)


class MockResolverSettings(ResolverSettings):
    """Settings type for MockResolver."""

    thresholds: dict[
        ModelResolutionName,
        Annotated[float, Field(ge=0.0, le=1.0)],
    ] = Field(default_factory=dict)

    def validate_inputs(self, model_names: Iterable[ModelResolutionName]) -> None:
        """Validate all model names are present in thresholds."""
        if missing := set(model_names) - set(self.thresholds.keys()):
            raise RuntimeError(f"Missing thresholds for models: {missing}")


def _connected_components_from_edges(
    model_edges: Mapping[ModelResolutionName, pl.DataFrame],
    thresholds: Mapping[ModelResolutionName, float],
) -> pl.DataFrame:
    """Generate clusters from model edge tables and per-model thresholds."""
    djs = DisjointSet[int]()

    for model_name, edges in model_edges.items():
        if edges.height == 0:
            continue

        threshold = threshold_float_to_int(thresholds[model_name])
        filtered_edges = edges.filter(pl.col("probability") >= threshold)
        for left_id, right_id in filtered_edges.select(
            "left_id", "right_id"
        ).iter_rows():
            djs.union(left_id, right_id)

    rows: list[dict[str, int]] = []
    for parent_id, component in enumerate(
        sorted(djs.get_components(), key=min), start=1
    ):
        rows.extend(
            {"parent_id": parent_id, "child_id": node_id}
            for node_id in sorted(component)
        )

    if not rows:
        return pl.from_arrow(SCHEMA_CLUSTERS.empty_table())
    return pl.DataFrame(rows).cast(pl.Schema(SCHEMA_CLUSTERS))


class MockResolver(ResolverMethod):
    """Mock resolver methodology used by resolver testkits."""

    resolver_type: ClassVar[ResolverType] = ResolverType.COMPONENTS
    settings: MockResolverSettings

    def compute_clusters(
        self, model_edges: Mapping[ModelResolutionName, pl.DataFrame]
    ) -> pl.DataFrame:
        """Compute mock clusters with deterministic DSU-connected-components."""
        self.settings.validate_inputs(model_edges.keys())
        return _connected_components_from_edges(
            model_edges=model_edges,
            thresholds=self.settings.thresholds,
        )


add_resolver_class(MockResolver)


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

    def query(self) -> Query:
        """Thin wrapper to Query this testkit's Sources via its Resolver."""
        return self.resolver.query()

    def fake_run(self) -> Self:
        """Set resolver results without running the resolver."""
        self.resolver.results = self.assignments
        return self

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


def resolver_factory(
    dag: DAG | None = None,
    inputs: Iterable[ModelTestkit] | None = None,
    true_entities: Iterable[SourceEntity] | None = None,
    name: ResolverResolutionName | None = None,
    description: str | None = None,
    thresholds: Mapping[ModelResolutionName, float] | None = None,
    seed: int = 42,
) -> ResolverTestkit:
    """Generate a complete resolver testkit.

    Allows autoconfiguration with minimal settings, or more nuanced control.

    Can either be used to generate a resolver in a pipeline, interconnected with
    existing testkit objects, or generate a standalone resolver with random data.

    Args:
        dag: DAG containing this resolver.
            Inferred from the first input testkit if not provided.
            A default DAG is created when inputs are also absent.
        inputs: An iterable of ModelTestkit objects to use as resolver inputs.
            If None, a single default deduper model testkit is created automatically.
            All inputs must belong to the same DAG.
        true_entities: Ground truth SourceEntity objects used to generate the
            expected cluster assignments. If None, the resolver testkit will have
            no expected entities.
        name: Name of the resolver. Defaults to a randomly generated word suffixed
            with '_resolver'.
        description: Description of the resolver.
        thresholds: Per-model probability thresholds in [0.0, 1.0]. If omitted,
            defaults to 0.0 for all resolver inputs.
        seed: Random seed for reproducibility.

    Returns:
        ResolverTestkit: A resolver testkit with generated assignments and expected
            entities.

    Raises:
        TypeError: If any element of inputs is not a ModelTestkit.
        ValueError: If inputs belong to different DAGs.
    """
    if inputs is None:
        if dag is None:
            dag = DAG(name="collection")
            dag.run = 1
        linked = linked_sources_factory(dag=dag, seed=seed)
        default_model = model_factory(
            left_testkit=linked.sources["crn"],
            true_entities=tuple(linked.true_entities),
            seed=seed,
        ).fake_run()
        inputs = (default_model,)
        source_names: set[SourceResolutionName] = set(linked.sources.keys())
    else:
        inputs = tuple(inputs)
        source_names: set[SourceResolutionName] = set()
        for testkit in inputs:
            if not isinstance(testkit, ModelTestkit):
                raise TypeError("resolver_factory inputs must be ModelTestkit.")
            source_names.update(testkit.model.sources)
        dag = dag or inputs[0].model.dag

    input_map: dict[str, ModelTestkit] = {}
    for testkit in inputs:
        input_map.setdefault(testkit.name, testkit)

    resolver_inputs: list[Model] = []
    model_edges: dict[ModelResolutionName, pl.DataFrame] = {}
    for testkit in input_map.values():
        if testkit.model.dag != dag:
            raise ValueError("Cannot mix DAGs when building a resolver testkit.")
        if testkit.model.results is None:
            testkit.fake_run()
        resolver_inputs.append(testkit.model)
        model_edges[testkit.name] = testkit.probabilities

    expected_model_names = set(input_map.keys())
    if thresholds is None:
        thresholds = {name: 0.0 for name in expected_model_names}

    if set(thresholds) != expected_model_names:
        raise ValueError("Threshold keys must exactly match resolver input models.")

    resolver_settings = MockResolverSettings(
        thresholds=thresholds,
    )

    generator = Faker()
    generator.seed_instance(seed)
    resolver = Resolver(
        dag=dag,
        name=name or f"{generator.unique.word()}_resolver",
        inputs=resolver_inputs,
        resolver_class=MockResolver,
        resolver_settings=resolver_settings,
        description=description,
    )
    assignments = _connected_components_from_edges(
        model_edges=model_edges,
        thresholds=thresholds,
    )

    source_entities = tuple(true_entities or ())
    entities = tuple(
        projected
        for entity in source_entities
        if (projected := entity.to_cluster_entity(*source_names)) is not None
    )

    return ResolverTestkit(
        resolver=resolver,
        assignments=assignments,
        entities=entities,
    )
