"""Simplified TestkitDAG that's just a registry of test data."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.resolvers import Resolver
from matchbox.common.dtos import (
    ModelResolutionName,
    ResolutionName,
    ResolverResolutionName,
    SourceResolutionName,
)
from matchbox.common.factories.models import ModelTestkit
from matchbox.common.factories.sources import LinkedSourcesTestkit, SourceTestkit
from matchbox.common.resolvers import (
    Components,
    ComponentsSettings,
    ResolverMethod,
    ResolverSettings,
)

if TYPE_CHECKING:
    from matchbox.server.base import MatchboxDBAdapter


def _default_dag() -> DAG:
    """Create a default empty DAG."""
    dag = DAG(name="collection")
    dag.run = 1
    return dag


def add_components_resolver(
    dag: DAG,
    *,
    name: ResolverResolutionName,
    inputs: Iterable[Model | Resolver],
    thresholds: dict[str, int] | None = None,
    description: str | None = None,
) -> Resolver:
    """Create a Components resolver with canonical threshold defaults."""
    unique_inputs: list[Model | Resolver] = []
    seen: set[str] = set()
    for node in inputs:
        if node.name in seen:
            continue
        seen.add(node.name)
        unique_inputs.append(node)

    if thresholds is None:
        thresholds = {node.name: 0 for node in unique_inputs}

    return dag.resolver(
        name=name,
        inputs=unique_inputs,
        resolver_class=Components,
        resolver_settings=ComponentsSettings(thresholds=thresholds),
        description=description,
    )


class TestkitDAG(BaseModel):
    """DAG test wrapper that's just a registry of test data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The real DAG that handles all logic
    dag: DAG = Field(default_factory=_default_dag)

    # Just registries of test data - no complex logic
    sources: dict[SourceResolutionName, SourceTestkit] = {}
    models: dict[ModelResolutionName, ModelTestkit] = {}
    resolvers: dict[ResolverResolutionName, Resolver] = {}
    linked: dict[str, LinkedSourcesTestkit] = {}
    source_to_linked: dict[str, LinkedSourcesTestkit] = {}

    def add_linked_sources(self, testkit: LinkedSourcesTestkit) -> None:
        """Add system of linked sources to the real DAG and register test data."""
        linked_key = f"linked_{'_'.join(sorted(testkit.sources.keys()))}"
        self.linked[linked_key] = testkit

        for source_testkit in testkit.sources.values():
            self.source_to_linked[source_testkit.name] = testkit
            self.add_source(source_testkit)

    def add_source(self, testkit: SourceTestkit) -> None:
        """Add source to the real DAG and register test data."""
        self.dag._add_step(testkit.source)
        self.sources[testkit.name] = testkit

    def add_model(self, testkit: ModelTestkit) -> None:
        """Add model to the real DAG and register test data."""
        self.dag._add_step(testkit.model)
        self.models[testkit.name] = testkit

    def add_resolver(self, resolver: Resolver) -> None:
        """Add resolver to the real DAG and register it."""
        self.dag._add_step(resolver)
        self.resolvers[resolver.name] = resolver

    @staticmethod
    def resolver_name_for_model(model_name: str) -> str:
        """Build the canonical resolver name for a model."""
        return f"resolver_{model_name}"

    def materialise_resolver(
        self,
        backend: "MatchboxDBAdapter",
        *,
        name: ResolverResolutionName,
        inputs: Iterable[Model | Resolver | ResolutionName],
        resolver_class: type[ResolverMethod] | str,
        resolver_settings: ResolverSettings | dict[str, Any],
        description: str | None = None,
    ) -> Resolver:
        """Create, run, upload, and register a resolver in one step."""
        resolver = self.dag.resolver(
            name=name,
            inputs=list(inputs),
            resolver_class=resolver_class,
            resolver_settings=resolver_settings,
            description=description,
        )
        resolver.run()
        backend.create_resolution(
            resolution=resolver.to_resolution(),
            path=resolver.resolution_path,
        )

        if resolver._upload_results is None:  # noqa: SLF001
            raise RuntimeError("Resolver upload payload missing after run().")

        mapping = pl.from_arrow(
            backend.insert_resolver_data(
                path=resolver.resolution_path,
                data=resolver._upload_results.to_arrow(),  # noqa: SLF001
            )
        ).cast(
            {
                "client_cluster_id": pl.UInt64,
                "cluster_id": pl.UInt64,
            }
        )
        resolver.results = (
            resolver._upload_results.join(  # noqa: SLF001
                mapping,
                on="client_cluster_id",
                how="left",
            )
            .drop("client_cluster_id")
            .select("cluster_id", "node_id")
        )
        if resolver.results["cluster_id"].null_count() > 0:
            raise RuntimeError(
                "Resolver upload mapping was incomplete in TestkitDAG materialisation."
            )
        resolver.results = resolver._with_root_membership(resolver.results)  # noqa: SLF001

        self.add_resolver(resolver)
        return resolver

    def materialise_model_resolver(
        self,
        backend: "MatchboxDBAdapter",
        *,
        model_name: ModelResolutionName,
        threshold: int = 0,
        resolver_name: ResolverResolutionName | None = None,
    ) -> Resolver:
        """Materialise canonical Components resolver for one model testkit."""
        model = self.models[model_name].model
        inputs: list[Model | Resolver] = [model]
        thresholds: dict[str, int] = {model.name: threshold}

        for query in (model.left_query, model.right_query):
            if query and query.resolver:
                inputs.append(query.resolver)
                thresholds[query.resolver.name] = 0

        # Preserve order while deduplicating by name.
        unique_inputs: list[Model | Resolver] = []
        seen: set[str] = set()
        for node in inputs:
            if node.name in seen:
                continue
            seen.add(node.name)
            unique_inputs.append(node)

        return self.materialise_resolver(
            backend=backend,
            name=resolver_name or self.resolver_name_for_model(model_name),
            inputs=unique_inputs,
            resolver_class=Components,
            resolver_settings=ComponentsSettings(thresholds=thresholds),
        )
