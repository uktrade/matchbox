"""Factory helpers for resolver testkits."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import BaseModel, ConfigDict

from matchbox.client.dags import DAG
from matchbox.client.models import Model
from matchbox.client.resolvers import Resolver
from matchbox.common.dtos import ResolverResolutionName
from matchbox.common.resolvers import (
    Components,
    ComponentsSettings,
    ResolverMethod,
    ResolverSettings,
)

if TYPE_CHECKING:
    from matchbox.server.base import MatchboxDBAdapter


def resolver_name_for_model(model_name: str) -> str:
    """Build canonical resolver name for a model."""
    return f"resolver_{model_name}"


class ResolverTestkit(BaseModel):
    """A testkit wrapper around a resolver node."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolver: Resolver

    @property
    def name(self) -> str:
        """Return resolver name."""
        return self.resolver.name

    def materialise(self, backend: "MatchboxDBAdapter") -> "ResolverTestkit":
        """Run resolver, upload assignments, and hydrate backend IDs."""
        self.resolver.run()
        backend.create_resolution(
            resolution=self.resolver.to_resolution(),
            path=self.resolver.resolution_path,
        )

        if self.resolver._upload_results is None:  # noqa: SLF001
            raise RuntimeError("Resolver upload payload missing after run().")

        mapping = pl.from_arrow(
            backend.insert_resolver_data(
                path=self.resolver.resolution_path,
                data=self.resolver._upload_results.to_arrow(),  # noqa: SLF001
            )
        ).cast(
            {
                "client_cluster_id": pl.UInt64,
                "cluster_id": pl.UInt64,
            }
        )
        self.resolver.results = (
            self.resolver._upload_results.join(  # noqa: SLF001
                mapping,
                on="client_cluster_id",
                how="left",
            )
            .drop("client_cluster_id")
            .select("cluster_id", "node_id")
        )
        if self.resolver.results["cluster_id"].null_count() > 0:
            raise RuntimeError(
                "Resolver upload mapping was incomplete in resolver materialisation."
            )
        self.resolver.results = self.resolver._with_root_membership(  # noqa: SLF001
            self.resolver.results
        )
        return self


def resolver_factory(
    *,
    dag: DAG,
    name: ResolverResolutionName,
    inputs: Iterable[Model | Resolver],
    resolver_class: type[ResolverMethod] | str = Components,
    resolver_settings: ResolverSettings | dict[str, Any] | None = None,
    thresholds: dict[str, int] | None = None,
    description: str | None = None,
) -> ResolverTestkit:
    """Create a resolver testkit with sensible defaults."""
    unique_inputs: list[Model | Resolver] = []
    seen: set[str] = set()
    for node in inputs:
        if node.name in seen:
            continue
        seen.add(node.name)
        unique_inputs.append(node)

    if resolver_settings is None:
        is_components = resolver_class in {Components, "Components"}
        if not is_components:
            raise ValueError(
                "resolver_settings must be provided for non-Components resolvers."
            )
        effective_thresholds = thresholds or {node.name: 0 for node in unique_inputs}
        resolver_settings = ComponentsSettings(thresholds=effective_thresholds)
    elif thresholds is not None:
        raise ValueError("Cannot set both resolver_settings and thresholds.")

    resolver = dag.resolver(
        name=name,
        inputs=unique_inputs,
        resolver_class=resolver_class,
        resolver_settings=resolver_settings,
        description=description,
    )
    return ResolverTestkit(resolver=resolver)
