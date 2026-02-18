"""Resolver nodes that materialise clusters from model and resolver inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import polars as pl

from matchbox.client import _handler
from matchbox.client.models.models import Model
from matchbox.client.queries import Query
from matchbox.common.arrow import SCHEMA_RESOLVER_UPLOAD
from matchbox.common.dtos import (
    Resolution,
    ResolutionName,
    ResolutionType,
    ResolverConfig,
    ResolverResolutionName,
    ResolverResolutionPath,
    ResolverType,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.hash import hash_arrow_table
from matchbox.common.logging import logger, profile_time
from matchbox.common.resolvers import (
    Components,
    ComponentsSettings,
    ResolverMethod,
    ResolverSettings,
    get_resolver_class,
    run_resolver_method,
)

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.sources import Source
else:
    DAG = Any
    Source = Any


class Resolver:
    """Client-side node that computes clusters from model/resolver inputs."""

    def __init__(
        self,
        dag: DAG,
        name: ResolverResolutionName,
        inputs: Iterable[Model | Resolver],
        resolver_class: type[ResolverMethod] | str,
        resolver_settings: ResolverSettings | dict,
        description: str | None = None,
    ) -> None:
        """Create a resolver node that computes clusters from its inputs."""
        self.dag = dag
        self.name = ResolverResolutionName(name)
        deduped_inputs: list[Model | Resolver] = []
        seen_names: set[str] = set()
        for node in inputs:
            if node.name in seen_names:
                continue
            seen_names.add(node.name)
            deduped_inputs.append(node)
        self.inputs = tuple(deduped_inputs)
        self.description = description

        if len(self.inputs) < 1:
            raise ValueError("Resolver needs at least one input")

        if isinstance(resolver_class, str):
            self.resolver_class = get_resolver_class(resolver_class)
        else:
            self.resolver_class = resolver_class

        self.resolver_instance = self.resolver_class(settings=resolver_settings)
        self.resolver_type = self._infer_resolver_type(self.resolver_class)

        if isinstance(resolver_settings, dict):
            SettingsClass = self.resolver_instance.__annotations__["settings"]
            self.resolver_settings = SettingsClass(**resolver_settings)
        else:
            self.resolver_settings = resolver_settings

        self.resolver_instance.settings = self.resolver_settings
        self._normalise_components_settings()

        self.results: pl.DataFrame | None = None
        self._upload_results: pl.DataFrame | None = None

    @staticmethod
    def _infer_resolver_type(
        resolver_class: type[ResolverMethod],
    ) -> ResolverType:
        """Infer resolver type from resolver class metadata."""
        resolver_type = getattr(resolver_class, "resolver_type", None)
        if resolver_type is None:
            if issubclass(resolver_class, Components):
                return ResolverType.COMPONENTS
            raise ValueError(
                f"Resolver class '{resolver_class.__name__}' must define resolver_type"
            )
        return ResolverType(resolver_type)

    def _normalise_components_settings(self) -> None:
        """Ensure Components settings are aligned with configured inputs."""
        if self.resolver_type != ResolverType.COMPONENTS:
            return

        if not isinstance(self.resolver_settings, ComponentsSettings):
            self.resolver_settings = ComponentsSettings.model_validate(
                self.resolver_settings.model_dump(mode="json")
            )

        input_names = tuple(node.name for node in self.inputs)
        threshold_input = dict(self.resolver_settings.thresholds)

        extra_thresholds = [name for name in threshold_input if name not in input_names]
        if extra_thresholds:
            raise ValueError(
                "Thresholds were provided for unknown resolver inputs: "
                f"{extra_thresholds}"
            )

        for node_name in input_names:
            threshold_input.setdefault(node_name, 0)

        self.resolver_settings = ComponentsSettings(thresholds=threshold_input)
        self.resolver_instance.settings = self.resolver_settings

    @property
    def config(self) -> ResolverConfig:
        """Generate config DTO from Resolver."""
        return ResolverConfig(
            type=self.resolver_type,
            resolver_class=self.resolver_class.__name__,
            resolver_settings=self.resolver_settings.model_dump_json(),
            inputs=tuple(node.name for node in self.inputs),
        )

    @property
    def sources(self) -> set[SourceResolutionName]:
        """Set of source names upstream of this node."""
        upstream: set[SourceResolutionName] = set()
        for node in self.inputs:
            upstream.update(node.sources)
        return upstream

    @property
    def resolution_path(self) -> ResolverResolutionPath:
        """Return resolver path."""
        return ResolverResolutionPath(
            collection=self.dag.name,
            run=self.dag.run,
            name=self.name,
        )

    def _get_model_edges(self, model: Model) -> pl.DataFrame:
        """Retrieve model edges either from memory or backend."""
        if model.results is not None:
            return model.results.probabilities
        return model.download_results().probabilities

    def _get_resolver_assignments(self, resolver: Resolver) -> pl.DataFrame:
        """Retrieve resolver assignments either from memory or backend."""
        if resolver.results is not None:
            return resolver.results.select("cluster_id", "node_id")
        return resolver.download_results().select("cluster_id", "node_id")

    @staticmethod
    def _with_root_membership(assignments: pl.DataFrame) -> pl.DataFrame:
        """Ensure each cluster includes itself as a node."""
        if assignments.height == 0:
            return assignments
        roots = assignments.select(
            pl.col("cluster_id"),
            pl.col("cluster_id").alias("node_id"),
        )
        return (
            pl.concat([assignments, roots], how="vertical")
            .unique()
            .sort("cluster_id", "node_id")
        )

    @profile_time(attr="name")
    def compute_clusters(
        self,
        model_edges: Mapping[ResolutionName, pl.DataFrame],
        resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
    ) -> pl.DataFrame:
        """Delegate cluster computation to resolver methodology instance."""
        return run_resolver_method(
            resolver_class=self.resolver_class,
            settings_payload=self.resolver_settings,
            model_edges=model_edges,
            resolver_assignments=resolver_assignments,
        )

    @profile_time(attr="name")
    def run(self) -> pl.DataFrame:
        """Run the resolver and materialise cluster assignments."""
        model_edges: dict[ResolutionName, pl.DataFrame] = {}
        resolver_assignments: dict[ResolutionName, pl.DataFrame] = {}

        for node in self.inputs:
            if isinstance(node, Model):
                model_edges[node.name] = self._get_model_edges(node)
            else:
                resolver_assignments[node.name] = self._get_resolver_assignments(node)

        clusters = self.compute_clusters(
            model_edges=model_edges,
            resolver_assignments=resolver_assignments,
        )

        upload_results = (
            clusters.rename({"cluster_id": "client_cluster_id"})
            .select("client_cluster_id", "node_id")
            .cast({"client_cluster_id": pl.UInt64, "node_id": pl.UInt64})
        )

        if upload_results.height == 0:
            upload_results = pl.DataFrame(schema=pl.Schema(SCHEMA_RESOLVER_UPLOAD))

        self._upload_results = upload_results
        self.results = self._with_root_membership(
            upload_results.rename({"client_cluster_id": "cluster_id"})
        )
        return self.results

    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        if self._upload_results is None:
            raise RuntimeError("Resolver must be run before converting to a resolution")

        upload_table = self._upload_results.to_arrow().cast(SCHEMA_RESOLVER_UPLOAD)
        return Resolution(
            description=self.description,
            resolution_type=ResolutionType.RESOLVER,
            config=self.config,
            fingerprint=hash_arrow_table(upload_table),
        )

    @profile_time(attr="name")
    def sync(self) -> None:
        """Send resolver config and assignments to the server."""
        resolution = self.to_resolution()
        log_prefix = f"Sync {self.name}"

        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
            logger.info("Found existing resolution", prefix=log_prefix)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None

        should_upload = False

        if existing_resolution:
            if (existing_resolution.fingerprint == resolution.fingerprint) and (
                existing_resolution.config.parents == resolution.config.parents
            ):
                logger.info("Updating existing resolution", prefix=log_prefix)
                _handler.update_resolution(
                    resolution=resolution,
                    path=self.resolution_path,
                )
            else:
                logger.info(
                    "Update not possible. Deleting existing resolution",
                    prefix=log_prefix,
                )
                _handler.delete_resolution(path=self.resolution_path, certain=True)
                existing_resolution = None

        if not existing_resolution:
            logger.info("Creating new resolution", prefix=log_prefix)
            _handler.create_resolution(resolution=resolution, path=self.resolution_path)
            should_upload = True

        if should_upload:
            if self._upload_results is None:
                raise RuntimeError("Resolver must be run before sync")

            upload_id = _handler.set_data(
                path=self.resolution_path,
                data=self._upload_results,
            )
            mapping = pl.from_arrow(
                _handler.get_resolver_mapping(
                    path=self.resolution_path,
                    upload_id=upload_id,
                )
            ).cast(
                {
                    "client_cluster_id": pl.UInt64,
                    "cluster_id": pl.UInt64,
                }
            )
            mapped_results = (
                self._upload_results.join(
                    mapping,
                    on="client_cluster_id",
                    how="left",
                )
                .drop("client_cluster_id")
                .select("cluster_id", "node_id")
            )
            self.results = mapped_results
            if self.results["cluster_id"].null_count() > 0:
                raise RuntimeError(
                    "Resolver upload mapping was incomplete; "
                    "some clusters were not mapped."
                )
            self.results = self._with_root_membership(self.results)
        else:
            # Ensure downstream nodes see backend IDs rather than local IDs.
            self.results = self.download_results()

    def query(self, *sources: Source, **kwargs: Any) -> Query:
        """Create a query rooted at this resolver."""
        if not sources:
            sources = tuple(self.dag.get_source(name) for name in sorted(self.sources))
        return Query(*sources, resolver=self, dag=self.dag, **kwargs)

    def download_results(self) -> pl.DataFrame:
        """Download resolver assignments directly from the resolution data API."""
        self.results = pl.from_arrow(
            _handler.get_resolver_data(path=self.resolution_path)
        ).cast(
            {
                "cluster_id": pl.UInt64,
                "node_id": pl.UInt64,
            }
        )
        return self.results

    def clear_data(self) -> None:
        """Drop local resolver data."""
        self.results = None
        self._upload_results = None

    def delete(self, certain: bool = False) -> bool:
        """Delete resolver and associated data from backend."""
        result = _handler.delete_resolution(path=self.resolution_path, certain=certain)
        return result.success
