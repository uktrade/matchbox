"""Resolver nodes that fuse model edges into materialised clusters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import polars as pl

from matchbox.client import _handler
from matchbox.client.models.models import Model
from matchbox.client.queries import Query
from matchbox.common.arrow import SCHEMA_RESOLVER_UPLOAD
from matchbox.common.dtos import (
    FusionStrategy,
    Resolution,
    ResolutionName,
    ResolutionType,
    ResolverConfig,
    ResolverResolutionName,
    ResolverResolutionPath,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.fusion import fuse_components
from matchbox.common.hash import hash_arrow_table
from matchbox.common.logging import logger, profile_time
from matchbox.common.transform import threshold_float_to_int

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.sources import Source
else:
    DAG = Any
    Source = Any


class Resolver:
    """Client-side node that fuses multiple model/resolver inputs."""

    def __init__(
        self,
        dag: DAG,
        name: ResolverResolutionName,
        inputs: Iterable[Model | Resolver],
        thresholds: dict[ResolutionName, int | float] | None = None,
        strategy: FusionStrategy | str = FusionStrategy.UNION,
        description: str | None = None,
    ) -> None:
        """Create a resolver node that fuses model and resolver inputs."""
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
        self.strategy = FusionStrategy(strategy)

        if len(self.inputs) < 1:
            raise ValueError("Resolver needs at least one input")

        input_names = tuple(node.name for node in self.inputs)
        threshold_input = thresholds or {}

        extra_thresholds = [name for name in threshold_input if name not in input_names]
        if extra_thresholds:
            raise ValueError(
                "Thresholds were provided for unknown resolver inputs: "
                f"{extra_thresholds}"
            )

        self.thresholds: dict[ResolutionName, int] = {}
        for node_name in input_names:
            threshold = threshold_input.get(node_name, 0)
            self.thresholds[node_name] = self._normalise_threshold(threshold)

        self.results: pl.DataFrame | None = None
        self._upload_results: pl.DataFrame | None = None

    @staticmethod
    def _normalise_threshold(value: int | float) -> int:
        """Normalise threshold input to integer percent."""
        if isinstance(value, float):
            return threshold_float_to_int(value)
        if isinstance(value, int) and 0 <= value <= 100:
            return value
        raise ValueError("Thresholds must be floats in [0,1] or ints in [0,100]")

    @property
    def config(self) -> ResolverConfig:
        """Generate config DTO from Resolver."""
        return ResolverConfig(
            inputs=tuple(node.name for node in self.inputs),
            thresholds=self.thresholds,
            strategy=self.strategy,
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
    def run(self) -> pl.DataFrame:
        """Run the resolver by fusing all configured inputs with UNION."""
        model_edges: list[pl.DataFrame] = []
        resolver_assignments: list[pl.DataFrame] = []

        for node in self.inputs:
            threshold = self.thresholds[node.name]

            if isinstance(node, Model):
                model_edges.append(
                    self._get_model_edges(node).filter(
                        pl.col("probability") >= threshold
                    )
                )
            else:
                resolver_assignments.append(self._get_resolver_assignments(node))

        fused = fuse_components(
            strategy=self.strategy,
            model_edges=model_edges,
            resolver_assignments=resolver_assignments,
        )

        upload_results = (
            fused.rename({"cluster_id": "client_cluster_id"})
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
        """Reconstruct resolver assignments from backend query API."""
        rows: list[pl.DataFrame] = []
        for source_name in sorted(self.sources):
            source = self.dag.get_source(source_name)
            result = pl.from_arrow(
                _handler.query(
                    source=source.resolution_path,
                    resolution=self.resolution_path,
                    return_leaf_id=True,
                )
            ).select(
                pl.col("id").cast(pl.UInt64).alias("cluster_id"),
                pl.col("leaf_id").cast(pl.UInt64).alias("node_id"),
            )
            rows.append(result)

        if rows:
            assignments = pl.concat(rows, how="vertical").unique()
        else:
            assignments = pl.DataFrame(
                schema={"cluster_id": pl.UInt64, "node_id": pl.UInt64}
            )

        self.results = self._with_root_membership(assignments)
        return self.results

    def clear_data(self) -> None:
        """Drop local resolver data."""
        self.results = None
        self._upload_results = None

    def delete(self, certain: bool = False) -> bool:
        """Delete resolver and associated data from backend."""
        result = _handler.delete_resolution(path=self.resolution_path, certain=certain)
        return result.success
