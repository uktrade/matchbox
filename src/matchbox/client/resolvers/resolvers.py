"""Resolver nodes and methodology registry for client-side execution."""

import json
from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl

from matchbox.client import _handler
from matchbox.client.models.models import Model
from matchbox.client.queries import Query
from matchbox.client.resolvers.base import ResolverMethod, ResolverSettings
from matchbox.client.resolvers.components import Components
from matchbox.common.arrow import SCHEMA_CLUSTERS, check_schema
from matchbox.common.dtos import (
    Resolution,
    ResolutionName,
    ResolutionType,
    ResolverConfig,
    ResolverResolutionName,
    ResolverResolutionPath,
    SourceResolutionName,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxResolutionTypeError,
)
from matchbox.common.hash import hash_arrow_table
from matchbox.common.logging import logger, profile_time

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.sources import Source
else:
    DAG = Any
    Source = Any

T = TypeVar("T")

_RESOLVER_CLASSES: dict[str, type[ResolverMethod]] = {}


def add_resolver_class(resolver_class: type[ResolverMethod]) -> None:
    """Register a resolver methodology class."""
    if not issubclass(resolver_class, ResolverMethod):
        raise ValueError("The argument is not a proper subclass of ResolverMethod.")
    _RESOLVER_CLASSES[resolver_class.__name__] = resolver_class
    logger.debug(f"Registered resolver class: {resolver_class.__name__}")


add_resolver_class(Components)


def post_run(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that a method is called after resolver run.

    Raises:
        RuntimeError: If run hasn't happened.
    """

    @wraps(method)
    def wrapper(self: "Resolver", *args: Any, **kwargs: Any) -> T:
        if self.results is None:
            raise RuntimeError(
                "The resolver must be run before attempting this operation."
            )
        return method(self, *args, **kwargs)

    return wrapper


class Resolver:
    """Client-side node that computes clusters from model and resolver inputs."""

    def __init__(
        self,
        dag: DAG,
        name: ResolverResolutionName,
        inputs: Iterable[Model],
        resolver_class: type[ResolverMethod] | str,
        resolver_settings: ResolverSettings | dict[str, Any],
        description: str | None = None,
    ) -> None:
        """Create a resolver node that computes clusters from its inputs."""
        self.dag = dag
        self.name = ResolverResolutionName(name)
        deduped_inputs: list[Model] = []
        seen_names: set[str] = set()
        for node in inputs:
            if not isinstance(node, Model):
                raise MatchboxResolutionTypeError(
                    resolution_name=getattr(node, "name", node),
                    expected_resolution_types=[ResolutionType.MODEL],
                )
            if node.name in seen_names:
                continue
            seen_names.add(node.name)
            deduped_inputs.append(node)
        self.inputs = tuple(deduped_inputs)
        self.description = description

        if len(self.inputs) < 1:
            raise ValueError("Resolver needs at least one input")

        if isinstance(resolver_class, str):
            self.resolver_class: type[ResolverMethod] = _RESOLVER_CLASSES[
                resolver_class
            ]
        else:
            self.resolver_class = resolver_class

        self.resolver_instance = self.resolver_class(settings=resolver_settings)

        if isinstance(resolver_settings, dict):
            SettingsClass = self.resolver_instance.__annotations__["settings"]
            self.resolver_settings = SettingsClass(**resolver_settings)
        else:
            self.resolver_settings = resolver_settings

        self.results: pl.DataFrame | None = None

    @property
    def config(self) -> ResolverConfig:
        """Generate config DTO from Resolver."""
        return ResolverConfig(
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

    @profile_time(attr="name")
    def compute_clusters(
        self, model_edges: Mapping[ResolutionName, pl.DataFrame]
    ) -> pl.DataFrame:
        """Delegate cluster computation to the configured resolver instance."""
        return self.resolver_instance.compute_clusters(model_edges=model_edges)

    @profile_time(attr="name")
    def run(self) -> pl.DataFrame:
        """Run the resolver and materialise cluster assignments."""
        model_edges: dict[ResolutionName, pl.DataFrame] = {}

        for node in self.inputs:
            if node.results is None:
                raise ValueError(
                    f"Resolver input '{node.name}' has no local results. "
                    "Run or download upstream nodes before running this resolver."
                )
            model_edges[node.name] = node.results

        clusters = self.compute_clusters(model_edges=model_edges)

        self.results = clusters
        return self.results

    @post_run
    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        upload_table = self.results.to_arrow()
        check_schema(expected=SCHEMA_CLUSTERS, actual=upload_table.schema)
        return Resolution(
            description=self.description,
            resolution_type=ResolutionType.RESOLVER,
            config=self.config,
            fingerprint=hash_arrow_table(upload_table),
        )

    @classmethod
    def from_resolution(
        cls,
        resolution: Resolution,
        resolution_name: str,
        dag: DAG,
    ) -> "Resolver":
        """Reconstruct from Resolution."""
        if resolution.resolution_type != ResolutionType.RESOLVER:
            raise ValueError("Resolution must be of type 'resolver'")

        return cls(
            dag=dag,
            name=ResolverResolutionName(resolution_name),
            description=resolution.description,
            inputs=[dag.nodes[name] for name in resolution.config.inputs],
            resolver_class=resolution.config.resolver_class,
            resolver_settings=json.loads(resolution.config.resolver_settings),
        )

    @post_run
    @profile_time(attr="name")
    def sync(self) -> None:
        """Send resolver config and assignments to the server."""
        log_prefix = f"Sync {self.name}"
        resolution = self.to_resolution()

        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
            logger.info("Found existing resolution", prefix=log_prefix)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None

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
            logger.info("Setting data for new resolution", prefix=log_prefix)
            _handler.set_data(
                path=self.resolution_path,
                data=self.results,
            )

        # Refresh local state from backend canonical resolver data.
        self.results = self.download_results()

    def query(self, *sources: Source, **kwargs: Any) -> Query:
        """Create a query rooted at this resolver."""
        if not sources:
            sources = tuple(self.dag.get_source(name) for name in sorted(self.sources))
        return Query(*sources, resolver=self, dag=self.dag, **kwargs)

    def download_results(self) -> pl.DataFrame:
        """Download resolver assignments directly from the resolution data API.

        These IDs will be inconsistent with those allocated locally.
        """
        table = _handler.get_resolver_data(path=self.resolution_path)
        check_schema(expected=SCHEMA_CLUSTERS, actual=table.schema)
        self.results = pl.from_arrow(table)
        return self.results

    def clear_data(self) -> None:
        """Drop local resolver data."""
        self.results = None

    def delete(self, certain: bool = False) -> bool:
        """Delete resolver and associated data from backend."""
        return _handler.delete_resolution(
            path=self.resolution_path,
            certain=certain,
        ).success
