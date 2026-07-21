"""Base classes and utilities for Matchbox local (client-side) database adapters."""

from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Protocol

from pyarrow import Table
from pydantic import BaseModel

from matchbox.common.adapters.protocol import (
    MatchboxClusterStoreAdapter,
    MatchboxLocalBackends,
)
from matchbox.common.dtos import SourceStepPath, StepName, StepPath


class HasGraph(Protocol):
    """Structural type for anything exposing step topology as `graph`."""

    graph: dict[StepName, list[StepName]]


def _ancestor_depths(
    graph: dict[StepName, list[StepName]], start: StepName
) -> dict[StepName, int]:
    """BFS over parent edges from `start`.

    Shortest depth per ancestor, self excluded.
    """
    depths: dict[StepName, int] = {}
    queue: deque[tuple[StepName, int]] = deque(
        (parent, 1) for parent in graph.get(start, [])
    )
    while queue:
        name, depth = queue.popleft()
        if name in depths and depths[name] <= depth:
            continue
        depths[name] = depth
        queue.extend((parent, depth + 1) for parent in graph.get(name, []))
    return depths


def compute_lineage(
    graph: dict[StepName, list[StepName]],
    resolver: StepName,
    sources: list[StepName] | None = None,
) -> list[StepName]:
    """Ordered ancestor step names for `resolver`, closest first, self first.

    When `sources` is given, ancestors are restricted to those on a path
    to one of `sources`: kept if it is a source, or is itself downstream
    of one.
    """
    ancestors = _ancestor_depths(graph, resolver)

    if sources:
        children: dict[StepName, list[StepName]] = {}
        for name, parents in graph.items():
            for parent in parents:
                children.setdefault(parent, []).append(name)

        allowed = set(sources)
        for source in sources:
            allowed.update(_ancestor_depths(children, source))
        ancestors = {
            name: depth for name, depth in ancestors.items() if name in allowed
        }

    ordered = sorted(ancestors.items(), key=lambda pair: (pair[1], pair[0]))
    return [resolver, *(name for name, _ in ordered)]


def compute_descendants(
    graph: dict[StepName, list[StepName]], step: StepName
) -> list[StepName]:
    """All step names downstream of `step` - inverts the parent graph, then BFS."""
    children: dict[StepName, list[StepName]] = {}
    for name, parents in graph.items():
        for parent in parents:
            children.setdefault(parent, []).append(name)
    return list(_ancestor_depths(children, step))


class MatchboxLocalSettings(BaseModel):
    """Settings for a local (client-side) Matchbox backend."""

    backend_type: MatchboxLocalBackends = MatchboxLocalBackends.DUCKDB
    path: Path | None = None
    """Path to the local database file. None means in-memory and ephemeral."""


class MatchboxLocalDBAdapter(MatchboxClusterStoreAdapter, ABC):
    """An abstract base class for Matchbox local (client-side) database adapters.

    Extends MatchboxClusterStoreAdapter (the query block, data block, and
    cluster counts) with the local-only surface: raw warehouse data, a
    disposable query cache, and explicit cascade invalidation.

    Local stores replace-on-rerun (with cascade invalidation of descendant
    steps), unlike server backends, which are write-once per step per run.

    The adapter is bound to a live DAG via bind(). lineage() and
    descendants() read its graph fresh on every call, so steps added
    later are picked up automatically.
    """

    _graph_source: HasGraph | None = None

    def bind(self, graph_source: HasGraph) -> None:
        """Bind a live source of step topology.

        Stored by reference, not copied: lineage()/descendants() read
        graph_source.graph fresh on every call, so later mutations are
        picked up automatically.
        """
        self._graph_source = graph_source

    def _graph(self) -> dict[StepName, list[StepName]]:
        if self._graph_source is None:
            raise RuntimeError(
                "Adapter has no bound graph source. Call bind() before "
                "lineage() or descendants()."
            )
        return self._graph_source.graph

    def lineage(
        self, resolver: StepName, sources: list[StepName] | None = None
    ) -> list[StepName]:
        """Ordered ancestor step names, highest priority first, self first.

        Restricted to paths that lead to `sources`, when given. Walks the
        bound graph fresh on every call.
        """
        return compute_lineage(self._graph(), resolver, sources)

    def descendants(self, step: StepName) -> list[StepName]:
        """All step names downstream of `step` (for cascade invalidation)."""
        return compute_descendants(self._graph(), step)

    @abstractmethod
    def insert_raw_data(self, path: SourceStepPath, table: Table) -> None:
        """Insert raw warehouse rows for a source step. Canonical local state.

        Replaces any existing raw data for this step, and cascades:
        descendant steps' data is dropped, so stale downstream results can
        never be served.
        """
        ...

    @abstractmethod
    def get_raw_data(
        self, path: SourceStepPath, keys: list[str] | None = None
    ) -> Table:
        """Get raw warehouse rows for a source step, optionally filtered by key."""
        ...

    @abstractmethod
    def cache_query(self, key: str, table: Table, depends_on: list[StepPath]) -> None:
        """Cache cleaned query data, keyed by query config plus upstream fingerprints.

        Disposable: unlike insert_raw_data, this is just a cache, and can be
        recomputed at any time.

        Args:
            key: Opaque identity of the query definition this result was
                built from.
            table: The data to cache.
            depends_on: The steps (sources and resolver) this result
                depends on, so cascade invalidation can drop it when any
                of them change.
        """
        ...

    @abstractmethod
    def get_cached_query(self, key: str) -> Table | None:
        """Get cached cleaned query data, or None if not cached."""
        ...

    @abstractmethod
    def drop_step_data(self, path: StepPath) -> None:
        """Explicitly invalidate a step's data. Cascades to descendant steps."""
        ...


def get_local_backend_class(
    backend_type: MatchboxLocalBackends,
) -> type[MatchboxLocalDBAdapter]:
    """Get the appropriate local backend class based on the backend type."""
    if backend_type == MatchboxLocalBackends.DUCKDB:
        from matchbox.client.adapters.duckdb import (  # noqa: PLC0415
            MatchboxLocalDuckDB,
        )

        return MatchboxLocalDuckDB
    else:
        raise ValueError(f"Unsupported local backend type: {backend_type}")


def settings_to_local_backend(
    settings: MatchboxLocalSettings,
) -> MatchboxLocalDBAdapter:
    """Create a local backend adapter with injected settings."""
    BackendClass = get_local_backend_class(settings.backend_type)
    return BackendClass(path=settings.path)
