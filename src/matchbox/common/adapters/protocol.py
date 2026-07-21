"""The cluster store protocol shared by every Matchbox backend."""

import json
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Protocol, TypeAlias

from pyarrow import Table
from pydantic import BaseModel, field_validator

from matchbox.common.dtos import (
    Match,
    ModelStepPath,
    ResolverStepPath,
    SourceStepPath,
    Step,
    StepPath,
)


class Countable(Protocol):
    """A protocol for objects that can be counted."""

    def count(self) -> int:
        """Counts the number of items in the object."""
        ...


class Listable(Protocol):
    """A protocol for objects that can be listed."""

    def list_all(self) -> list[str]:
        """Lists the items in the object."""
        ...


class ListableAndCountable(Countable, Listable):
    """A protocol for objects that can be counted and listed."""

    pass


class MatchboxServerBackends(StrEnum):
    """The available server backends for Matchbox."""

    POSTGRES = "postgres"


class MatchboxLocalBackends(StrEnum):
    """The available local (client-side) backends for Matchbox."""

    DUCKDB = "duckdb"


MatchboxBackends: TypeAlias = MatchboxServerBackends | MatchboxLocalBackends
"""Any backend, server or local, that can produce a MatchboxSnapshot."""


class MatchboxSnapshot(BaseModel):
    """A snapshot of the Matchbox database."""

    backend_type: MatchboxBackends
    data: dict[str, Any]

    @field_validator("data")
    @classmethod
    def check_serialisable(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate that the value can be serialised to JSON."""
        try:
            json.dumps(value)
            return value
        except (TypeError, OverflowError) as e:
            raise ValueError(f"Value is not JSON serialisable: {e}") from e


class MatchboxClusterStoreAdapter(ABC):
    """The contract shared by anything that stores and serves cluster data.

    Implemented by MatchboxDBAdapter, and so by the PostgreSQL backend.

    TODO: a local client-side store, backed by an embedded database, will
    implement this too.
    """

    # Counts
    source_clusters: Countable
    model_clusters: Countable
    all_clusters: Countable
    creates: Countable  # clusters proposed by resolver
    merges: Countable  # cluster lineage tree
    proposes: Countable  # raw model edge scores

    # Query block

    @abstractmethod
    def query(
        self,
        source: SourceStepPath,
        resolver: ResolverStepPath | None = None,
        return_leaf_id: bool = False,
        limit: int | None = None,
    ) -> Table:
        """Queries the database from an optional resolution.

        Args:
            source: The step path identifying the source to query.
            resolver (optional): The resolver path to use for filtering results.
                If not specified, the source step is used for the queried source.
            return_leaf_id (optional): whether to return cluster ID of leaves
            limit (optional): the number to use in a limit clause. Useful for testing

        Returns:
            The resulting matchbox IDs in Arrow format
        """
        ...

    @abstractmethod
    def match(
        self,
        key: str,
        source: SourceStepPath,
        targets: list[SourceStepPath],
        resolver: ResolverStepPath,
    ) -> list[Match]:
        """Match an ID in a source step and return the keys in the targets.

        Args:
            key: The key to match from the source.
            source: The path of the source step.
            targets: The paths of the target source steps.
            resolver: The resolver path to use for matching.
        """
        ...

    # Data block

    @abstractmethod
    def create_step(self, step: Step, path: StepPath) -> None:
        """Write a step to Matchbox.

        Args:
            step: Step object with a source, model, or resolver config
            path: The step path
        """
        ...

    @abstractmethod
    def insert_source_data(self, path: SourceStepPath, data_hashes: Table) -> None:
        """Insert hash data for a source step.

        Only possible if data fingerprint matches fingerprint declared when the
        step was created. Data can only be set once on a step.

        Args:
            path: The path of the source step to index.
            data_hashes: The Arrow table with the hash of each data row
        """
        ...

    @abstractmethod
    def insert_model_data(self, path: ModelStepPath, results: Table) -> None:
        """Insert results data for a model step.

        Only possible if data fingerprint matches fingerprint declared when the
        step was created. Data can only be set once on a step.
        """
        ...

    @abstractmethod
    def insert_resolver_data(self, path: ResolverStepPath, data: Table) -> None:
        """Insert resolver cluster assignments for a resolver step."""
        ...

    @abstractmethod
    def get_model_data(self, path: ModelStepPath) -> Table:
        """Get the results for a model step."""
        ...

    @abstractmethod
    def get_resolver_data(self, path: ResolverStepPath) -> Table:
        """Get cluster assignments for a resolver step."""
        ...

    # Data management

    @abstractmethod
    def dump(self) -> MatchboxSnapshot:
        """Dumps the entire database to a snapshot.

        Returns:
            A MatchboxSnapshot with the database's current state, tagged
                with the backend type that produced it.
        """
        ...

    @abstractmethod
    def restore(self, snapshot: MatchboxSnapshot) -> None:
        """Restores the database from a snapshot.

        Args:
            snapshot: A MatchboxSnapshot with the database's state

        Raises:
            TypeError: If the snapshot's backend type doesn't match this one
        """
        ...

    @abstractmethod
    def clear(self, certain: bool) -> None:
        """Soft clear the database by deleting all rows but retaining tables.

        Args:
            certain: Whether to delete the database without confirmation.
        """
        ...

    @abstractmethod
    def drop(self, certain: bool) -> None:
        """Hard clear the database by dropping all tables and re-creating.

        Args:
            certain: Whether to drop the database without confirmation.
        """
        ...
