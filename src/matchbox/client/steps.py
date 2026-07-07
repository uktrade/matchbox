"""Base class for client-side DAG step nodes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

import polars as pl
import pyarrow as pa

from matchbox.client import _handler
from matchbox.common.arrow import check_schema_subset
from matchbox.common.dtos import (
    SourceStepName,
    Step,
    StepName,
    StepPath,
)
from matchbox.common.exceptions import MatchboxStepNotFoundError
from matchbox.common.hash import hash_arrow_table
from matchbox.common.logging import logger
from matchbox.common.stats import DAGStats

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
else:
    DAG = Any

T = TypeVar("T")


class StepConfigProtocol(Protocol):
    """Minimal protocol required by client DAG step config DTOs."""

    @property
    def dependencies(self) -> list[StepName]:
        """Execution prerequisites required before running the step."""
        ...

    @property
    def parents(self) -> list[StepName]:
        """Direct DAG edges to this step."""
        ...

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialise the config for stable hashing."""
        ...


def post_run(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that a method is called after step run.

    Raises:
        RuntimeError: If the step hasn't been run yet.
    """

    @wraps(method)
    def wrapper(self: "StepABC", *args: Any, **kwargs: Any) -> T:
        if self._local_data is None:
            raise RuntimeError("The step must be run before attempting this operation.")
        return method(self, *args, **kwargs)

    return wrapper


class StepABC(ABC):
    """Base class for client-side DAG nodes that compute and sync data."""

    _local_data_schema: ClassVar[pa.Schema]

    def __init__(
        self,
        dag: DAG,
        name: str,
        description: str | None = None,
    ) -> None:
        """Initialise the step."""
        self.dag = dag
        self.name = name
        self.description = description
        self._local_data: pl.DataFrame | None = None
        self._stats: DAGStats = dag.stats

    # Local data access

    @property
    def local_data(self) -> pl.DataFrame | None:
        """The locally computed results for this step."""
        return self._local_data

    def clear_data(self) -> None:
        """Drop locally computed data."""
        self._local_data = None

    # Abstract interface

    @property
    @abstractmethod
    def path(self) -> StepPath:
        """The step path used to identify this step on the server."""
        ...

    @property
    @abstractmethod
    def sources(self) -> set[SourceStepName]:
        """Set of source names upstream of this node."""
        ...

    @property
    @abstractmethod
    def config(self) -> StepConfigProtocol:
        """Config DTO for this step."""
        ...

    @abstractmethod
    def to_dto(self) -> Step:
        """Convert to Step DTO for API calls."""
        ...

    @classmethod
    def from_dto(
        cls,
        step: Step,
        step_name: str,
        dag: DAG,
        **kwargs: Any,
    ) -> "StepABC":
        """Reconstruct from Step DTO. Subclasses should override this."""
        raise NotImplementedError(f"{cls.__name__} must implement from_dto.")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        """Execute the step, populate _local_data, and return it."""
        ...

    # Concrete shared behaviour

    @property
    def cache_path(self) -> Path:
        """Path within the DAG cache for storing this step's local data."""
        return self.dag.cache_path / f"{self.name}.parquet"

    def __hash__(self) -> int:
        """Return a hash of the step based on its config."""
        return hash(self.config.model_dump_json())

    def __eq__(self, other: object) -> bool:
        """Check equality of two step objects based on their config."""
        if type(other) is not type(self):
            return False
        return self.config == other.config

    @post_run
    def _fingerprint(self) -> bytes:
        """Compute a content hash of the local data for fingerprinting."""
        check_schema_subset(
            expected=self._local_data_schema, actual=self._local_data.to_arrow().schema
        )
        return hash_arrow_table(self._local_data.to_arrow())

    def delete(self, certain: bool = False) -> bool:
        """Delete this step and its associated data from the backend."""
        return _handler.delete_step(
            path=self.path,
            certain=certain,
        ).success

    def download(self) -> pl.DataFrame:
        """Fetch remote data for this step and store it locally."""
        table = _handler.get_data(path=self.path)
        check_schema_subset(expected=self._local_data_schema, actual=table.schema)
        self._local_data = pl.from_arrow(table)
        return self._local_data

    @post_run
    @DAGStats.time("sync")
    def sync(self) -> None:
        """Send step config and local data to the server.

        Not resistant to race conditions: only one client should call sync at a time.
        """
        log_prefix = f"Sync {self.name}"
        step = self.to_dto()

        try:
            existing_step = _handler.get_step(path=self.path)
            logger.info("Found existing step", prefix=log_prefix)
        except MatchboxStepNotFoundError:
            existing_step = None

        if existing_step:
            if (existing_step.fingerprint == step.fingerprint) and (
                existing_step.config.parents == step.config.parents
            ):
                logger.info("Updating existing step", prefix=log_prefix)
                _handler.update_step(
                    step=step,
                    path=self.path,
                )
            else:
                logger.info(
                    "Update not possible. Deleting existing step",
                    prefix=log_prefix,
                )
                _handler.delete_step(path=self.path, certain=True)
                existing_step = None

        if not existing_step:
            logger.info("Creating new step", prefix=log_prefix)
            _handler.create_step(step=step, path=self.path)
            logger.info("Setting data for new step", prefix=log_prefix)
            _handler.set_data(path=self.path, data=self._local_data)
