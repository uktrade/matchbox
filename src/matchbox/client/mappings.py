"""Interface to external record mapping."""

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from pyarrow import Table as ArrowTable

from matchbox.common.dtos import ResolutionPath
from matchbox.common.logging import profile_time

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.locations import Location
else:
    DAG = Any
    Location = Any

T = TypeVar("T")


def post_run(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that a method is called after mapping run.

    Raises:
        RuntimeError: If run hasn't happened.
    """

    @wraps(method)
    def wrapper(self: "Mapping", *args: Any, **kwargs: Any) -> T:
        if self.data is None:
            raise RuntimeError(
                "The mapping must be run before attempting this operation."
            )
        return method(self, *args, **kwargs)

    return wrapper


class Mapping:
    """Links encoded by external data."""

    def __init__(
        self,
        dag: DAG,
        location: Location,
        name: str,
        extract_transform: str,
        description: str | None = None,
    ) -> None:
        """Initialise mapping node."""
        self.dag = dag
        self.location = location
        self.name = name
        self.extract_transform = extract_transform
        self.description = description

    TODO: TypeAlias = None

    @property
    def config(self) -> TODO:
        """Generate MappingConfig from Mapping."""
        raise NotImplementedError

    def to_resolution(self) -> None:
        """Generate MappingConfig from Mapping."""
        raise NotImplementedError

    def from_resolution(self) -> None:
        """Generate MappingConfig from Mapping."""
        raise NotImplementedError

    @property
    def resolution_path(self) -> ResolutionPath:
        """Returns the resolution path."""
        return ResolutionPath(
            collection=self.dag.name, run=self.dag.run, name=self.name
        )

    @profile_time(attr="name")
    def run(self, batch_size: int | None = None) -> ArrowTable:
        """."""
        pass

    @post_run
    @profile_time(attr="name")
    def sync(self) -> None:
        """."""

    def clear_data(self) -> None:
        """Deletes data computed for node."""
        self.data = None
