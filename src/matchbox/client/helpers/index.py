"""Functions to index data sources to the Matchbox server."""

from matchbox.client import _handler
from matchbox.common.sources import SourceConfig


def index(
    source: SourceConfig,
    batch_size: int | None = None,
) -> None:
    """Indexes data in Matchbox.

    Args:
        source: A SourceConfig with credentials set
        batch_size: the size of each batch when fetching data from the warehouse,
            which helps reduce the load on the database. Default is None.
    """
    if not source.location.credentials:
        raise ValueError("Source credentials are not set")

    _handler.index(source=source, batch_size=batch_size)
