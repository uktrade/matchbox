"""Functions to extract data out of the Matchbox server."""

import re

from pyarrow import Table as ArrowTable

from matchbox.client import _handler
from matchbox.common.dtos import ModelResolutionName
from matchbox.common.exceptions import MatchboxSourceNotFoundError
from matchbox.common.sources import SourceConfig


def _qualify_identifier(source: SourceConfig) -> str:
    """Return the identifier for the source qualified by its name."""
    return source.name.replace(".", "_") + "_" + source.identifier.name


def primary_keys_map(
    resolution: ModelResolutionName,
    location_match: str | None = None,
) -> ArrowTable:
    """Return matchbox IDs to source IDs mapping, optionally filtering location URIs.

    Args:
        resolution: The model resolution name.
        location_match: An optional regex pattern to filter location URIs.
    """
    # Get all sources in scope of the resolution
    sources = _handler.get_resolution_sources(name=resolution)

    if location_match:
        # Filter only sources compatible with pattern
        pattern = re.compile(location_match)
        sources = [s for s in sources if pattern.search(s.location.uri)]

    if not sources:
        raise MatchboxSourceNotFoundError("No compatible source was found")

    source_mb_ids: list[ArrowTable] = []
    source_identifiers: dict[str, str] = {}

    for s in sources:
        # Get Matchbox IDs from backend
        source_mb_ids.append(
            _handler.query(
                source=s.name,
                resolution=resolution,
            )
        )

        source_identifiers[s.name] = s.identifier.name

    # Join Matchbox IDs to form mapping table
    mapping = source_mb_ids[0]
    mapping = mapping.rename_columns(
        {"source_identifier": _qualify_identifier(sources[0])}
    )
    if len(sources) > 1:
        for s, mb_ids in zip(sources[1:], source_mb_ids[1:], strict=True):
            mapping = mapping.join(
                right_table=mb_ids, keys="id", join_type="full outer"
            )
            mapping = mapping.rename_columns(
                {"source_identifier": _qualify_identifier(s)}
            )

    return mapping
