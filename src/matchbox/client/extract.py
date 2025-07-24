"""Functions to extract data out of the Matchbox server."""

from pyarrow import Table as ArrowTable

from matchbox.client import _handler
from matchbox.common.dtos import ModelResolutionName, SourceResolutionName
from matchbox.common.exceptions import MatchboxSourceNotFoundError


def key_field_map(
    resolution: ModelResolutionName,
    source_filter: list[SourceResolutionName] | SourceResolutionName | None = None,
    uri_filter: list[str] | str | None = None,
) -> ArrowTable:
    """Return matchbox IDs to source key mapping, optionally filtering.

    Args:
        resolution: The resolution name to use for the query.
        source_filter: A substring or list of substrings to filter source names.
        uri_filter: A substring or list of substrings to filter location URIs.
    """
    # Get all sources in scope of the resolution
    sources = _handler.get_resolution_source_configs(name=resolution)

    if source_filter:
        if isinstance(source_filter, str):
            source_filter = [source_filter]

        # Filter sources by name
        sources = [s for s in sources if s.name in source_filter]

    if uri_filter:
        if isinstance(uri_filter, str):
            uri_filter = [uri_filter]

        # Filter sources by location URI
        sources = [
            s for s in sources if any(sub in str(s.location.uri) for sub in uri_filter)
        ]

    if not sources:
        raise MatchboxSourceNotFoundError("No compatible source was found")

    source_mb_ids: list[ArrowTable] = []
    source_to_key_field: dict[str, str] = {}

    for s in sources:
        # Get Matchbox IDs from backend
        source_mb_ids.append(
            _handler.query(
                source=s.name,
                resolution=resolution,
                return_leaf_id=False,
            )
        )

        source_to_key_field[s.name] = s.key_field.name

    # Join Matchbox IDs to form mapping table
    mapping = source_mb_ids[0]
    mapping = mapping.rename_columns({"key": sources[0].qualified_key})
    if len(sources) > 1:
        for s, mb_ids in zip(sources[1:], source_mb_ids[1:], strict=True):
            mapping = mapping.join(
                right_table=mb_ids, keys="id", join_type="full outer"
            )
            mapping = mapping.rename_columns({"key": s.qualified_key})

    return mapping
