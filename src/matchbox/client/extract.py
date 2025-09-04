"""Functions to extract data out of the Matchbox server."""

from pyarrow import Table as ArrowTable

from matchbox.client import _handler
from matchbox.common.exceptions import MatchboxSourceNotFoundError
from matchbox.common.graph import ModelResolutionName, SourceResolutionName


def key_field_map(
    resolution: ModelResolutionName,
    source_filter: list[SourceResolutionName] | None = None,
    location_names: list[str] | None = None,
) -> ArrowTable:
    """Return matchbox IDs to source key mapping, optionally filtering.

    Args:
        resolution: The resolution name to use for the query.
        source_filter: An optional list of source resolution names to filter by.
        location_names: An optional list of location names to filter by.
    """
    # Get all sources in scope of the resolution
    sources = _handler.get_resolution_source_configs(name=resolution)

    if source_filter:
        sources = [s for s in sources if s.name in source_filter]

    if location_names:
        sources = [s for s in sources if s.location_config.name in location_names]

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
