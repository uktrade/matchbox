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
        sources = [s for s in sources if s.location.name in location_names]

    if not sources:
        raise MatchboxSourceNotFoundError("No compatible source was found")

    source_mb_ids: list[ArrowTable] = []
    source_to_key_field: dict[str, str] = {}

    # Store source names and key field mappings
    source_names = [s.name for s in sources]
    for s in sources:
        source_to_key_field[s.name] = s.key_field.name

    if len(sources) == 1:
        # Single source - make individual call
        source_mb_ids.append(
            _handler.query(
                sources=[sources[0].name],
                resolution=resolution,
                return_leaf_id=False,
            )
        )
    else:
        # Multiple sources - make single multi-source call
        combined_result = _handler.query(
            sources=source_names,
            resolution=resolution,
            return_leaf_id=False,
        )

        # Split the combined result by source
        import polars as pl

        combined_df = pl.from_arrow(combined_result)
        for source_name in source_names:
            source_data = combined_df.filter(pl.col("source") == source_name).to_arrow()
            source_mb_ids.append(source_data)

    # Join Matchbox IDs to form mapping table
    mapping = source_mb_ids[0].select(["id", "key"])
    mapping = mapping.rename_columns({"key": sources[0].qualified_key})
    if len(sources) > 1:
        for s, mb_ids in zip(sources[1:], source_mb_ids[1:], strict=True):
            mb_ids_selected = mb_ids.select(["id", "key"])
            mapping = mapping.join(
                right_table=mb_ids_selected, keys="id", join_type="full outer"
            )
            mapping = mapping.rename_columns({"key": s.qualified_key})

    return mapping
