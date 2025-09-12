"""Functions to index data sources to the Matchbox server."""

from matchbox.client import _handler
from matchbox.client.sources import Location, Source


def get_source(
    name: str,
    location: Location,
    extract_transform: str | None = None,
    key_field: str | None = None,
    index_fields: list[str] | None = None,
) -> Source:
    """Get a Source for an existing source.

    Args:
        name: The name of the source resolution.
        location: If provided, will validate the returned SourceConfig
            against the location.
        extract_transform: If provided, will validate the returned SourceConfig
            against the extract/transform logic.
        key_field: If provided, will validate the returned SourceConfig
            against the key field.
        index_fields: If provided, will validate the returned SourceConfig
            against the index fields.

    Returns:
        A Source object.
    """
    resolution = _handler.get_source_resolution(name=name)

    validations = [
        (location.config, resolution.config.location_config, "location"),
        (extract_transform, resolution.config.extract_transform, "extract/transform"),
        (key_field, resolution.config.key_field, "key field"),
    ]

    for provided, actual, field_name in validations:
        if provided is not None and actual != provided:
            raise ValueError(
                f"Source {name} does not match the provided {field_name}: {provided}"
            )

    if index_fields is not None and set(resolution.config.index_fields) != set(
        index_fields
    ):
        raise ValueError(
            f"Source {name} does not match the provided index fields: {index_fields}"
        )

    return Source.from_resolution(resolution=resolution, location=location)
