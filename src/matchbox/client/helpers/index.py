"""Functions to index data sources to the Matchbox server."""

from matchbox.client import _handler
from matchbox.client.sources import Location, Source


def index(
    source: Source,
    batch_size: int | None = None,
) -> None:
    """Indexes data in Matchbox.

    Args:
        source: A Source
        batch_size: the size of each batch when fetching data from the warehouse,
            which helps reduce the load on the database. Default is None.
    """
    if not source.location.client:
        raise ValueError("Source client not set")

    data_hashes = source.hash_data(batch_size=batch_size)
    _handler.index(source_config=source.config, data_hashes=data_hashes)


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
        A SourceConfig object.
    """
    config = _handler.get_source_config(name=name)

    validations = [
        (location.config, config.location_config, "location"),
        (extract_transform, config.extract_transform, "extract/transform"),
        (key_field, config.key_field, "key field"),
    ]

    for provided, actual, field_name in validations:
        if provided is not None and actual != provided:
            raise ValueError(
                f"Source {name} does not match the provided {field_name}: {provided}"
            )

    if index_fields is not None and set(config.index_fields) != set(index_fields):
        raise ValueError(
            f"Source {name} does not match the provided index fields: {index_fields}"
        )

    return Source(
        location=location,
        name=config.name,
        extract_transform=config.extract_transform,
        key_field=config.key_field,
        index_fields=config.index_fields,
    )
