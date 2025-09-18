import pytest
from httpx import Response
from respx import MockRouter
from sqlalchemy import Engine, create_engine

from matchbox.client.helpers.index import get_source
from matchbox.client.sources import RelationalDBLocation, Source
from matchbox.common.dtos import (
    BackendResourceType,
    NotFoundError,
)
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
)
from matchbox.common.factories.sources import source_factory


def test_get_source_success(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Test successful retrieval of source config."""
    # Create test source
    testkit = source_factory(
        engine=sqlite_warehouse, name="test_source"
    ).write_to_location()

    # Mock API response
    matchbox_api.get("/resolutions/test_source").mock(
        return_value=Response(
            200, json=testkit.source.to_resolution().model_dump(mode="json")
        )
    )

    # Call function
    result = get_source(
        "test_source",
        location=RelationalDBLocation(
            name=testkit.source.to_resolution().config.location_config.name,
            client=sqlite_warehouse,
        ),
    )

    # Verify result
    assert result.name == "test_source"
    assert isinstance(result, Source)


def test_get_source_with_valid_location(
    matchbox_api: MockRouter, sqlite_warehouse: Engine
):
    """Test get_source with matching location validation."""
    testkit = source_factory(
        engine=sqlite_warehouse, name="test_source"
    ).write_to_location()

    matchbox_api.get("/resolutions/test_source").mock(
        return_value=Response(
            200, json=testkit.source.to_resolution().model_dump(mode="json")
        )
    )

    # Should succeed when location matches
    location = RelationalDBLocation(
        name=testkit.source.to_resolution().config.location_config.name,
        client=sqlite_warehouse,
    )
    result = get_source("test_source", location=location)
    assert result.name == "test_source"


@pytest.mark.parametrize(
    ["validation_param", "validation_value", "expected_error"],
    [
        pytest.param(
            "location",
            RelationalDBLocation(
                name="other_location", client=create_engine("sqlite:///:memory:")
            ),
            "does not match the provided location",
            id="location-mismatch",
        ),
        pytest.param(
            "extract_transform",
            "different_transform",
            "does not match the provided extract/transform",
            id="extract-transform-mismatch",
        ),
        pytest.param(
            "key_field",
            "different_key",
            "does not match the provided key field",
            id="key-field-mismatch",
        ),
        pytest.param(
            "index_fields",
            ["different_field"],
            "does not match the provided index fields",
            id="index-fields-mismatch",
        ),
    ],
)
def test_get_source_validation_mismatch(
    validation_param: str,
    validation_value,
    expected_error: str,
    matchbox_api: MockRouter,
    sqlite_warehouse: Engine,
):
    """Test get_source raises error when validation parameters don't match."""
    testkit = source_factory(
        engine=sqlite_warehouse, name="test_source"
    ).write_to_location()

    matchbox_api.get("/resolutions/test_source").mock(
        return_value=Response(
            200, json=testkit.source.to_resolution().model_dump(mode="json")
        )
    )

    kwargs = {validation_param: validation_value}
    if validation_param != "location":
        kwargs["location"] = testkit.source.location
    with pytest.raises(ValueError, match=expected_error):
        get_source("test_source", **kwargs)


def test_get_source_404_error(matchbox_api: MockRouter):
    """Test get_source handles 404 source not found error."""
    matchbox_api.get("/resolutions/nonexistent").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="SourceConfig nonexistent not found",
                entity=BackendResourceType.RESOLUTION,
            ).model_dump(),
        )
    )

    with pytest.raises(MatchboxResolutionNotFoundError, match="nonexistent"):
        get_source(name="nonexistent", location=RelationalDBLocation("", None))
