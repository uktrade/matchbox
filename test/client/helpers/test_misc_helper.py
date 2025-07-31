from importlib.metadata import version
from typing import Callable

import pytest
from httpx import Response
from respx import MockRouter
from sqlalchemy import Engine

from matchbox.client._handler import create_client
from matchbox.client._settings import ClientSettings
from matchbox.client.helpers import comparison, select
from matchbox.common.dtos import BackendResourceType, NotFoundError
from matchbox.common.exceptions import MatchboxSourceNotFoundError
from matchbox.common.factories.sources import (
    linked_sources_factory,
    source_from_tuple,
)


def test_comparisons():
    comparison_name_id = comparison(
        sql_condition=(
            "l.company_name = r.company_name and l.data_hub_id = r.data_hub_id"
        )
    )

    assert comparison_name_id is not None


def test_create_client():
    mock_settings = ClientSettings(api_root="http://example.com", timeout=20)
    client = create_client(mock_settings)

    assert client.headers.get("X-Matchbox-Client-Version") == version("matchbox_db")
    assert client.base_url == mock_settings.api_root
    assert client.timeout.connect == mock_settings.timeout
    assert client.timeout.pool == mock_settings.timeout
    assert client.timeout.read == 60 * 30
    assert client.timeout.write == 60 * 30


def test_select_default_engine(
    matchbox_api: MockRouter,
    env_setter: Callable[[str, str], None],
    sqlite_warehouse: Engine,
):
    """We can select without explicit client if default is set."""
    default_engine = sqlite_warehouse.url.render_as_string(hide_password=False)
    env_setter("MB__CLIENT__DEFAULT_WAREHOUSE", default_engine)

    # Dummy data and source
    testkit = source_from_tuple(
        data_tuple=({"a": 1, "b": "2"}, {"a": 10, "b": "20"}),
        data_keys=["0", "1"],
        name="bar",
        engine=sqlite_warehouse,
    )
    testkit.write_to_location(sqlite_warehouse, set_client=True)

    # Mock API
    matchbox_api.get(f"/sources/{testkit.source_config.name}").mock(
        return_value=Response(200, json=testkit.source_config.model_dump(mode="json"))
    )

    # Select sources
    selection = select("bar")[0]

    # Check selector contains what we expect
    assert set(f.name for f in selection.fields) == {"a", "b"}
    assert selection.source.name == "bar"
    assert str(selection.source.location.client.url) == str(sqlite_warehouse.url)


def test_select_missing_client():
    """We must pass client if a default is not set"""
    with pytest.raises(ValueError, match="Client"):
        select("test.bar")


def test_select_mixed_style(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """We can select specific columns from some of the sources"""
    linked = linked_sources_factory(engine=sqlite_warehouse)
    linked.write_to_location(sqlite_warehouse, set_client=True)

    source1 = linked.sources["crn"].source_config
    source2 = linked.sources["cdms"].source_config

    # Mock API
    matchbox_api.get(f"/sources/{source1.name}").mock(
        return_value=Response(200, json=source1.model_dump(mode="json"))
    )
    matchbox_api.get(f"/sources/{source2.name}").mock(
        return_value=Response(200, json=source2.model_dump(mode="json"))
    )

    selection = select({"crn": ["company_name"]}, "cdms", client=sqlite_warehouse)

    assert set(f.name for f in selection[0].fields) == {"company_name"}
    assert set(f.name for f in selection[1].fields) == {"cdms", "crn"}
    assert selection[0].source.name == source1.name
    assert selection[1].source.name == source2.name
    assert selection[0].source.location.client == sqlite_warehouse
    assert selection[1].source.location.client == sqlite_warehouse


def test_select_404_source_get(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """Handles source 404 error when retrieving source."""
    # Mock API
    matchbox_api.get("/sources/foo").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="SourceConfig 42 not found",
                entity=BackendResourceType.SOURCE,
            ).model_dump(),
        )
    )

    with pytest.raises(MatchboxSourceNotFoundError, match="42"):
        select({"foo": ["a", "b"]}, client=sqlite_warehouse)
