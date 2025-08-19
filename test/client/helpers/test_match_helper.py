import pytest
from httpx import Response
from respx import MockRouter
from sqlalchemy import Engine

from matchbox import match
from matchbox.client.queries import Match
from matchbox.common.dtos import BackendResourceType, NotFoundError
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.factories.sources import source_factory


def test_match_ok(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """The client can perform the right call for matching."""
    # Set up mocks
    source_testkit = source_factory(engine=sqlite_warehouse, name="source")
    source_testkit.write_to_location(sqlite_warehouse, set_client=True)
    target1_testkit = source_factory(engine=sqlite_warehouse, name="target1")
    target1_testkit.write_to_location(sqlite_warehouse, set_client=True)
    target2_testkit = source_factory(engine=sqlite_warehouse, name="target2")
    target2_testkit.write_to_location(sqlite_warehouse, set_client=True)

    mock_match1 = Match(
        cluster=1,
        source="source",
        source_id={"a"},
        target="target",
        target_id={"b"},
    )
    mock_match2 = Match(
        cluster=1,
        source="source",
        source_id={"a"},
        target="target2",
        target_id={"b"},
    )
    # The standard JSON serialiser does not handle Pydantic objects
    serialised_matches = (
        f"[{mock_match1.model_dump_json()}, {mock_match2.model_dump_json()}]"
    )

    match_route = matchbox_api.get("/match").mock(
        return_value=Response(200, content=serialised_matches)
    )
    matchbox_api.get(f"/sources/{source_testkit.source_config.name}").mock(
        return_value=Response(
            200, json=source_testkit.source_config.model_dump(mode="json")
        )
    )
    matchbox_api.get(f"/sources/{target1_testkit.source_config.name}").mock(
        return_value=Response(
            200, json=target1_testkit.source_config.model_dump(mode="json")
        )
    )
    matchbox_api.get(f"/sources/{target2_testkit.source_config.name}").mock(
        return_value=Response(
            200, json=target2_testkit.source_config.model_dump(mode="json")
        )
    )

    # Use match function
    res = match(
        "target1",
        "target2",
        source="source",
        key="pk1",
        resolution="foo",
    )

    # Verify results
    assert len(res) == 2
    assert isinstance(res[0], Match)
    param_set = sorted(match_route.calls.last.request.url.params.multi_items())
    assert param_set == sorted(
        [
            ("targets", "target1"),
            ("targets", "target2"),
            ("source", "source"),
            ("key", "pk1"),
            ("resolution", "foo"),
        ]
    )


def test_match_404_resolution(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """The client can handle a resolution not found error."""
    # Set up mocks
    source_testkit = source_factory(engine=sqlite_warehouse, name="source")
    target_testkit = source_factory(engine=sqlite_warehouse, name="target")

    matchbox_api.get("/match").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="Resolution 42 not found",
                entity=BackendResourceType.RESOLUTION,
            ).model_dump(),
        )
    )
    matchbox_api.get(f"/sources/{source_testkit.source_config.name}").mock(
        return_value=Response(
            200, json=source_testkit.source_config.model_dump(mode="json")
        )
    )
    matchbox_api.get(f"/sources/{target_testkit.source_config.name}").mock(
        return_value=Response(
            200, json=target_testkit.source_config.model_dump(mode="json")
        )
    )

    # Use match function
    with pytest.raises(MatchboxResolutionNotFoundError, match="42"):
        match(
            "target",
            source="source",
            key="pk1",
            resolution="foo",
        )


def test_match_404_source(matchbox_api: MockRouter, sqlite_warehouse: Engine):
    """The client can handle a source not found error."""
    target_testkit = source_factory(engine=sqlite_warehouse, name="target")

    matchbox_api.get("/sources/source").mock(
        return_value=Response(
            404,
            json=NotFoundError(
                details="SourceConfig 42 not found",
                entity=BackendResourceType.SOURCE,
            ).model_dump(),
        )
    )
    matchbox_api.get(f"/sources/{target_testkit.source_config.name}").mock(
        return_value=Response(
            200, json=target_testkit.source_config.model_dump(mode="json")
        )
    )

    # Use match function
    with pytest.raises(MatchboxSourceNotFoundError, match="42"):
        match(
            "target",
            source="source",
            key="pk1",
            resolution="foo",
        )
