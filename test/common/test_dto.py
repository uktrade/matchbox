import pytest

from matchbox.common.dtos import Match


def test_match_validates():
    """Match objects are validated when they're instantiated."""
    Match(
        cluster=1,
        source="test.source_config",
        source_id={"a"},
        target="test.target",
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source="test.source_config",
            target="test.target",
            target_id={"b"},
        )

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(
            source="test.source_config",
            source_id={"a"},
            target="test.target",
            target_id={"b"},
        )

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(
            cluster=1,
            source="test.source_config",
            target="test.target",
        )
