import pytest

from matchbox.common.dtos import (
    Collection,
    Match,
    ResolutionPath,
)


def test_match_validates() -> None:
    """Match objects are validated when they're instantiated."""
    source_path = ResolutionPath(name="source", collection="default", run=1)
    target_path = ResolutionPath(name="target", collection="default", run=1)
    Match(
        cluster=1,
        source=source_path,
        source_id={"a"},
        target=target_path,
        target_id={"b"},
    )

    # Missing source_id with target_id
    with pytest.raises(ValueError):
        Match(cluster=1, source=source_path, target=target_path, target_id={"b"})

    # Missing cluster with target_id
    with pytest.raises(ValueError):
        Match(source=source_path, source_id={"a"}, target=target_path, target_id={"b"})

    # Missing source_id with cluster
    with pytest.raises(ValueError):
        Match(cluster=1, source=source_path, target=target_path)


def test_validate_collection() -> None:
    """Default run name needs to be within all runs."""
    # No default
    Collection(runs=[1, 2])

    # Valid default
    Collection(default_run=2, runs=[1, 2])

    with pytest.raises(ValueError):
        Collection(default_run=2, runs=[1])
