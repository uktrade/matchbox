import pytest

from matchbox.common.dtos import (
    Collection,
    CollectionName,
    Match,
    ModelResolutionName,
    ResolutionName,
    ResolutionPath,
    SourceResolutionName,
    VersionName,
)
from matchbox.common.exceptions import MatchboxNameError


def test_validate_names():
    name_classes = [
        CollectionName,
        ModelResolutionName,
        ResolutionName,
        SourceResolutionName,
        VersionName,
    ]

    [NameClass("Valid.name_-") for NameClass in name_classes]

    for NameClass in name_classes:
        with pytest.raises(MatchboxNameError):
            NameClass("invalid name")


def test_match_validates():
    """Match objects are validated when they're instantiated."""
    source_path = ResolutionPath(name="source", collection="default", version="v1")
    target_path = ResolutionPath(name="target", collection="default", version="v1")
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


def test_validate_collection():
    """Default version name needs to be within all versions."""
    # No default
    Collection(versions=["v1", "v2"])

    # Valid default
    Collection(default_version="v2", versions=["v1", "v2"])

    with pytest.raises(ValueError):
        Collection(default_version="v2", versions=["v1"])
