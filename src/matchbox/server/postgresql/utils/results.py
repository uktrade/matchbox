"""Utilities for querying model results from the PostgreSQL backend."""

from typing import NamedTuple

from sqlalchemy import select

from matchbox.common.dtos import ModelConfig, ModelType
from matchbox.common.graph import ResolutionNodeType
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    ResolutionFrom,
    Resolutions,
)


class SourceInfo(NamedTuple):
    """Information about a model's sources."""

    left: int
    right: int | None
    left_ancestors: set[int]
    right_ancestors: set[int] | None


def _get_model_parents(resolution_id: int) -> tuple[bytes, bytes | None]:
    """Get the model's immediate parent models."""
    parent_query = (
        select(Resolutions.resolution_id, Resolutions.type)
        .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.parent)
        .where(ResolutionFrom.child == resolution_id)
        .where(ResolutionFrom.level == 1)
    )

    with MBDB.get_session() as session:
        parents = session.execute(parent_query).fetchall()

    if len(parents) == 1:
        return parents[0][0], None
    elif len(parents) == 2:
        p1, p2 = parents
        p1_id, p1_type = p1
        p2_id, p2_type = p2
        # Put source first if it exists
        if p1_type == ResolutionNodeType.SOURCE:
            return p1_id, p2_id
        elif p2_type == ResolutionNodeType.SOURCE:
            return p2_id, p1_id
        # Both models, maintain original order
        return p1_id, p2_id
    else:
        raise ValueError(f"Model has unexpected number of parents: {len(parents)}")


def _get_source_info(resolution_id: int) -> SourceInfo:
    """Get source resolutions and their ancestry information."""
    left_id, right_id = _get_model_parents(resolution_id=resolution_id)

    with MBDB.get_session() as session:
        left = session.get(Resolutions, left_id)
        right = session.get(Resolutions, right_id) if right_id else None

        left_ancestors = {left_id} | {m.resolution_id for m in left.ancestors}
        if right:
            right_ancestors = {right_id} | {m.resolution_id for m in right.ancestors}
        else:
            right_ancestors = None

    return SourceInfo(
        left=left_id,
        right=right_id,
        left_ancestors=left_ancestors,
        right_ancestors=right_ancestors,
    )


def get_model_config(resolution: Resolutions) -> ModelConfig:
    """Get metadata for a model resolution."""
    if resolution.type != ResolutionNodeType.MODEL:
        raise ValueError("Expected resolution of type model")

    source_info: SourceInfo = _get_source_info(resolution_id=resolution.resolution_id)

    with MBDB.get_session() as session:
        left = session.get(Resolutions, source_info.left)
        right = (
            session.get(Resolutions, source_info.right) if source_info.right else None
        )

        return ModelConfig(
            name=resolution.name,
            description=resolution.description or "",
            type=ModelType.DEDUPER if source_info.right is None else ModelType.LINKER,
            left_resolution=left.name,
            right_resolution=right.name if source_info.right else None,
        )
