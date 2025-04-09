"""Utilities for querying model results from the PostgreSQL backend."""

from typing import NamedTuple

from pyarrow import Table
from sqlalchemy import Engine, and_, case, func, select
from sqlalchemy.orm import Session

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ModelMetadata, ModelType
from matchbox.common.graph import ResolutionNodeType
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
)
from matchbox.server.postgresql.utils.db import compile_sql


class SourceInfo(NamedTuple):
    """Information about a model's sources."""

    left: int
    right: int | None
    left_ancestors: set[int]
    right_ancestors: set[int] | None


def _get_model_parents(
    engine: Engine, resolution_id: int
) -> tuple[bytes, bytes | None]:
    """Get the model's immediate parent models."""
    parent_query = (
        select(Resolutions.resolution_id, Resolutions.type)
        .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.parent)
        .where(ResolutionFrom.child == resolution_id)
        .where(ResolutionFrom.level == 1)
    )

    with engine.connect() as conn:
        parents = conn.execute(parent_query).fetchall()

    if len(parents) == 1:
        return parents[0][0], None
    elif len(parents) == 2:
        p1, p2 = parents
        p1_id, p1_type = p1
        p2_id, p2_type = p2
        # Put dataset first if it exists
        if p1_type == ResolutionNodeType.DATASET:
            return p1_id, p2_id
        elif p2_type == ResolutionNodeType.DATASET:
            return p2_id, p1_id
        # Both models, maintain original order
        return p1_id, p2_id
    else:
        raise ValueError(f"Model has unexpected number of parents: {len(parents)}")


def _get_source_info(engine: Engine, resolution_id: int) -> SourceInfo:
    """Get source resolutions and their ancestry information."""
    left_id, right_id = _get_model_parents(engine=engine, resolution_id=resolution_id)

    with Session(engine) as session:
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


def get_model_metadata(engine: Engine, resolution: Resolutions) -> ModelMetadata:
    """Get metadata for a model resolution."""
    if resolution.type != ResolutionNodeType.MODEL:
        raise ValueError("Expected resolution of type model")

    source_info: SourceInfo = _get_source_info(
        engine=engine, resolution_id=resolution.resolution_id
    )

    with Session(engine) as session:
        left = session.get(Resolutions, source_info.left)
        right = (
            session.get(Resolutions, source_info.right) if source_info.right else None
        )

        return ModelMetadata(
            name=resolution.name,
            description=resolution.description or "",
            type=ModelType.DEDUPER if source_info.right is None else ModelType.LINKER,
            left_resolution=left.name,
            right_resolution=right.name if source_info.right else None,
        )


def get_model_results(resolution: Resolutions) -> Table:
    """Recover the model's pairwise probabilities and return as a PyArrow table.

    For each probability this model assigned:
    - Get its two immediate children
    - Filter for children that aren't parents of other clusters this model scored
    - Determine left/right by tracing ancestry to source resolutions using query helpers

    Args:
        resolution: Resolution of type model to query

    Returns:
        Table containing the original pairwise probabilities
    """
    engine = MBDB.get_engine()
    if resolution.type != ResolutionNodeType.MODEL:
        raise ValueError("Expected resolution of type model")

    source_info: SourceInfo = _get_source_info(
        engine=engine, resolution_id=resolution.resolution_id
    )

    # First get all clusters this resolution assigned probabilities to
    resolution_clusters = (
        select(Probabilities.cluster)
        .where(Probabilities.resolution == resolution.resolution_id)
        .cte("resolution_clusters")
    )

    # Get clusters that are parents in Contains for resolution's probabilities
    resolution_parents = (
        select(Contains.parent)
        .join(resolution_clusters, Contains.child == resolution_clusters.c.cluster)
        .cte("resolution_parents")
    )

    # Get valid pairs (those with exactly 2 children)
    # where neither child is a parent in the resolution's hierarchy
    valid_pairs = (
        select(Contains.parent)
        .join(
            Probabilities,
            and_(
                Probabilities.cluster == Contains.parent,
                Probabilities.resolution == resolution.resolution_id,
            ),
        )
        .where(~Contains.child.in_(select(resolution_parents)))
        .group_by(Contains.parent)
        .having(func.count() == 2)
        .cte("valid_pairs")
    )

    # Join to get children and probabilities
    pairs = (
        select(
            Contains.parent.label("id"),
            func.array_agg(
                case(
                    (
                        Contains.child.in_(list(source_info.left_ancestors)),
                        Contains.child,
                    ),
                    (
                        Contains.child.in_(list(source_info.right_ancestors))
                        if source_info.right_ancestors
                        else Contains.child.notin_(list(source_info.left_ancestors)),
                        Contains.child,
                    ),
                )
            ).label("children"),
            func.min(Probabilities.probability).label("probability"),
        )
        .join(valid_pairs, valid_pairs.c.parent == Contains.parent)
        .join(
            Probabilities,
            and_(
                Probabilities.cluster == Contains.parent,
                Probabilities.resolution == resolution.resolution_id,
            ),
        )
        .group_by(Contains.parent)
    ).cte("pairs")

    # Final select to properly split out left and right
    final_select = select(
        pairs.c.id,
        pairs.c.children[1].label("left_id"),
        pairs.c.children[2].label("right_id"),
        pairs.c.probability,
    )

    with MBDB.get_adbc_connection() as conn:
        return sql_to_df(
            stmt=compile_sql(final_select),
            connection=conn,
            return_type="arrow",
        )
