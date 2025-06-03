"""Utilities for querying model results from the PostgreSQL backend."""

from typing import NamedTuple

import pyarrow as pa
from sqlalchemy import and_, select

from matchbox.common.dtos import ModelConfig, ModelType
from matchbox.common.graph import ResolutionNodeType
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
)
from matchbox.server.postgresql.utils.query import get_clusters_with_leaves


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


def get_model_results(resolution: Resolutions) -> pa.Table:
    """Recovers the original Results object for a model resolution.

    This function reconstructs the left_id, right_id, probability table that was
    originally passed to insert_results() by examining the stored probabilities
    and finding which parent clusters combined to form each pair.

    Args:
        resolution: Model resolution to recover results for

    Returns:
        PyArrow table with columns: id, left_id, right_id, probability
    """
    # 1. Get all parent clusters this model knows about using tested function
    parent_clusters_dict = get_clusters_with_leaves(resolution)

    if not parent_clusters_dict:
        # No parent clusters means no results to recover
        return pa.table(
            {
                "id": pa.array([], type=pa.uint64()),
                "left_id": pa.array([], type=pa.uint64()),
                "right_id": pa.array([], type=pa.uint64()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    # 2. Get pairwise probabilities for this model (role_flag <= 1)
    with MBDB.get_session() as session:
        pairwise_query = select(Probabilities.cluster, Probabilities.probability).where(
            and_(
                Probabilities.resolution == resolution.resolution_id,
                Probabilities.role_flag <= 1,  # Pairs only (0=pairwise, 1=both)
            )
        )

        pairwise_results = session.execute(pairwise_query).all()

    if not pairwise_results:
        # No pairwise clusters means no results to recover
        return pa.table(
            {
                "id": pa.array([], type=pa.uint64()),
                "left_id": pa.array([], type=pa.uint64()),
                "right_id": pa.array([], type=pa.uint64()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    # 3. Build mapping from leaf cluster to parent cluster
    # Each parent cluster "owns" certain leaf clusters
    leaf_to_parent = {}
    for parent_cluster_id, cluster_info in parent_clusters_dict.items():
        for leaf_info in cluster_info["leaves"]:
            leaf_id = leaf_info["leaf_id"]
            leaf_to_parent[leaf_id] = parent_cluster_id

    # 4. For each pairwise cluster, find which parent clusters it combines
    results = []

    with MBDB.get_session() as session:
        for pair_cluster_id, probability in pairwise_results:
            # Get leaves of this pairwise cluster
            leaves_query = select(Contains.leaf).where(Contains.root == pair_cluster_id)
            leaves = [row[0] for row in session.execute(leaves_query).all()]

            # Find which parent clusters these leaves belong to
            parent_clusters_for_pair = set()
            for leaf_id in leaves:
                if leaf_id in leaf_to_parent:
                    parent_clusters_for_pair.add(leaf_to_parent[leaf_id])

            # We expect exactly 2 parent clusters for a valid pair
            if len(parent_clusters_for_pair) == 2:
                parent_list = sorted(parent_clusters_for_pair)
                results.append(
                    {
                        "id": pair_cluster_id,  # Include the pair cluster ID
                        "left_id": parent_list[0],
                        "right_id": parent_list[1],
                        "probability": probability,
                    }
                )
            elif len(parent_clusters_for_pair) == 1:
                # Self-pair within same parent cluster - skip or handle differently
                continue
            else:
                # Invalid pair (0 or >2 parents) - skip
                continue

    if not results:
        return pa.table(
            {
                "id": pa.array([], type=pa.uint64()),
                "left_id": pa.array([], type=pa.uint64()),
                "right_id": pa.array([], type=pa.uint64()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    # 5. Convert to PyArrow table
    return pa.table(
        {
            "id": pa.array([r["id"] for r in results], type=pa.uint64()),
            "left_id": pa.array([r["left_id"] for r in results], type=pa.uint64()),
            "right_id": pa.array([r["right_id"] for r in results], type=pa.uint64()),
            "probability": pa.array(
                [r["probability"] for r in results], type=pa.uint8()
            ),
        }
    )
