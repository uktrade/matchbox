"""Utilities for querying model results from the PostgreSQL backend."""

from typing import NamedTuple

import pyarrow as pa
from sqlalchemy import and_, func, literal, literal_column, select, union

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ModelConfig, ModelType
from matchbox.common.graph import ResolutionNodeType
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import compile_sql


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
    session = MBDB.get_session()

    # Get direct parent resolutions
    parent_query = (
        select(ResolutionFrom.parent)
        .where(ResolutionFrom.child == resolution.resolution_id)
        .where(ResolutionFrom.level == 1)
    )
    parent_ids = [row[0] for row in session.execute(parent_query).all()]

    if not parent_ids:
        # No parents means no results to recover
        return pa.table(
            {
                "left_id": pa.array([], type=pa.uint64()),
                "right_id": pa.array([], type=pa.uint64()),
                "probability": pa.array([], type=pa.uint8()),
                "cluster_id": pa.array([], type=pa.uint64()),
            }
        )

    # Get pair clusters from this model (role_flag <= 1)
    pair_clusters_cte = (
        select(
            Probabilities.cluster.label("pair_cluster"), Probabilities.probability
        ).where(
            and_(
                Probabilities.resolution == resolution.resolution_id,
                Probabilities.role_flag <= 1,  # Pairs only
            )
        )
    ).cte("pair_clusters")

    # Get all leaves for each pair cluster
    pair_leaves_cte = (
        select(
            pair_clusters_cte.c.pair_cluster,
            pair_clusters_cte.c.probability,
            Contains.leaf,
        )
        .select_from(pair_clusters_cte)
        .join(Contains, Contains.root == pair_clusters_cte.c.pair_cluster)
    ).cte("pair_leaves")

    # Get parent clusters - need to handle both model and source parents differently
    parent_resolutions = []
    for parent_id in parent_ids:
        parent_res = session.get(Resolutions, parent_id)
        if parent_res:
            parent_resolutions.append(parent_res)

    # For each parent, get the clusters using the hierarchy resolution logic
    all_parent_clusters = []

    for parent_res in parent_resolutions:
        if parent_res.type == "source":
            # For source resolutions, the "clusters" are the leaf clusters themselves
            # Get all clusters that belong to this source
            source_clusters_query = (
                select(
                    Clusters.cluster_id.label("parent_cluster"),
                    literal(parent_res.resolution_id).label("parent_resolution"),
                )
                .select_from(Clusters)
                .join(
                    ClusterSourceKey, ClusterSourceKey.cluster_id == Clusters.cluster_id
                )
                .join(
                    SourceConfigs,
                    SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
                )
                .where(SourceConfigs.resolution_id == parent_res.resolution_id)
            )
        else:
            # For model resolutions, get clusters with role_flag >= 1
            source_clusters_query = select(
                Probabilities.cluster.label("parent_cluster"),
                literal(parent_res.resolution_id).label("parent_resolution"),
            ).where(
                and_(
                    Probabilities.resolution == parent_res.resolution_id,
                    Probabilities.role_flag >= 1,
                )
            )

        parent_clusters = session.execute(source_clusters_query).all()
        all_parent_clusters.extend(parent_clusters)

    if not all_parent_clusters:
        return pa.table(
            {
                "left_id": pa.array([], type=pa.uint64()),
                "right_id": pa.array([], type=pa.uint64()),
                "probability": pa.array([], type=pa.uint8()),
                "cluster_id": pa.array([], type=pa.uint64()),
            }
        )

    # Create a CTE for parent clusters
    parent_clusters_data = [
        {"parent_cluster": cluster_id, "parent_resolution": res_id}
        for cluster_id, res_id in all_parent_clusters
    ]

    # Convert to temporary table or use direct values
    parent_clusters_cte = (
        select(
            literal_column(str(pc["parent_cluster"])).label("parent_cluster"),
            literal_column(str(pc["parent_resolution"])).label("parent_resolution"),
        )
        for pc in parent_clusters_data
    )

    if len(parent_clusters_data) > 1:
        parent_clusters_cte = union(*parent_clusters_cte).cte("parent_clusters")
    else:
        parent_clusters_cte = parent_clusters_cte[0].cte("parent_clusters")

    # Get leaves for each parent cluster
    # (for source clusters, they are leaves themselves)
    parent_leaves_cte = (
        select(
            parent_clusters_cte.c.parent_cluster,
            parent_clusters_cte.c.parent_resolution,
            func.coalesce(Contains.leaf, parent_clusters_cte.c.parent_cluster).label(
                "leaf"
            ),
        )
        .select_from(parent_clusters_cte)
        .outerjoin(Contains, Contains.root == parent_clusters_cte.c.parent_cluster)
    ).cte("parent_leaves")

    # Find which parent clusters combine to form each pair
    results_query = (
        select(
            pair_leaves_cte.c.pair_cluster.label("id"),
            func.min(parent_leaves_cte.c.parent_cluster).label("left_id"),
            func.max(parent_leaves_cte.c.parent_cluster).label("right_id"),
            pair_leaves_cte.c.probability,
        )
        .select_from(pair_leaves_cte)
        .join(parent_leaves_cte, parent_leaves_cte.c.leaf == pair_leaves_cte.c.leaf)
        .group_by(pair_leaves_cte.c.pair_cluster, pair_leaves_cte.c.probability)
        .having(func.count(func.distinct(parent_leaves_cte.c.parent_cluster)) == 2)
    )

    with MBDB.get_adbc_connection() as conn:
        stmt = compile_sql(results_query)
        logger.debug(f"Recover results SQL: \n {stmt}")

        results_df = sql_to_df(stmt=stmt, connection=conn, return_type="arrow")

    if results_df.shape[0] == 0:
        return pa.table(
            {
                "id": pa.array([], type=pa.uint64()),
                "left_id": pa.array([], type=pa.uint64()),
                "right_id": pa.array([], type=pa.uint64()),
                "probability": pa.array([], type=pa.uint8()),
            }
        )

    return results_df
