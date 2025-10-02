"""Evaluation logic for PostgreSQL adapter."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import Table
from sqlalchemy import BIGINT, func, select

from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES,
    SCHEMA_JUDGEMENTS,
)
from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.eval import Judgement, precision_recall
from matchbox.common.exceptions import (
    MatchboxUserNotFoundError,
)
from matchbox.common.transform import hash_cluster_leaves
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    EvalJudgements,
    PKSpace,
    Probabilities,
    Resolutions,
    SourceConfigs,
    Users,
)
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    ingest_to_temporary_table,
)
from matchbox.server.postgresql.utils.query import (
    build_unified_query,
)


def insert_judgement(judgement: Judgement):
    """Record judgement to server."""
    # Note: we don't currently check that the shown cluster ID points to
    # the source cluster IDs. We must assume this is well-formed.

    # Check that the user exists
    with MBDB.get_session() as session:
        if not session.scalar(
            select(Users.name).where(Users.user_id == judgement.user_id)
        ):
            raise MatchboxUserNotFoundError(user_id=judgement.user_id)

    for leaves in judgement.endorsed:
        with MBDB.get_session() as session:
            # Compute hash corresponding to set of source clusters (leaves)
            leaf_hashes = [
                session.scalar(
                    select(Clusters.cluster_hash).where(Clusters.cluster_id == leaf_id)
                )
                for leaf_id in leaves
            ]
            endorsed_cluster_hash = hash_cluster_leaves(leaf_hashes)

            # If cluster with this hash does not exist, create it.
            # Note that only endorsed clusters might be new. The cluster shown to
            # the user is guaranteed to exist in the backend; we have checked above.
            if not (
                endorsed_cluster_id := session.scalar(
                    select(Clusters.cluster_id).where(
                        Clusters.cluster_hash == endorsed_cluster_hash
                    )
                )
            ):
                endorsed_cluster_id = PKSpace.reserve_block(
                    table="clusters", block_size=1
                )
                session.add(
                    Clusters(
                        cluster_id=endorsed_cluster_id,
                        cluster_hash=endorsed_cluster_hash,
                    )
                )
                for leaf_id in leaves:
                    session.add(Contains(root=endorsed_cluster_id, leaf=leaf_id))

            session.add(
                EvalJudgements(
                    user_id=judgement.user_id,
                    shown_cluster_id=judgement.shown,
                    endorsed_cluster_id=endorsed_cluster_id,
                    timestamp=datetime.now(UTC),
                )
            )

            session.commit()


def get_judgements() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Get all judgements from server."""
    judgements_stmt = select(
        EvalJudgements.user_id,
        EvalJudgements.endorsed_cluster_id.label("endorsed"),
        EvalJudgements.shown_cluster_id.label("shown"),
    )

    with MBDB.get_adbc_connection() as conn:
        judgements = sql_to_df(
            stmt=compile_sql(judgements_stmt),
            connection=conn.dbapi_connection,
            return_type="arrow",
        )

        if not len(judgements):
            return (
                pl.DataFrame([], schema=pl.Schema(SCHEMA_JUDGEMENTS)),
                pl.DataFrame([], schema=pl.Schema(SCHEMA_CLUSTER_EXPANSION)),
            )

        shown_clusters = set(judgements["shown"].to_pylist())
        endorsed_clusters = set(judgements["endorsed"].to_pylist())
        referenced_clusters = Table.from_pydict(
            {"root": list(shown_clusters | endorsed_clusters)}
        )

    with ingest_to_temporary_table(
        table_name="judgements",
        schema_name="mb",
        column_types={
            "root": BIGINT,
        },
        data=referenced_clusters,
    ) as temp_table:
        cluster_expansion_stmt = (
            select(temp_table.c.root, func.array_agg(Contains.leaf).label("leaves"))
            .select_from(temp_table)
            .join(Contains, Contains.root == temp_table.c.root)
            .group_by(temp_table.c.root)
        )

        with MBDB.get_adbc_connection() as conn:
            cluster_expansion = sql_to_df(
                stmt=compile_sql(cluster_expansion_stmt),
                connection=conn.dbapi_connection,
                return_type="arrow",
            )

    # Do a bit of casting to conform to data transfer schema
    for i, col in enumerate(["user_id", "endorsed", "shown"]):
        judgements = judgements.set_column(i, col, judgements[col].cast(pa.uint64()))

    cluster_expansion = cluster_expansion.set_column(
        0, "root", cluster_expansion["root"].cast(pa.uint64())
    )
    cluster_expansion = cluster_expansion.set_column(
        1, "leaves", cluster_expansion["leaves"].cast(pa.list_(pa.uint64()))
    )

    return judgements, cluster_expansion


def sample(n: int, resolution_path: ModelResolutionPath, user_id: int):
    """Sample some clusters from a resolution."""
    # Not currently checking validity of the user_id
    # If the user ID does not exist, the exclusion by previous judgements breaks
    if n > 100:
        # This reasonable assumption means simple "IS IN" function later is fine
        raise ValueError("Can only sample 100 entries at a time.")

    with MBDB.get_session() as session:
        # Use ORM to get resolution metadata
        resolution_orm = Resolutions.from_path(path=resolution_path, session=session)
        resolution_id = resolution_orm.resolution_id
        truth = resolution_orm.truth

    # Get a list of cluster IDs and features for this resolution and user
    user_judgements = (
        select(EvalJudgements).where(EvalJudgements.user_id == user_id).subquery()
    )
    cluster_features_stmt = (
        select(
            Probabilities.cluster_id,
            # We expect only one probability per cluster within one resolution
            func.max(Probabilities.probability).label("probability"),
            func.max(user_judgements.c.timestamp).label("latest_ts"),
        )
        .join(
            user_judgements,
            Probabilities.cluster_id == user_judgements.c.shown_cluster_id,
            isouter=True,
        )
        .where(
            Probabilities.resolution_id == resolution_id,
        )
        .group_by(Probabilities.cluster_id)
    )

    with MBDB.get_adbc_connection() as conn:
        cluster_features = sql_to_df(
            stmt=compile_sql(cluster_features_stmt),
            connection=conn.dbapi_connection,
            return_type="polars",
        )

    # Exclude clusters recently judged by this user
    to_sample = cluster_features.filter(
        (pl.col("latest_ts") < datetime.now(UTC) - timedelta(days=365))
        | (pl.col("latest_ts").is_null())
    )

    # Return early if nothing to sample from
    if not len(to_sample):
        return pl.DataFrame(
            {"root": [], "leaf": [], "key": [], "source": []},
            schema=pl.Schema(SCHEMA_EVAL_SAMPLES),
        )

    # Sample proportionally to distance from the truth, and get 1D array
    distances = np.abs(to_sample.select("probability").to_numpy() - truth)[:, 0]
    # Add small noise to avoid division by 0 if all distances are 0
    unnormalised_probs = distances + 0.001
    probs = unnormalised_probs / unnormalised_probs.sum()

    # With fewer clusters than requested, return all
    if to_sample.shape[0] <= n:
        sampled_cluster_ids = to_sample.select("cluster_id").to_series().to_list()
    else:
        indices = np.random.choice(to_sample.shape[0], size=n, p=probs, replace=False)
        sampled_cluster_ids = (
            to_sample[indices].select("cluster_id").to_series().to_list()
        )

    # Get all info we need for the cluster IDs we've sampled, i.e.:
    # source cluster IDs, keys and source resolutions
    with MBDB.get_adbc_connection() as conn:
        source_clusters = (
            select(Contains.root, Contains.leaf)
            .where(Contains.root.in_(sampled_cluster_ids))
            .subquery()
        )
        enrich_stmt = (
            select(
                source_clusters.c.root,
                source_clusters.c.leaf,
                ClusterSourceKey.key,
                Resolutions.name.label("source"),
            )
            .select_from(source_clusters)
            .join(
                ClusterSourceKey,
                ClusterSourceKey.cluster_id == source_clusters.c.leaf,
            )
            .join(
                SourceConfigs,
                SourceConfigs.source_config_id == ClusterSourceKey.source_config_id,
            )
            .join(
                Resolutions,
                Resolutions.resolution_id == SourceConfigs.resolution_id,
            )
        )

        final_samples = sql_to_df(
            stmt=compile_sql(enrich_stmt),
            connection=conn.dbapi_connection,
            return_type="polars",
        )
        return final_samples.with_columns(
            [pl.col("root").cast(pl.UInt64), pl.col("leaf").cast(pl.UInt64)]
        ).to_arrow()


def compare_models(
    resolutions: list[ModelResolutionPath],
    judgements: pl.DataFrame,
    expansion: pl.DataFrame,
):
    """Compare models on the basis of precision and recall."""

    def get_root_leaf(resolution_path: ModelResolutionPath) -> Table:
        with MBDB.get_session() as session:
            resolution = Resolutions.from_path(resolution_path, session=session)
            # The session which fetched the resolution needs to be alive while
            # the root-leaf query is built
            root_leaf_query = build_unified_query(
                resolution, threshold=resolution.truth
            )

        with MBDB.get_adbc_connection() as conn:
            root_leaf_results = sql_to_df(
                stmt=compile_sql(root_leaf_query),
                connection=conn.dbapi_connection,
                return_type="arrow",
            )
            return root_leaf_results.rename_columns(
                {"root_id": "root", "leaf_id": "leaf"}
            ).select(["root", "leaf"])

    models_root_leaf = [pl.from_arrow(get_root_leaf(res)) for res in resolutions]

    pr_values = precision_recall(
        models_root_leaf=models_root_leaf,
        judgements=judgements,
        expansion=expansion,
    )

    return dict(zip(resolutions, pr_values, strict=True))
