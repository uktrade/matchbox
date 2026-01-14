"""Evaluation PostgreSQL mixin for Matchbox server."""

from datetime import UTC, datetime, timedelta
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import Table
from sqlalchemy import BIGINT, func, select
from sqlalchemy.orm import Session

from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES,
    SCHEMA_JUDGEMENTS,
)
from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.eval import Judgement as CommonJudgement
from matchbox.common.exceptions import (
    MatchboxTooManySamplesRequested,
    MatchboxUserNotFoundError,
)
from matchbox.common.transform import hash_cluster_leaves
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    EvalJudgements,
    Probabilities,
    Resolutions,
    SourceConfigs,
    Users,
)
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    ingest_to_temporary_table,
)

if TYPE_CHECKING:
    from pyarrow import Table as ArrowTable
else:
    ArrowTable = Any


class MatchboxPostgresEvaluationMixin:
    """Evaluation mixin for the PostgreSQL adapter for Matchbox."""

    def insert_judgement(self, judgement: CommonJudgement) -> None:  # noqa: D102
        def get_or_create_cluster(leaves: list[int], session: Session) -> int:
            # Compute hash corresponding to set of source clusters (leaves)
            leaf_hashes = [
                session.scalar(
                    select(Clusters.cluster_hash).where(Clusters.cluster_id == leaf_id)
                )
                for leaf_id in leaves
            ]
            cluster_hash = hash_cluster_leaves(leaf_hashes)

            # Add new cluster to session, if it doesn't exist already
            if not (
                cluster_id := session.scalar(
                    select(Clusters.cluster_id).where(
                        Clusters.cluster_hash == cluster_hash
                    )
                )
            ):
                new_c = Clusters(cluster_hash=cluster_hash)
                session.add(new_c)
                session.flush()
                cluster_id = new_c.cluster_id
                for leaf_id in leaves:
                    session.add(Contains(root=cluster_id, leaf=leaf_id))

            return cluster_id

        # Check that all referenced leaf IDs exist
        ids = list(chain(*judgement.endorsed)) + judgement.shown
        self.validate_ids(ids)

        # Check that the user exists
        with MBDB.get_session() as session:
            user = session.scalar(
                select(Users).where(Users.name == judgement.user_name)
            )
            if not user:
                raise MatchboxUserNotFoundError(
                    f"User '{judgement.user_name}' not found"
                )

        with MBDB.get_session() as session:
            shown_id = get_or_create_cluster(judgement.shown, session)
            session.commit()

            for leaves in judgement.endorsed:
                # If cluster with this hash does not exist, create it.
                # Note that only endorsed clusters might be new. The cluster shown to
                # the user is guaranteed to exist in the backend; we have checked above.

                session.add(
                    EvalJudgements(
                        user_id=user.user_id,
                        tag=judgement.tag,
                        shown_cluster_id=shown_id,
                        endorsed_cluster_id=get_or_create_cluster(leaves, session),
                        timestamp=datetime.now(UTC),
                    )
                )
                session.commit()

    def get_judgements(self, tag: str | None = None) -> tuple[Table, Table]:  # noqa: D102
        def _cast_tables(
            judgements: pl.DataFrame, cluster_expansion: pl.DataFrame
        ) -> tuple[pa.Table, pa.Table]:
            """Cast judgement tables to conform to data transfer schema."""
            judgements = judgements.cast(pl.Schema(SCHEMA_JUDGEMENTS))
            cluster_expansion = cluster_expansion.cast(
                pl.Schema(SCHEMA_CLUSTER_EXPANSION)
            )

            return (
                judgements.to_arrow().cast(SCHEMA_JUDGEMENTS),
                cluster_expansion.to_arrow().cast(SCHEMA_CLUSTER_EXPANSION),
            )

        judgements_stmt = select(
            Users.name.label("user_name"),
            EvalJudgements.endorsed_cluster_id.label("endorsed"),
            EvalJudgements.shown_cluster_id.label("shown"),
        ).join(Users)

        if tag:
            judgements_stmt = judgements_stmt.where(EvalJudgements.tag == tag)

        with MBDB.get_adbc_connection() as conn:
            judgements = sql_to_df(
                stmt=compile_sql(judgements_stmt),
                connection=conn.dbapi_connection,
                return_type="polars",
            )

        if not len(judgements):
            cluster_expansion = pl.DataFrame(schema=pl.Schema(SCHEMA_CLUSTER_EXPANSION))
            return _cast_tables(judgements, cluster_expansion)

        shown_clusters = set(judgements["shown"].to_list())
        endorsed_clusters = set(judgements["endorsed"].to_list())
        referenced_clusters = Table.from_pydict(
            {"root": list(shown_clusters | endorsed_clusters)}
        )

        with ingest_to_temporary_table(
            table_name="judgements",
            schema_name="mb",
            column_types={
                "root": BIGINT(),
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
                    return_type="polars",
                )

        return _cast_tables(judgements, cluster_expansion)

    def sample_for_eval(  # noqa: D102
        self, n: int, path: ModelResolutionPath, user_name: str
    ) -> ArrowTable:
        """Sample some clusters from a resolution."""
        # Not currently checking validity of the user_name
        # If the user ID does not exist, the exclusion by previous judgements breaks
        if n > 100:
            # This reasonable assumption means simple "IS IN" function later is fine
            raise MatchboxTooManySamplesRequested(
                "Can only sample 100 entries at a time."
            )

        with MBDB.get_session() as session:
            # Get user
            user = session.scalar(select(Users).where(Users.name == user_name))
            if not user:
                raise MatchboxUserNotFoundError(f"User '{user_name}' not found")

            # Use ORM to get resolution metadata
            resolution_orm = Resolutions.from_path(path=path, session=session)
            resolution_id = resolution_orm.resolution_id
            truth = resolution_orm.truth

        # Get a list of cluster IDs and features for this resolution and user
        user_judgements = (
            select(EvalJudgements)
            .where(EvalJudgements.user_id == user.user_id)
            .subquery()
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
            return pl.DataFrame(schema=pl.Schema(SCHEMA_EVAL_SAMPLES)).to_arrow()

        # Sample proportionally to distance from the truth, and get 1D array
        distances = np.abs(to_sample.select("probability").to_numpy() - truth)[:, 0]
        # Add small noise to avoid division by 0 if all distances are 0
        unnormalised_probs = distances + 0.001
        probs = unnormalised_probs / unnormalised_probs.sum()

        # With fewer clusters than requested, return all
        if to_sample.shape[0] <= n:
            sampled_cluster_ids = to_sample.select("cluster_id").to_series().to_list()
        else:
            indices = np.random.choice(
                to_sample.shape[0], size=n, p=probs, replace=False
            )
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

            # The same leaf can be reused to represent rows across different sources
            # We only want to retrieve info for sources upstream of resolution
            source_resolution_ids = [
                res.resolution_id
                for res in resolution_orm.ancestors
                if res.type == "source"
            ]
            source_resolutions = (
                select(Resolutions.name, Resolutions.resolution_id)
                .where(Resolutions.resolution_id.in_(source_resolution_ids))
                .subquery()
            )

            enrich_stmt = (
                select(
                    source_clusters.c.root,
                    source_clusters.c.leaf,
                    ClusterSourceKey.key,
                    source_resolutions.c.name.label("source"),
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
                    source_resolutions,
                    source_resolutions.c.resolution_id == SourceConfigs.resolution_id,
                )
            )

            final_samples = sql_to_df(
                stmt=compile_sql(enrich_stmt),
                connection=conn.dbapi_connection,
                return_type="polars",
            )
            return final_samples.cast(pl.Schema(SCHEMA_EVAL_SAMPLES)).to_arrow()
