"""Evaluation PostgreSQL mixin for Matchbox server."""

from datetime import UTC, datetime, timedelta
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow as pa
from pyarrow import Table
from sqlalchemy import BIGINT, func, select
from sqlalchemy.exc import IntegrityError

from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES_DOWNLOAD,
    SCHEMA_JUDGEMENTS,
)
from matchbox.common.db import sql_to_df
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.eval import Judgement as CommonJudgement
from matchbox.common.exceptions import (
    MatchboxTooManySamplesRequested,
    MatchboxUserNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.common.transform import Cluster, IntMap, hash_cluster_leaves
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Collections,
    Contains,
    EvalJudgements,
    EvalSamples,
    EvalSampleSets,
    PKSpace,
    Probabilities,
    Resolutions,
    SourceConfigs,
    Users,
)
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    ingest_to_temporary_table,
    large_append,
)
from matchbox.server.postgresql.utils.insert import (
    create_clusters_dataframe,
    make_model_cluster_tables,
)

if TYPE_CHECKING:
    from pyarrow import Table as ArrowTable
else:
    ArrowTable = Any


class MatchboxPostgresEvaluationMixin:
    """Evaluation mixin for the PostgreSQL adapter for Matchbox."""

    def insert_samples(  # noqa: D102
        self,
        samples: Table,
        name: str,
        collection: str,
        description: str | None = None,
    ) -> None:
        log_prefix = f"Sample set {name}"

        if not samples.num_rows:
            logger.info("No samples to add.", prefix=log_prefix)
            return

        with MBDB.get_session() as session:
            # Generate new sample set ORM object
            collection_id = Collections.from_name(collection, session).collection_id
            sample_set = EvalSampleSets(
                name=name, description=description, collection_id=collection_id
            )

            # Find all leaf IDs referenced
            samples_pl = pl.from_arrow(samples)
            leaf_ids = samples_pl["leaf"].unique().to_list()

            # Find cluster hashes for all leaf IDs
            with ingest_to_temporary_table(
                table_name="leaves",
                schema_name="mb",
                column_types={"cluster_id": BIGINT},
                data=pl.DataFrame({"cluster_id": leaf_ids}).to_arrow(),
            ) as temp_table:
                leaf_id_hash_stmt = select(
                    Clusters.cluster_id, Clusters.cluster_hash
                ).join(temp_table, temp_table.c.cluster_id == Clusters.cluster_id)

                with MBDB.get_adbc_connection() as conn:
                    leaf_id_hash: pl.DataFrame = sql_to_df(
                        stmt=compile_sql(leaf_id_hash_stmt),
                        connection=conn.dbapi_connection,
                        return_type="polars",
                    )

            # Create cluster objects for the leaves
            all_leaf_clusters: dict[int, Cluster] = {}
            im = IntMap()
            for leaf in leaf_id_hash.iter_rows(named=True):
                all_leaf_clusters[leaf["cluster_id"]] = Cluster(
                    intmap=im, id=leaf["cluster_id"], hash=leaf["cluster_hash"]
                )

            # Create cluster objects for the roots
            new_clusters = []
            root_id_to_hash = []
            for cluster_partition in samples_pl.partition_by(by="root"):
                leaf_clusters = [
                    all_leaf_clusters[row["leaf"]]
                    for row in cluster_partition.iter_rows(named=True)
                ]
                new_cluster = Cluster.combine(leaf_clusters)
                new_clusters.append(new_cluster)
                root_id_to_hash.append(
                    {
                        "cluster_hash": new_cluster.hash,
                        "root": cluster_partition["root"][0],
                    }
                )

            # Generate tables to insert to define new clusters
            new_clusters_df = create_clusters_dataframe(new_clusters)
            clusters_table, contains_table = make_model_cluster_tables(new_clusters_df)
            del new_clusters

            # Find mapping between placeholder (negative) cluster IDs and real IDs
            root_id_to_hash_df = pl.DataFrame(root_id_to_hash)
            root_to_cluster_id = root_id_to_hash_df.join(
                new_clusters_df, on="cluster_hash", how="left"
            ).select("cluster_id", "root")

            try:
                # Tentatively add sample set to DB
                session.add(sample_set)
                session.commit()
            except IntegrityError as e:
                raise ValueError(
                    "Sample with this name already exists in collection"
                ) from e
            logger.info("Created sample set", prefix=log_prefix)

            # Prepare samples table for insertion
            samples_table = (
                samples_pl.select("root", "weight")
                .unique()
                .join(root_to_cluster_id, on="root")
                .select("cluster_id", "weight")
                .with_columns(pl.lit(sample_set.sample_set_id).alias("sample_set_id"))
                .cast({"weight": pl.UInt8, "sample_set_id": pl.Int64})
            )

            # Insert dataframes
            with MBDB.get_adbc_connection() as adbc_connection:
                try:
                    large_append(
                        data=clusters_table.to_arrow(),
                        table_class=Clusters,
                        adbc_connection=adbc_connection,
                        max_chunksize=self.settings.batch_size,
                    )

                    logger.info(
                        f"Added {len(clusters_table):,} rows to Clusters table",
                        prefix=log_prefix,
                    )

                    large_append(
                        data=contains_table.to_arrow(),
                        table_class=Contains,
                        adbc_connection=adbc_connection,
                        max_chunksize=self.settings.batch_size,
                    )

                    logger.info(
                        f"Added {len(contains_table):,} rows to Contains table",
                        prefix=log_prefix,
                    )

                    large_append(
                        data=samples_table.to_arrow(),
                        table_class=EvalSamples,
                        adbc_connection=adbc_connection,
                        max_chunksize=self.settings.batch_size,
                    )

                    logger.info(
                        f"Added {len(samples_table):,} rows to EvalSamples table",
                        prefix=log_prefix,
                    )
                    adbc_connection.commit()
                except:
                    adbc_connection.rollback()
                    session.delete(sample_set)
                    session.commit()
                    logger.error(
                        "Deleting sample set, as adding samples generated an error"
                    )
                    raise

            logger.info("Insertion complete", prefix=log_prefix)

    def insert_judgement(self, judgement: CommonJudgement) -> None:  # noqa: D102
        # Check that all referenced cluster IDs exist
        ids = list(chain(*judgement.endorsed)) + [judgement.shown]
        self.validate_ids(ids)

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
                        select(Clusters.cluster_hash).where(
                            Clusters.cluster_id == leaf_id
                        )
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
                        sample_set_id=judgement.sample_set,
                        shown_cluster_id=judgement.shown,
                        endorsed_cluster_id=endorsed_cluster_id,
                        timestamp=datetime.now(UTC),
                    )
                )

                session.commit()

    def get_judgements(self) -> tuple[Table, Table]:  # noqa: D102
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
            EvalJudgements.user_id,
            EvalJudgements.endorsed_cluster_id.label("endorsed"),
            EvalJudgements.shown_cluster_id.label("shown"),
        )

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
                    return_type="polars",
                )

        return _cast_tables(judgements, cluster_expansion)

    def sample_for_eval(  # noqa: D102
        self, n: int, path: ModelResolutionPath, user_id: int
    ) -> ArrowTable:
        # Not currently checking validity of the user_id
        # If the user ID does not exist, the exclusion by previous judgements breaks
        if n > 100:
            # This reasonable assumption means simple "IS IN" function later is fine
            raise MatchboxTooManySamplesRequested(
                "Can only sample 100 entries at a time."
            )

        with MBDB.get_session() as session:
            # Use ORM to get resolution metadata
            resolution_orm = Resolutions.from_path(path=path, session=session)
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
                schema=pl.Schema(SCHEMA_EVAL_SAMPLES_DOWNLOAD)
            ).to_arrow()

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
            return final_samples.cast(
                pl.Schema(SCHEMA_EVAL_SAMPLES_DOWNLOAD)
            ).to_arrow()
