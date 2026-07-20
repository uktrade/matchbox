"""Composed PostgreSQL adapter for Matchbox server."""

from pydantic import BaseModel
from sqlalchemy import func, or_

from matchbox.common.dtos import StepType
from matchbox.server.base import MatchboxDBAdapter
from matchbox.server.postgresql.adapter.admin import MatchboxPostgresAdminMixin
from matchbox.server.postgresql.adapter.collections import (
    MatchboxPostgresCollectionsMixin,
)
from matchbox.server.postgresql.adapter.eval import MatchboxPostgresEvaluationMixin
from matchbox.server.postgresql.adapter.groups import MatchboxPostgresGroupsMixin
from matchbox.server.postgresql.adapter.query import MatchboxPostgresQueryMixin
from matchbox.server.postgresql.db import MBDB, MatchboxPostgresSettings
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    EvalJudgements,
    Groups,
    ModelConfigs,
    ModelEdges,
    ResolverClusters,
    ResolverConfigs,
    SourceConfigs,
    Steps,
    Users,
)


class FilteredClusters(BaseModel):
    """Wrapper class for filtered cluster queries."""

    has_source: bool | None = None

    def count(self) -> int:
        """Counts the number of clusters in the database."""
        with MBDB.get_session() as session:
            query = session.query(
                func.count(func.distinct(Clusters.cluster_id))
            ).select_from(Clusters)

            if self.has_source is not None:
                if self.has_source:
                    query = query.join(
                        ClusterSourceKey,
                        ClusterSourceKey.cluster_id == Clusters.cluster_id,
                    )
                else:
                    query = (
                        query.join(
                            ResolverClusters,
                            ResolverClusters.cluster_id == Clusters.cluster_id,
                            isouter=True,
                        )
                        .join(
                            EvalJudgements,
                            EvalJudgements.endorsed_cluster_id == Clusters.cluster_id,
                            isouter=True,
                        )
                        .filter(
                            or_(
                                EvalJudgements.endorsed_cluster_id.is_not(None),
                                ResolverClusters.cluster_id.is_not(None),
                            )
                        )
                    )

            return query.scalar()


class FilteredProbabilities(BaseModel):
    """Wrapper class for filtered model edge queries."""

    def count(self) -> int:
        """Counts the number of model edges in the database."""
        with MBDB.get_session() as session:
            return session.query(func.count()).select_from(ModelEdges).scalar()


class FilteredSteps(BaseModel):
    """Wrapper class for filtered step queries."""

    sources: bool = False
    models: bool = False
    resolvers: bool = False

    def count(self) -> int:
        """Counts the number of steps in the database."""
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Steps)

            filter_list = []
            if self.sources:
                filter_list.append(Steps.type == StepType.SOURCE)
            if self.models:
                filter_list.append(Steps.type == StepType.MODEL)
            if self.resolvers:
                filter_list.append(Steps.type == StepType.RESOLVER)

            if filter_list:
                query = query.filter(or_(*filter_list))

            return query.scalar()


class MatchboxPostgres(
    MatchboxPostgresQueryMixin,
    MatchboxPostgresEvaluationMixin,
    MatchboxPostgresCollectionsMixin,
    MatchboxPostgresAdminMixin,
    MatchboxPostgresGroupsMixin,
    MatchboxDBAdapter,
):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxPostgresSettings) -> None:
        """Initialise the PostgreSQL adapter."""
        self.settings = settings
        MBDB.settings = settings
        MBDB.run_migrations()

        Groups.initialise()
        self.sources = SourceConfigs
        self.models = ModelConfigs
        self.resolvers = ResolverConfigs
        self.source_clusters = FilteredClusters(has_source=True)
        self.model_clusters = FilteredClusters(has_source=False)
        self.all_clusters = FilteredClusters()
        self.creates = ResolverClusters  # clusters proposed by resolver
        self.merges = Contains  # cluster lineage tree
        self.proposes = FilteredProbabilities()  # raw model edge scores
        self.source_steps = FilteredSteps(sources=True, models=False, resolvers=False)
        self.users = Users
