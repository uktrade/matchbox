"""Composed PostgreSQL adapter for Matchbox server."""

from pydantic import BaseModel
from sqlalchemy import and_, func, or_

from matchbox.common.dtos import ResolutionType
from matchbox.server.base import MatchboxDBAdapter
from matchbox.server.postgresql.adapter.admin import MatchboxPostgresAdminMixin
from matchbox.server.postgresql.adapter.collections import (
    MatchboxPostgresCollectionsMixin,
)
from matchbox.server.postgresql.adapter.eval import MatchboxPostgresEvaluationMixin
from matchbox.server.postgresql.adapter.query import MatchboxPostgresQueryMixin
from matchbox.server.postgresql.db import MBDB, MatchboxPostgresSettings
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    EvalJudgements,
    Probabilities,
    Resolutions,
    SourceConfigs,
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
                            Probabilities,
                            Probabilities.cluster_id == Clusters.cluster_id,
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
                                Probabilities.cluster_id.is_not(None),
                            )
                        )
                    )

            return query.scalar()


class FilteredProbabilities(BaseModel):
    """Wrapper class for filtered probability queries."""

    over_truth: bool = False

    def count(self) -> int:
        """Counts the number of probabilities in the database."""
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Probabilities)

            if self.over_truth:
                query = query.join(
                    Resolutions,
                    Probabilities.resolution_id == Resolutions.resolution_id,
                ).filter(
                    and_(
                        Resolutions.truth.isnot(None),
                        Probabilities.probability >= Resolutions.truth,
                    )
                )
            return query.scalar()


class FilteredResolutions(BaseModel):
    """Wrapper class for filtered resolution queries."""

    sources: bool = False
    models: bool = False

    def count(self) -> int:
        """Counts the number of resolutions in the database."""
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Resolutions)

            filter_list = []
            if self.sources:
                filter_list.append(Resolutions.type == ResolutionType.SOURCE)
            if self.models:
                filter_list.append(Resolutions.type == ResolutionType.MODEL)

            if filter_list:
                query = query.filter(or_(*filter_list))

            return query.scalar()


class MatchboxPostgres(
    MatchboxPostgresQueryMixin,
    MatchboxPostgresEvaluationMixin,
    MatchboxPostgresCollectionsMixin,
    MatchboxPostgresAdminMixin,
    MatchboxDBAdapter,
):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxPostgresSettings) -> None:
        """Initialise the PostgreSQL adapter."""
        self.settings = settings
        MBDB.settings = settings
        MBDB.run_migrations()

        self.sources = SourceConfigs
        self.models = FilteredResolutions(sources=False, models=True)
        self.source_clusters = FilteredClusters(has_source=True)
        self.model_clusters = FilteredClusters(has_source=False)
        self.all_clusters = FilteredClusters()
        self.creates = FilteredProbabilities(over_truth=True)
        self.merges = Contains
        self.proposes = FilteredProbabilities()
        self.source_resolutions = FilteredResolutions(sources=True, models=False)
