"""PostgreSQL adapter for Matchbox server."""

from itertools import chain
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pyarrow import Table
from pydantic import BaseModel
from sqlalchemy import and_, bindparam, delete, func, or_, select

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import Match, Resolution
from matchbox.common.eval import Judgement as CommonJudgement
from matchbox.common.eval import ModelComparison
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxNoJudgements,
)
from matchbox.common.graph import (
    ModelResolutionName,
    ResolutionGraph,
    ResolutionName,
    ResolutionType,
    SourceResolutionName,
)
from matchbox.common.logging import logger
from matchbox.server.base import MatchboxDBAdapter, MatchboxSnapshot
from matchbox.server.postgresql.db import (
    MBDB,
    MatchboxBackends,
    MatchboxPostgresSettings,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    EvalJudgements,
    PKSpace,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Results,
    SourceConfigs,
    Users,
)
from matchbox.server.postgresql.utils import evaluation
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    dump,
    get_resolution_graph,
    restore,
)
from matchbox.server.postgresql.utils.insert import (
    insert_hashes,
    insert_results,
)
from matchbox.server.postgresql.utils.query import match, query

T = TypeVar("T")
P = ParamSpec("P")

if TYPE_CHECKING:
    from pyarrow import Table as ArrowTable
else:
    ArrowTable = Any


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


class MatchboxPostgres(MatchboxDBAdapter):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxPostgresSettings):
        """Initialise the PostgreSQL adapter."""
        self.settings = settings
        MBDB.settings = settings
        MBDB.run_migrations()

        PKSpace.initialise()

        self.sources = SourceConfigs
        self.models = FilteredResolutions(sources=False, models=True)
        self.data = FilteredClusters(has_source=True)
        self.clusters = FilteredClusters(has_source=False)
        self.creates = FilteredProbabilities(over_truth=True)
        self.merges = Contains
        self.proposes = FilteredProbabilities()
        self.source_resolutions = FilteredResolutions(sources=True, models=False)

    # Retrieval

    def query(  # noqa: D102
        self,
        source: SourceResolutionName,
        resolution: ResolutionName | None = None,
        threshold: int | None = None,
        return_leaf_id: bool = False,
        limit: int | None = None,
    ) -> ArrowTable:
        return query(
            source=source,
            resolution=resolution,
            threshold=threshold,
            return_leaf_id=return_leaf_id,
            limit=limit,
        )

    def match(  # noqa: D102
        self,
        key: str,
        source: SourceResolutionName,
        targets: list[SourceResolutionName],
        resolution: ResolutionName,
        threshold: int | None = None,
    ) -> list[Match]:
        return match(
            key=key,
            source=source,
            targets=targets,
            resolution=resolution,
            threshold=threshold,
        )

    # Resolution management

    def insert_resolution(self, resolution: Resolution) -> None:  # noqa: D102
        log_prefix = f"Insert {resolution.name}"
        with MBDB.get_session() as session:
            resolution_orm = Resolutions.from_dto(resolution, session)
            session.commit()

            logger.info(
                f"Inserted with ID {resolution_orm.resolution_id}", prefix=log_prefix
            )

    def get_resolution(  # noqa: D102
        self, name: ResolutionName, validate: ResolutionType | None = None
    ) -> Resolution:
        with MBDB.get_session() as session:
            resolution = Resolutions.from_name(
                name=name, res_type=validate, session=session
            )
            return resolution.to_dto()

    def delete_resolution(self, name: ResolutionName, certain: bool) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            resolution = Resolutions.from_name(name=name, session=session)
            if certain:
                delete_stmt = delete(Resolutions).where(
                    Resolutions.resolution_id.in_(
                        [
                            resolution.resolution_id,
                            *(d.resolution_id for d in resolution.descendants),
                        ]
                    )
                )
                session.execute(delete_stmt)
                session.commit()
            else:
                children = [r.name for r in resolution.descendants]
                raise MatchboxDeletionNotConfirmed(children=children)

    def get_leaf_source_resolutions(self, name: ResolutionName) -> list[Resolution]:  # noqa: D102
        with MBDB.get_session() as session:
            resolution = Resolutions.from_name(
                name=name, res_type=ResolutionType.MODEL, session=session
            )

            source_resolutions = (
                session.query(Resolutions)
                .join(
                    ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.parent
                )
                .filter(
                    ResolutionFrom.child == resolution.resolution_id,
                    Resolutions.type == ResolutionType.SOURCE.value,
                )
                .all()
            )

            return [r.to_dto() for r in source_resolutions]

    def get_resolution_graph(self) -> ResolutionGraph:  # noqa: D102
        return get_resolution_graph()

    # Data insertion

    def insert_source_data(  # noqa: D102
        self, name: SourceResolutionName, data_hashes: Table
    ) -> None:
        insert_hashes(
            name=name,
            data_hashes=data_hashes,
            batch_size=self.settings.batch_size,
        )

    def insert_model_data(self, name: ModelResolutionName, results: Table) -> None:  # noqa: D102
        insert_results(
            name=name,
            results=results,
            batch_size=self.settings.batch_size,
        )

    def get_model_data(self, name: ModelResolutionName) -> Table:  # noqa: D102
        results_query = (
            select(Results.left_id, Results.right_id, Results.probability)
            .join(
                Resolutions,
                Results.resolution_id == Resolutions.resolution_id,
            )
            .where(Resolutions.name == name, Resolutions.type == ResolutionType.MODEL)
        )
        with MBDB.get_adbc_connection() as conn:
            stmt: str = compile_sql(results_query)
            return sql_to_df(
                stmt=stmt, connection=conn.dbapi_connection, return_type="arrow"
            )

    def set_model_truth(self, name: ModelResolutionName, truth: int) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            resolution = Resolutions.from_name(
                name=name, res_type=ResolutionType.MODEL, session=session
            )
            resolution.truth = truth
            session.commit()

    def get_model_truth(self, name: ModelResolutionName) -> int:  # noqa: D102
        resolution = Resolutions.from_name(name=name, res_type=ResolutionType.MODEL)
        return resolution.truth

    # Data management

    def validate_ids(self, ids: list[int]) -> bool:  # noqa: D102
        with MBDB.get_session() as session:
            data_inner_join = (
                session.query(Clusters)
                .filter(
                    Clusters.cluster_id.in_(
                        bindparam(
                            "ins_ids",
                            ids,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        existing_ids = {item.cluster_id for item in data_inner_join}
        missing_ids = set(ids) - existing_ids

        if missing_ids:
            raise MatchboxDataNotFound(
                message="Some items don't exist in Clusters table.",
                table=Clusters.__tablename__,
                data=missing_ids,
            )

        return True

    def validate_hashes(self, hashes: list[bytes]) -> bool:  # noqa: D102
        with MBDB.get_session() as session:
            data_inner_join = (
                session.query(Clusters)
                .filter(
                    Clusters.cluster_hash.in_(
                        bindparam(
                            "ins_hashs",
                            hashes,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        existing_hashes = {item.cluster_hash for item in data_inner_join}
        missing_hashes = set(hashes) - existing_hashes

        if missing_hashes:
            raise MatchboxDataNotFound(
                message="Some items don't exist in Clusters table.",
                table=Clusters.__tablename__,
                data=missing_hashes,
            )

        return True

    def cluster_id_to_hash(self, ids: list[int]) -> dict[int, bytes | None]:  # noqa: D102
        initial_dict = {id: None for id in ids}

        with MBDB.get_session() as session:
            data_inner_join = (
                session.query(Clusters)
                .filter(
                    Clusters.cluster_id.in_(
                        bindparam(
                            "ins_ids",
                            ids,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        return initial_dict | {
            item.cluster_id: item.cluster_hash for item in data_inner_join
        }

    def dump(self) -> MatchboxSnapshot:  # noqa: D102
        return dump()

    def drop(self, certain: bool) -> None:  # noqa: D102
        if certain:
            MBDB.drop_database()
            PKSpace.initialise()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database and recreate it."
                "It's not expected to be used as part normal operations."
                "If you're sure you want to continue, rerun with certain=True"
            )

    def clear(self, certain: bool) -> None:  # noqa: D102
        if certain:
            MBDB.clear_database()
            PKSpace.initialise()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop all rows in the database but not the "
                "tables themselves. It's primarily used to reset following tests."
                "If you're sure you want to continue, rerun with certain=True"
            )

    def restore(self, snapshot: MatchboxSnapshot) -> None:  # noqa: D102
        if snapshot.backend_type != MatchboxBackends.POSTGRES:
            raise TypeError(
                f"Cannot restore {snapshot.backend_type} snapshot to PostgreSQL backend"
            )

        MBDB.clear_database()

        restore(
            snapshot=snapshot,
            batch_size=self.settings.batch_size,
        )

    # User management

    def login(self, user_name: str) -> int:  # noqa: D102
        with MBDB.get_session() as session:
            if user_id := session.scalar(
                select(Users.user_id).where(Users.name == user_name)
            ):
                return user_id

            user = Users(name=user_name)
            session.add(user)
            session.commit()

            return user.user_id

    # Evaluation management

    def insert_judgement(self, judgement: CommonJudgement) -> None:  # noqa: D102
        # Check that all referenced cluster IDs exist
        ids = list(chain(*judgement.endorsed)) + [judgement.shown]
        self.validate_ids(ids)
        evaluation.insert_judgement(judgement)

    def get_judgements(self) -> tuple[Table, Table]:  # noqa: D102
        return evaluation.get_judgements()

    def compare_models(self, resolutions: list[ModelResolutionName]) -> ModelComparison:  # noqa: D102
        judgements, expansion = self.get_judgements()
        if not len(judgements):
            raise MatchboxNoJudgements()
        return evaluation.compare_models(resolutions, judgements, expansion)

    def sample_for_eval(  # noqa: D102
        self, n: int, resolution: ModelResolutionName, user_id: int
    ) -> ArrowTable:
        return evaluation.sample(n, resolution, user_id)
