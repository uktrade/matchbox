"""PostgreSQL adapter for Matchbox server."""

from itertools import chain
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import polars as pl
from pyarrow import Table
from pydantic import BaseModel
from sqlalchemy import and_, bindparam, delete, func, or_, select, update

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    Collection,
    CollectionName,
    Match,
    ModelResolutionPath,
    Resolution,
    ResolutionPath,
    ResolutionType,
    Run,
    RunID,
    SourceResolutionPath,
    UploadStage,
)
from matchbox.common.dtos import ModelConfig as CommonModelConfig
from matchbox.common.dtos import (
    SourceConfig as CommonSourceConfig,
)
from matchbox.common.eval import Judgement as CommonJudgement
from matchbox.common.eval import ModelComparison
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxNoJudgements,
    MatchboxResolutionUpdateError,
    MatchboxRunNotWriteable,
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
    Collections,
    Contains,
    EvalJudgements,
    ModelConfigs,
    PKSpace,
    Probabilities,
    Resolutions,
    Results,
    Runs,
    SourceConfigs,
    SourceFields,
    Users,
)
from matchbox.server.postgresql.utils import evaluation
from matchbox.server.postgresql.utils.db import (
    compile_sql,
    dump,
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

    def __init__(self, settings: MatchboxPostgresSettings) -> None:
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
        source: SourceResolutionPath,
        point_of_truth: ResolutionPath | None = None,
        threshold: int | None = None,
        return_leaf_id: bool = False,
        limit: int | None = None,
    ) -> ArrowTable:
        return query(
            source=source,
            point_of_truth=point_of_truth,
            threshold=threshold,
            return_leaf_id=return_leaf_id,
            limit=limit,
        )

    def match(  # noqa: D102
        self,
        key: str,
        source: SourceResolutionPath,
        targets: list[SourceResolutionPath],
        point_of_truth: ResolutionPath,
        threshold: int | None = None,
    ) -> list[Match]:
        return match(
            key=key,
            source=source,
            targets=targets,
            point_of_truth=point_of_truth,
            threshold=threshold,
        )

    # Collection management

    def create_collection(self, name: CollectionName) -> Collection:  # noqa: D102
        with MBDB.get_session() as session:
            if (session.query(Collections).filter_by(name=name).first()) is None:
                new_collection = Collections(name=name)
                session.add(new_collection)
                session.commit()
                return new_collection.to_dto()
            else:
                raise MatchboxCollectionAlreadyExists

    def get_collection(  # noqa: D102
        self, name: CollectionName
    ) -> Collection:
        with MBDB.get_session() as session:
            collection_orm = Collections.from_name(name, session)
            return collection_orm.to_dto()

    def list_collections(self) -> list[CollectionName]:  # noqa: D102
        with MBDB.get_session() as session:
            collections = (
                session.execute(select(Collections.name).order_by(Collections.name))
                .scalars()
                .all()
            )
            return list(collections)

    def delete_collection(self, name: CollectionName, certain: bool) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            collection_orm = Collections.from_name(name, session)

            if not certain:
                version_names = [v.name for v in collection_orm.versions]
                raise MatchboxDeletionNotConfirmed(children=version_names)

            session.execute(
                delete(Collections).where(
                    Collections.collection_id == collection_orm.collection_id
                )
            )
            session.commit()

    # Run management

    def create_run(self, collection: CollectionName) -> Run:  # noqa: D102
        with MBDB.get_session() as session:
            # Can raise MatchboxCollectionNotFoundError
            collection_orm = Collections.from_name(collection, session)

            new_run = Runs(
                collection_id=collection_orm.collection_id,
                is_mutable=True,
                is_default=False,
            )
            session.add(new_run)
            session.commit()

            return new_run.to_dto()

    def set_run_mutable(  # noqa: D102
        self, collection: CollectionName, run_id: RunID, mutable: bool
    ) -> Run:
        with MBDB.get_session() as session:
            run_orm = Runs.from_id(collection, run_id, session)
            run_orm.is_mutable = mutable
            session.commit()

            return run_orm.to_dto()

    def set_run_default(  # noqa: D102
        self, collection: CollectionName, run_id: RunID, default: bool
    ) -> Run:
        with MBDB.get_session() as session:
            run_orm = Runs.from_id(collection, run_id, session)
            if default:
                if run_orm.is_mutable:
                    raise ValueError("Cannot set as default a mutable run")
                # Unset any existing default run for the collection
                session.execute(
                    update(Runs)
                    .where(
                        Runs.collection_id == run_orm.collection_id,
                        Runs.is_default.is_(True),
                    )
                    .values(is_default=False)
                )

            run_orm.is_default = default
            session.commit()

            return run_orm.to_dto()

    def get_run(self, collection: CollectionName, run_id: RunID) -> Run:  # noqa: D102
        with MBDB.get_session() as session:
            run_orm = Runs.from_id(collection, run_id, session)
            return run_orm.to_dto()

    def delete_run(  # noqa: D102
        self, collection: CollectionName, run_id: RunID, certain: bool
    ) -> None:
        with MBDB.get_session() as session:
            run_orm = Runs.from_id(collection, run_id, session)

            if not certain:
                resolution_names = [res.name for res in run_orm.resolutions]
                raise MatchboxDeletionNotConfirmed(children=resolution_names)

            session.execute(delete(Runs).where(Runs.run_id == run_orm.run_id))
            session.commit()

    # Resolution management

    def _check_writeable(self, path: ResolutionPath) -> None:
        run = Runs.from_id(collection=path.collection, run_id=path.run)
        if not run.is_mutable:
            raise MatchboxRunNotWriteable(
                f"Version {path.run} in collection {path.collection} is immutable"
            )

    def create_resolution(  # noqa: D102
        self, resolution: Resolution, path: ResolutionPath
    ) -> None:
        self._check_writeable(path)
        log_prefix = f"Insert {path.name}"
        with MBDB.get_session() as session:
            resolution_orm = Resolutions.from_dto(
                resolution=resolution, path=path, session=session
            )
            session.commit()

            logger.info(
                f"Inserted with ID {resolution_orm.resolution_id}", prefix=log_prefix
            )

    def get_resolution(  # noqa: D102
        self, path: ResolutionPath
    ) -> Resolution:
        with MBDB.get_session() as session:
            resolution = Resolutions.from_path(path=path, session=session)
            return resolution.to_dto()

    def update_resolution(  # noqa: D102
        self, resolution: Resolution, path: ResolutionPath
    ) -> None:
        new_config = resolution.config
        with MBDB.get_session() as session:
            # Get current ORM entry
            old_resolution = Resolutions.from_path(path=path, session=session)
            # Check current ORM entry can be updated
            if old_resolution.fingerprint != resolution.fingerprint:
                raise MatchboxResolutionUpdateError(
                    "Cannot update resolution with non-matching fingerprint."
                )
            # The following condition also protects against change of resolution type:
            # sources must have 0 parents, models must have 1 or more
            if old_resolution.to_dto().config.parents != new_config.parents:
                raise MatchboxResolutionUpdateError(
                    "Cannot change parents of a resolution."
                )

            # Update top-level metadata
            old_resolution.description = resolution.description
            old_resolution.truth = resolution.truth

            # Update config
            if old_resolution.type == "source":
                old_config: SourceConfigs = old_resolution.source_config
                if not isinstance(new_config, CommonSourceConfig):
                    raise ValueError("Config for source resolution expected.")
                old_config.location_name = new_config.location_config.name
                old_config.location_type = str(new_config.location_config.type)
                old_config.extract_transform = new_config.extract_transform

                # Update source fields
                if (
                    old_config.to_dto().key_field != new_config.key_field
                    or old_config.to_dto().index_fields != new_config.index_fields
                ):
                    # If any field differs, delete and start again
                    old_config.fields.clear()
                    session.flush()

                    new_fields = [
                        SourceFields(
                            index=0,
                            name=new_config.key_field.name,
                            is_key=True,
                            type=new_config.key_field.type.value,
                        )
                    ]
                    new_fields.extend(
                        [
                            SourceFields(
                                index=idx + 1,
                                name=field.name,
                                is_key=False,
                                type=field.type.value,
                            )
                            for idx, field in enumerate(new_config.index_fields)
                        ]
                    )

                    old_config.fields = new_fields

            else:
                old_config: ModelConfigs = old_resolution.model_config
                if not isinstance(new_config, CommonModelConfig):
                    raise ValueError("Config for model resolution expected.")
                old_config.model_class = new_config.model_class
                old_config.model_settings = new_config.model_settings
                old_config.left_query = new_config.left_query.model_dump_json()
                old_config.right_query = (
                    None
                    if not new_config.right_query
                    else new_config.right_query.model_dump_json()
                )

            session.commit()

    def delete_resolution(self, path: ResolutionPath, certain: bool) -> None:  # noqa: D102
        self._check_writeable(path)
        with MBDB.get_session() as session:
            resolution = Resolutions.from_path(path=path, session=session)
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

    # Data insertion
    def lock_resolution_data(self, path: ResolutionPath) -> None:  # noqa: D102
        self._check_writeable(path)
        with MBDB.get_session() as session:
            # lock resolution so only one client can initiate the upload
            # will fail if already locked
            resolution = Resolutions.from_path(
                path=path, session=session, for_update=True
            )

            # check status
            if resolution.upload_stage == UploadStage.COMPLETE:
                # unlock row
                session.rollback()
                raise ValueError(
                    "Once set to complete, resolution data stage cannot be changed."
                )
            elif resolution.upload_stage == UploadStage.PROCESSING:
                # unlock row
                session.rollback()
                raise ValueError("Upload already being processed.")

            resolution.upload_stage = UploadStage.PROCESSING
            session.commit()

    def unlock_resolution_data(self, path: ResolutionPath) -> None:  # noqa: D102
        self._check_writeable(path)
        with MBDB.get_session() as session:
            resolution = Resolutions.from_path(
                path=path, session=session, for_update=True
            )
            resolution.upload_stage = UploadStage.READY
            session.commit()

    def get_resolution_stage(self, path: ResolutionPath) -> UploadStage:  # noqa: D102
        resolution = Resolutions.from_path(path)
        return UploadStage(resolution.upload_stage)

    def insert_source_data(  # noqa: D102
        self, path: SourceResolutionPath, data_hashes: Table
    ) -> None:
        self._check_writeable(path)
        insert_hashes(
            path=path, data_hashes=data_hashes, batch_size=self.settings.batch_size
        )
        # self.set_resolution_stage(path=path, stage=UploadStage.COMPLETE)

    def insert_model_data(self, path: ModelResolutionPath, results: Table) -> None:  # noqa: D102
        self._check_writeable(path)
        insert_results(path=path, results=results, batch_size=self.settings.batch_size)
        # self.set_resolution_stage(path=path, stage=UploadStage.COMPLETE)

    def get_model_data(self, path: ModelResolutionPath) -> Table:  # noqa: D102
        with MBDB.get_session() as session:
            resolution = Resolutions.from_path(
                path=path, res_type=ResolutionType.MODEL, session=session
            )

            results_query = select(
                Results.left_id, Results.right_id, Results.probability
            ).where(Results.resolution_id == resolution.resolution_id)

        with MBDB.get_adbc_connection() as conn:
            stmt: str = compile_sql(results_query)
            return sql_to_df(
                stmt=stmt, connection=conn.dbapi_connection, return_type="arrow"
            )

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

    def compare_models(self, paths: list[ModelResolutionPath]) -> ModelComparison:  # noqa: D102
        judgements, expansion = self.get_judgements()
        if not len(judgements):
            raise MatchboxNoJudgements()
        return evaluation.compare_models(
            paths, pl.from_arrow(judgements), pl.from_arrow(expansion)
        )

    def sample_for_eval(  # noqa: D102
        self, n: int, path: ModelResolutionPath, user_id: int
    ) -> ArrowTable:
        return evaluation.sample(n, path, user_id)
