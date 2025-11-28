"""Collections PostgreSQL mixin for Matchbox server."""

from psycopg.errors import LockNotAvailable
from pyarrow import Table
from sqlalchemy import delete, select, update

from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    Collection,
    CollectionName,
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
from matchbox.common.dtos import SourceConfig as CommonSourceConfig
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxDeletionNotConfirmed,
    MatchboxLockError,
    MatchboxResolutionUpdateError,
    MatchboxRunNotWriteable,
)
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Collections,
    ModelConfigs,
    Resolutions,
    Results,
    Runs,
    SourceConfigs,
    SourceFields,
)
from matchbox.server.postgresql.utils.db import compile_sql
from matchbox.server.postgresql.utils.insert import insert_hashes, insert_results


class MatchboxPostgresCollectionsMixin:
    """Collections mixin for the PostgreSQL adapter for Matchbox."""

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
            # Lock resolution so only one client can initiate the upload
            # Will fail if already locked
            try:
                resolution = Resolutions.from_path(
                    path=path, session=session, for_update=True
                )
            except LockNotAvailable as e:
                raise MatchboxLockError("Resolution is locked.") from e

            # Check status
            # Will fail if stage not READY
            if resolution.upload_stage == UploadStage.COMPLETE:
                session.rollback()
                raise MatchboxLockError(
                    "Once set to complete, resolution data stage cannot be changed."
                )
            elif resolution.upload_stage == UploadStage.PROCESSING:
                session.rollback()
                raise MatchboxLockError("Upload already being processed.")

            resolution.upload_stage = UploadStage.PROCESSING
            session.commit()

    def unlock_resolution_data(  # noqa: D102
        self, path: ResolutionPath, complete: bool = False
    ) -> None:
        self._check_writeable(path)
        with MBDB.get_session() as session:
            resolution = Resolutions.from_path(
                path=path, session=session, for_update=True
            )
            if complete:
                resolution.upload_stage = UploadStage.COMPLETE
            else:
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
        self.unlock_resolution_data(path=path, complete=True)

    def insert_model_data(self, path: ModelResolutionPath, results: Table) -> None:  # noqa: D102
        self._check_writeable(path)
        insert_results(path=path, results=results, batch_size=self.settings.batch_size)
        self.unlock_resolution_data(path=path, complete=True)

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
