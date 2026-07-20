"""Collections PostgreSQL mixin for Matchbox server."""

from psycopg.errors import LockNotAvailable
from pyarrow import Table
from sqlalchemy import CursorResult, delete, select, update

from matchbox.common.adapters.sql.query import resolver_membership_subquery
from matchbox.common.arrow import SCHEMA_CLUSTERS, SCHEMA_MODEL_EDGES
from matchbox.common.db import sql_to_df
from matchbox.common.dtos import (
    Collection,
    CollectionName,
    ModelStepPath,
    PermissionGrant,
    ResolverStepPath,
    Run,
    RunID,
    SourceStepPath,
    Step,
    StepPath,
    StepType,
    UploadStage,
)
from matchbox.common.dtos import ModelConfig as CommonModelConfig
from matchbox.common.dtos import ResolverConfig as CommonResolverConfig
from matchbox.common.dtos import SourceConfig as CommonSourceConfig
from matchbox.common.exceptions import (
    MatchboxCollectionAlreadyExists,
    MatchboxDeletionNotConfirmed,
    MatchboxLockError,
    MatchboxRunNotWriteable,
    MatchboxStepUpdateError,
)
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Collections,
    ModelConfigs,
    ModelEdges,
    ResolverConfigs,
    Runs,
    SourceConfigs,
    SourceFields,
    Steps,
    insert,
)
from matchbox.server.postgresql.utils.db import compile_sql, grant_permission
from matchbox.server.postgresql.utils.insert import (
    insert_clusters,
    insert_hashes,
    insert_model_edges,
)
from matchbox.server.postgresql.utils.query import (
    require_complete_resolver,
)


class MatchboxPostgresCollectionsMixin:
    """Collections mixin for the PostgreSQL adapter for Matchbox."""

    # Collection management

    def create_collection(  # noqa: D102
        self, name: CollectionName, permissions: list[PermissionGrant]
    ) -> Collection:
        with MBDB.get_session() as session:
            # Attempt to insert the collection
            result: CursorResult = session.execute(
                insert(Collections)
                .values(name=name)
                .on_conflict_do_nothing(constraint="collections_name_key")
                .returning(Collections.collection_id)
            )

            collection_id = result.scalar_one_or_none()
            if collection_id is None:
                raise MatchboxCollectionAlreadyExists

            # Get the newly created collection
            new_collection = session.get(Collections, collection_id)

            # Grant initial permissions
            for permission_grant in permissions:
                grant_permission(
                    session=session,
                    group_name=permission_grant.group_name,
                    permission=permission_grant.permission,
                    resource=name,
                )

            session.commit()
            return new_collection.to_dto()

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
                step_names = [res.name for res in run_orm.steps]
                raise MatchboxDeletionNotConfirmed(children=step_names)

            session.execute(delete(Runs).where(Runs.run_id == run_orm.run_id))
            session.commit()

    # Step management

    def _check_writeable(self, path: StepPath) -> None:
        run = Runs.from_id(collection=path.collection, run_id=path.run)
        if not run.is_mutable:
            raise MatchboxRunNotWriteable(
                f"Version {path.run} in collection {path.collection} is immutable"
            )

    def create_step(  # noqa: D102
        self, step: Step, path: StepPath
    ) -> None:
        self._check_writeable(path)
        log_prefix = f"Insert {path.name}"
        with MBDB.get_session() as session:
            step_orm = Steps.from_dto(step=step, path=path, session=session)
            session.commit()

            logger.info(f"Inserted with ID {step_orm.step_id}", prefix=log_prefix)

    def get_step(  # noqa: D102
        self, path: StepPath
    ) -> Step:
        with MBDB.get_session() as session:
            step = Steps.from_path(path=path, session=session)
            return step.to_dto()

    def update_step(  # noqa: D102
        self, step: Step, path: StepPath
    ) -> None:
        new_config = step.config
        with MBDB.get_session() as session:
            # Get current ORM entry
            old_step = Steps.from_path(path=path, session=session)
            # Check current ORM entry can be updated
            if old_step.fingerprint != step.fingerprint:
                raise MatchboxStepUpdateError(
                    "Cannot update step with non-matching fingerprint."
                )
            # The following condition also protects against change of step type:
            # sources must have 0 parents, models must have 1 or more
            if old_step.to_dto().config.parents != new_config.parents:
                raise MatchboxStepUpdateError("Cannot change parents of a step.")

            # Update top-level metadata
            old_step.description = step.description

            # Update config
            if old_step.type == "source":
                old_config: SourceConfigs = old_step.source_config
                if not isinstance(new_config, CommonSourceConfig):
                    raise ValueError("Config for source step expected.")
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

            elif old_step.type == StepType.MODEL:
                old_config: ModelConfigs = old_step.model_config
                if not isinstance(new_config, CommonModelConfig):
                    raise ValueError("Config for model step expected.")
                old_config.model_class = new_config.model_class
                old_config.model_settings = new_config.model_settings
                old_config.left_query = new_config.left_query.model_dump(mode="json")
                old_config.right_query = (
                    None
                    if not new_config.right_query
                    else new_config.right_query.model_dump(mode="json")
                )
            elif old_step.type == StepType.RESOLVER:
                old_config: ResolverConfigs = old_step.resolver_config
                if not isinstance(new_config, CommonResolverConfig):
                    raise ValueError("Config for resolver step expected.")
                old_config.resolver_class = new_config.resolver_class
                old_config.resolver_settings = new_config.resolver_settings
            else:
                raise ValueError(f"Unsupported step type for update: {old_step.type}")

            session.commit()

    def delete_step(self, path: StepPath, certain: bool) -> None:  # noqa: D102
        self._check_writeable(path)
        with MBDB.get_session() as session:
            step = Steps.from_path(path=path, session=session)
            if certain:
                delete_stmt = delete(Steps).where(
                    Steps.step_id.in_(
                        [
                            step.step_id,
                            *(d.step_id for d in step.descendants),
                        ]
                    )
                )
                session.execute(delete_stmt)
                session.commit()
            else:
                children = [r.name for r in step.descendants]
                raise MatchboxDeletionNotConfirmed(children=children)

    # Data insertion

    def lock_step_data(self, path: StepPath) -> None:  # noqa: D102
        self._check_writeable(path)
        with MBDB.get_session() as session:
            # Lock step so only one client can initiate the upload
            # Will fail if already locked
            try:
                step = Steps.from_path(path=path, session=session, for_update=True)
            except LockNotAvailable as e:
                raise MatchboxLockError("Step is locked.") from e

            # Check status
            # Will fail if stage not READY
            if step.upload_stage == UploadStage.COMPLETE:
                session.rollback()
                raise MatchboxLockError(
                    "Once set to complete, step data stage cannot be changed."
                )
            elif step.upload_stage == UploadStage.PROCESSING:
                session.rollback()
                raise MatchboxLockError("Upload already being processed.")

            step.upload_stage = UploadStage.PROCESSING
            session.commit()

    def unlock_step_data(  # noqa: D102
        self, path: StepPath, complete: bool = False
    ) -> None:
        self._check_writeable(path)
        with MBDB.get_session() as session:
            step = Steps.from_path(path=path, session=session, for_update=True)
            if complete:
                step.upload_stage = UploadStage.COMPLETE
            else:
                step.upload_stage = UploadStage.READY
            session.commit()

    def get_step_stage(self, path: StepPath) -> UploadStage:  # noqa: D102
        step = Steps.from_path(path)
        return UploadStage(step.upload_stage)

    def insert_source_data(  # noqa: D102
        self, path: SourceStepPath, data_hashes: Table
    ) -> None:
        self._check_writeable(path)
        insert_hashes(
            path=path, data_hashes=data_hashes, batch_size=self.settings.batch_size
        )
        self.unlock_step_data(path=path, complete=True)

    def insert_model_data(self, path: ModelStepPath, results: Table) -> None:  # noqa: D102
        self._check_writeable(path)
        insert_model_edges(
            path=path, results=results, batch_size=self.settings.batch_size
        )
        self.unlock_step_data(path=path, complete=True)

    def insert_resolver_data(self, path: ResolverStepPath, data: Table) -> None:  # noqa: D102
        self._check_writeable(path)
        insert_clusters(
            path=path,
            cluster_assignments=data,
            batch_size=self.settings.batch_size,
        )
        self.unlock_step_data(path=path, complete=True)

    def get_model_data(self, path: ModelStepPath) -> Table:  # noqa: D102
        with MBDB.get_session() as session:
            step = Steps.from_path(path=path, res_type=StepType.MODEL, session=session)

            results_query = select(
                ModelEdges.left_id, ModelEdges.right_id, ModelEdges.score
            ).where(ModelEdges.step_id == step.step_id)

        with MBDB.get_adbc_connection() as conn:
            stmt: str = compile_sql(results_query)
            res: Table = sql_to_df(stmt=stmt, connection=conn, return_type="arrow")
            return res.cast(SCHEMA_MODEL_EDGES)

    def get_resolver_data(self, path: ResolverStepPath) -> Table:  # noqa: D102
        with MBDB.get_session() as session:
            step = require_complete_resolver(session=session, path=path)
            assignments_query = resolver_membership_subquery(
                step_id=step.step_id,
                alias="assignments",
            )
            ordered_query = select(
                assignments_query.c.root_id.label("parent_id"),
                assignments_query.c.leaf_id.label("child_id"),
            ).order_by(
                assignments_query.c.root_id,
                assignments_query.c.leaf_id,
            )

        with MBDB.get_adbc_connection() as conn:
            stmt: str = compile_sql(ordered_query)
            res: Table = sql_to_df(stmt=stmt, connection=conn, return_type="arrow")
            return res.cast(SCHEMA_CLUSTERS)
