"""ORM classes for the Matchbox PostgreSQL database."""

import json
from typing import Any, Optional

from sqlalchemy import (
    BIGINT,
    BOOLEAN,
    INTEGER,
    SMALLINT,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Identity,
    Index,
    UniqueConstraint,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, TEXT
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship, selectinload

from matchbox.common.dtos import Collection as CommonCollection
from matchbox.common.dtos import (
    CollectionName,
    LocationConfig,
    ModelType,
    ResolutionName,
    ResolutionPath,
    ResolutionType,
    RunID,
    UploadStage,
)
from matchbox.common.dtos import ModelConfig as CommonModelConfig
from matchbox.common.dtos import Resolution as CommonResolution
from matchbox.common.dtos import Run as CommonRun
from matchbox.common.dtos import SourceConfig as CommonSourceConfig
from matchbox.common.dtos import SourceField as CommonSourceField
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxResolutionAlreadyExists,
    MatchboxResolutionNotFoundError,
    MatchboxRunNotFoundError,
)
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class Collections(CountMixin, MBDB.MatchboxBase):
    """Named collections of resolutions and runs."""

    __tablename__ = "collections"

    collection_id: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)

    # Relationships
    runs: Mapped[list["Runs"]] = relationship(back_populates="collection")
    permissions: Mapped[list["Permissions"]] = relationship(
        back_populates="collection",
        passive_deletes=True,
    )

    # Constraints
    __table_args__ = (UniqueConstraint("name", name="collections_name_key"),)

    @classmethod
    def from_name(
        cls,
        name: CollectionName,
        session: Session | None = None,
    ) -> "Collections":
        """Resolve a collection name to a Collections object.

        Args:
            name: The name of the collection to resolve.
            session: Optional session to use for the query.

        Raises:
            MatchboxCollectionNotFoundError: If the collection doesn't exist.
        """
        query = select(cls).where(cls.name == name).options(selectinload(cls.runs))

        if session:
            collection = session.execute(query).scalar_one_or_none()
        else:
            with MBDB.get_session() as session:
                collection = session.execute(query).scalar_one_or_none()

        if not collection:
            raise MatchboxCollectionNotFoundError(f"Collection '{name}' not found.")

        return collection

    def to_dto(self) -> CommonCollection:
        """Convert ORM collection to a matchbox.common Collection object."""
        run_ids: list[RunID] = []
        default_run = None
        if runs := self.runs:
            run_ids = [r.run_id for r in runs]
            default_run_list = [r.run_id for r in runs if r.is_default]
            if default_run_list:
                default_run = default_run_list[0]

        return CommonCollection(runs=run_ids, default_run=default_run)


class Runs(CountMixin, MBDB.MatchboxBase):
    """Runs of collections of resolutions."""

    __tablename__ = "runs"

    run_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    collection_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("collections.collection_id", ondelete="CASCADE"),
        nullable=False,
    )
    is_mutable: Mapped[bool] = mapped_column(BOOLEAN, default=False, nullable=True)
    is_default: Mapped[bool] = mapped_column(BOOLEAN, default=False, nullable=True)

    # Relationships
    collection: Mapped["Collections"] = relationship(back_populates="runs")
    resolutions: Mapped[list["Resolutions"]] = relationship(back_populates="run")

    # Constraints
    __table_args__ = (
        UniqueConstraint("collection_id", "run_id", name="unique_run_id"),
        Index(
            "ix_default_run_collection",
            "collection_id",
            unique=True,
            postgresql_where=text("is_default = true"),
        ),
    )

    @classmethod
    def from_id(
        cls,
        collection: CollectionName,
        run_id: RunID,
        session: Session | None = None,
    ) -> "Runs":
        """Resolve a collection and run name to a Runs object.

        Args:
            collection: The name of the collection containing the run.
            run_id: The ID of the run within that collection.
            session: Optional session to use for the query.

        Raises:
            MatchboxRunNotFoundError: If the run doesn't exist.
        """
        query = (
            select(cls)
            .where(cls.run_id == run_id)
            .options(
                selectinload(cls.resolutions)
                .selectinload(Resolutions.source_config)
                .selectinload(SourceConfigs.fields),
                selectinload(cls.resolutions).selectinload(Resolutions.model_config),
                selectinload(cls.collection),
            )
        )

        if session:
            run_orm = session.execute(query).scalar_one_or_none()
        else:
            with MBDB.get_session() as session:
                run_orm = session.execute(query).scalar_one_or_none()

        if not run_orm:
            raise MatchboxRunNotFoundError

        if run_orm.collection.name != collection:
            raise MatchboxRunNotFoundError(
                run_id=id,
                message=f"Run {id} not found in collection {collection}",
            )

        return run_orm

    def to_dto(self) -> CommonRun:
        """Convert ORM run to a matchbox.common Run object."""
        resolutions: dict[ResolutionName, CommonResolution] = {}
        if self.resolutions:
            resolutions = {
                resolution.name: resolution.to_dto() for resolution in self.resolutions
            }

        return CommonRun(
            run_id=self.run_id,
            is_default=self.is_default,
            is_mutable=self.is_mutable,
            resolutions=resolutions,
        )


class ResolutionFrom(CountMixin, MBDB.MatchboxBase):
    """Resolution lineage closure table with cached truth values."""

    __tablename__ = "resolution_from"

    # Columns
    parent: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    child: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    level: Mapped[int] = mapped_column(INTEGER, nullable=False)
    truth_cache: Mapped[int | None] = mapped_column(SMALLINT, nullable=True)

    # Constraints
    __table_args__ = (
        CheckConstraint("parent != child", name="no_self_reference"),
        CheckConstraint("level > 0", name="positive_level"),
    )


class Resolutions(CountMixin, MBDB.MatchboxBase):
    """Table of resolution points corresponding to models, and sources.

    Resolutions produce probabilities or own data in the clusters table.
    """

    __tablename__ = "resolutions"

    # Columns
    resolution_id: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=True
    )
    run_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False
    )
    upload_stage: Mapped[UploadStage] = mapped_column(
        Enum(UploadStage, native_enum=True, name="upload_stages", schema="mb"),
        nullable=False,
        default=UploadStage.READY,
    )
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    type: Mapped[str] = mapped_column(TEXT, nullable=False)
    fingerprint: Mapped[bytes] = mapped_column(BYTEA, nullable=False)
    truth: Mapped[int | None] = mapped_column(SMALLINT, nullable=True)

    # Relationships
    source_config: Mapped[Optional["SourceConfigs"]] = relationship(
        back_populates="source_resolution", uselist=False
    )
    model_config: Mapped[Optional["ModelConfigs"]] = relationship(
        back_populates="model_resolution", uselist=False
    )
    probabilities: Mapped[list["Probabilities"]] = relationship(
        back_populates="proposed_by",
        passive_deletes=True,
    )
    results: Mapped[list["Results"]] = relationship(
        back_populates="proposed_by",
        passive_deletes=True,
    )
    children: Mapped[list["Resolutions"]] = relationship(
        secondary=ResolutionFrom.__table__,
        primaryjoin="Resolutions.resolution_id == ResolutionFrom.parent",
        secondaryjoin="Resolutions.resolution_id == ResolutionFrom.child",
        backref="parents",
    )
    run: Mapped["Runs"] = relationship(back_populates="resolutions")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "type IN ('model', 'source')",
            name="resolution_type_constraints",
        ),
        UniqueConstraint("run_id", "name", name="resolutions_name_key"),
    )

    @property
    def ancestors(self) -> set["Resolutions"]:
        """Returns all ancestors (parents, grandparents, etc.) of this resolution."""
        with MBDB.get_session() as session:
            ancestor_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(
                    ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.parent
                )
                .where(ResolutionFrom.child == self.resolution_id)
            )
            return set(session.execute(ancestor_query).scalars().all())

    @property
    def descendants(self) -> set["Resolutions"]:
        """Returns descendants (children, grandchildren, etc.) of this resolution."""
        with MBDB.get_session() as session:
            descendant_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.child)
                .where(ResolutionFrom.parent == self.resolution_id)
            )
            return set(session.execute(descendant_query).scalars().all())

    def get_lineage(
        self, sources: list["SourceConfigs"] | None = None, threshold: int | None = None
    ) -> list[tuple[int, int, float | None]]:
        """Returns lineage ordered by priority.

        Highest priority (lowest level) first, then by resolution_id for stability.

        Args:
            sources: If provided, only return lineage paths that lead to these sources
            threshold: If provided, override this resolution's threshold

        Returns:
            List of tuples (resolution_id, source_config_id, threshold) ordered by
                priority.
        """
        with MBDB.get_session() as session:
            query = (
                select(
                    ResolutionFrom.parent,
                    SourceConfigs.source_config_id,
                    ResolutionFrom.truth_cache,
                )
                .join(
                    SourceConfigs,
                    ResolutionFrom.parent == SourceConfigs.resolution_id,
                    isouter=True,
                )
                .where(ResolutionFrom.child == self.resolution_id)
            )

            if sources:
                # Filter by source configs
                source_resolution_ids = [sc.resolution_id for sc in sources]

                descendant_ids = (
                    session.execute(
                        select(ResolutionFrom.child).where(
                            ResolutionFrom.parent.in_(source_resolution_ids)
                        )
                    )
                    .scalars()
                    .all()
                )

                query = query.where(
                    ResolutionFrom.parent.in_(source_resolution_ids + descendant_ids)
                )

            results = session.execute(
                query.order_by(ResolutionFrom.level.asc(), ResolutionFrom.parent.asc())
            ).all()

            # Get self's source config ID
            self_source_config_id = (
                self.source_config.source_config_id if self.source_config else None
            )

            # Threshold handling
            self_threshold = threshold if threshold is not None else self.truth

            # Add self at beginning (highest priority - level 0)
            return [(self.resolution_id, self_source_config_id, self_threshold)] + list(
                results
            )

    @classmethod
    def from_path(
        cls,
        path: ResolutionPath,
        res_type: ResolutionType | None = None,
        session: Session | None = None,
        for_update: bool = False,
    ) -> "Resolutions":
        """Resolves a resolution name to a Resolution object.

        Args:
            path: The path of the resolution to resolve.
            res_type: A resolution type to use as filter.
            session: A session to get the resolution for updates.
            for_update: Locks the row until updated.

        Raises:
            MatchboxResolutionNotFoundError: If the resolution doesn't exist.
        """
        query = (
            select(cls)
            .join(cls.run)
            .join(Runs.collection)
            .where(
                cls.name == path.name,
                Runs.run_id == path.run,
                Collections.name == path.collection,
            )
        )

        if res_type:
            query = query.where(cls.type == res_type.value)

        if for_update:
            query = query.with_for_update(nowait=True)

        if session:
            resolution = session.execute(query).scalar()
        else:
            with MBDB.get_session() as session:
                resolution = session.execute(query).scalar()

        if resolution:
            return resolution

        raise MatchboxResolutionNotFoundError(
            message=f"No resolution {path} of type {res_type or 'any'}."
        )

    @classmethod
    def from_dto(
        cls, resolution: CommonResolution, path: ResolutionPath, session: Session
    ) -> "Resolutions":
        """Create a Resolutions instance from a Resolution DTO object.

        The resolution will be added to the session and flushed (but not committed).

        For model resolutions, lineage entries will be created automatically.

        Args:
            resolution: The Resolution DTO to convert
            path: The full resolution path
            session: Database session (caller must commit)

        Returns:
            A Resolutions ORM instance with ID and relationships established
        """
        # Find the run ID for the given collection and run ID
        run_obj = session.execute(
            select(Runs)
            .join(Collections)
            .where(Collections.name == path.collection, Runs.run_id == path.run)
        ).scalar_one_or_none()

        if not run_obj:
            raise MatchboxRunNotFoundError(run_id=path.run)

        # Check if resolution already exists within run
        existing_resolutions: Resolutions = run_obj.resolutions
        for res in existing_resolutions:
            if res.name == path.name:
                raise MatchboxResolutionAlreadyExists(
                    f"Resolution {path.name} already exists"
                )

        # Create new resolution
        resolution_orm = cls(
            run_id=run_obj.run_id,
            name=path.name,
            description=resolution.description,
            type=resolution.resolution_type.value,
            truth=resolution.truth,
            fingerprint=resolution.fingerprint,
        )
        session.add(resolution_orm)
        session.flush()  # Get resolution_id

        if resolution.resolution_type == ResolutionType.SOURCE:
            resolution_orm.source_config = SourceConfigs.from_dto(resolution.config)

        elif resolution.resolution_type == ResolutionType.MODEL:
            resolution_orm.model_config = ModelConfigs.from_dto(resolution.config)
            # Create lineage
            left_parent = cls.from_path(
                path=ResolutionPath(
                    collection=path.collection,
                    run=path.run,
                    name=resolution.config.left_query.point_of_truth,
                ),
                session=session,
            )
            cls._create_closure_entries(session, resolution_orm, left_parent)

            if resolution.config.type == ModelType.LINKER:
                right_parent = cls.from_path(
                    path=ResolutionPath(
                        collection=path.collection,
                        run=path.run,
                        name=resolution.config.right_query.point_of_truth,
                    ),
                    session=session,
                )
                cls._create_closure_entries(session, resolution_orm, right_parent)

        return resolution_orm

    def to_dto(self) -> CommonResolution:
        """Convert ORM resolution to a matchbox.common Resolution object."""
        if self.type == ResolutionType.SOURCE:
            config = self.source_config.to_dto()
        else:
            config = self.model_config.to_dto()

        return CommonResolution(
            description=self.description,
            truth=self.truth,
            resolution_type=ResolutionType(self.type),
            config=config,
            fingerprint=self.fingerprint,
        )

    @staticmethod
    def _create_closure_entries(
        session: Session, child: "Resolutions", parent: "Resolutions"
    ) -> None:
        """Create closure table entries for a parent-child relationship."""
        # Direct relationship
        session.add(
            ResolutionFrom(
                parent=parent.resolution_id,
                child=child.resolution_id,
                level=1,
                truth_cache=parent.truth,
            )
        )

        # Transitive closure
        ancestors = (
            session.execute(
                select(ResolutionFrom).where(
                    ResolutionFrom.child == parent.resolution_id
                )
            )
            .scalars()
            .all()
        )

        for ancestor in ancestors:
            session.add(
                ResolutionFrom(
                    parent=ancestor.parent,
                    child=child.resolution_id,
                    level=ancestor.level + 1,
                    truth_cache=ancestor.truth_cache,
                )
            )


class SourceFields(CountMixin, MBDB.MatchboxBase):
    """Table for storing column details for SourceConfigs."""

    __tablename__ = "source_fields"

    # Columns
    field_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    source_config_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("source_configs.source_config_id", ondelete="CASCADE"),
        nullable=False,
    )
    index: Mapped[int] = mapped_column(INTEGER, nullable=False)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    type: Mapped[str] = mapped_column(TEXT, nullable=False)
    is_key: Mapped[bool] = mapped_column(BOOLEAN, nullable=False)

    # Relationships
    source_config: Mapped["SourceConfigs"] = relationship(
        back_populates="fields",
        foreign_keys=[source_config_id],
    )

    # Constraints and indices
    __table_args__ = (
        UniqueConstraint("source_config_id", "index", name="unique_index"),
        Index("ix_source_columns_source_config_id", "source_config_id"),
        Index(
            "ix_unique_key_field",
            "source_config_id",
            unique=True,
            postgresql_where=text("is_key = true"),
        ),
    )


class ClusterSourceKey(CountMixin, MBDB.MatchboxBase):
    """Table for storing source primary keys for clusters."""

    __tablename__ = "cluster_keys"

    # Columns
    key_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    cluster_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    source_config_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("source_configs.source_config_id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(TEXT, nullable=False)

    # Relationships
    cluster: Mapped["Clusters"] = relationship(back_populates="keys")
    source_config: Mapped["SourceConfigs"] = relationship(back_populates="cluster_keys")

    # Constraints and indices
    __table_args__ = (
        Index("ix_cluster_keys_cluster_id", "cluster_id"),
        Index("ix_cluster_keys_keys", "key"),
        Index("ix_cluster_keys_source_config_id", "source_config_id"),
        UniqueConstraint("key_id", "source_config_id", name="unique_keys_source"),
    )


class SourceConfigs(CountMixin, MBDB.MatchboxBase):
    """Table of source_configs of data for Matchbox."""

    __tablename__ = "source_configs"

    # Columns
    source_config_id: Mapped[int] = mapped_column(
        BIGINT, Identity(start=1), primary_key=True
    )
    resolution_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        nullable=False,
    )
    location_type: Mapped[str] = mapped_column(TEXT, nullable=False)
    location_name: Mapped[str] = mapped_column(TEXT, nullable=False)
    extract_transform: Mapped[str] = mapped_column(TEXT, nullable=False)

    @property
    def name(self) -> str:
        """Get the name of the related resolution."""
        return self.source_resolution.name

    # Relationships
    source_resolution: Mapped["Resolutions"] = relationship(
        back_populates="source_config"
    )
    fields: Mapped[list["SourceFields"]] = relationship(
        back_populates="source_config",
        passive_deletes=True,
        cascade="all, delete-orphan",
    )
    key_field: Mapped[Optional["SourceFields"]] = relationship(
        primaryjoin=(
            "and_(SourceConfigs.source_config_id == SourceFields.source_config_id, "
            "SourceFields.is_key == True)"
        ),
        viewonly=True,
        uselist=False,
    )
    index_fields: Mapped[list["SourceFields"]] = relationship(
        primaryjoin=(
            "and_(SourceConfigs.source_config_id == SourceFields.source_config_id, "
            "SourceFields.is_key == False)"
        ),
        viewonly=True,
        order_by="SourceFields.index",
        collection_class=list,
    )
    cluster_keys: Mapped[list["ClusterSourceKey"]] = relationship(
        back_populates="source_config",
        passive_deletes=True,
    )
    clusters: Mapped[list["Clusters"]] = relationship(
        secondary=ClusterSourceKey.__table__,
        primaryjoin=(
            "SourceConfigs.source_config_id == ClusterSourceKey.source_config_id"
        ),
        secondaryjoin="ClusterSourceKey.cluster_id == Clusters.cluster_id",
        viewonly=True,
    )

    def __init__(
        self,
        key_field: SourceFields | None = None,
        index_fields: list[SourceFields] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise SourceConfigs with optional field objects."""
        super().__init__(**kwargs)

        # Add the key field and mark it as the key
        if key_field is not None:
            key_field.is_key = True
            key_field.index = 0
            self.fields.append(key_field)

        # Add index fields with proper indices
        if index_fields is not None:
            for idx, field in enumerate(index_fields):
                field.is_key = False
                field.index = idx + 1
                self.fields.append(field)

    @classmethod
    def list_all(cls) -> list["SourceConfigs"]:
        """Returns all source_configs in the database."""
        with MBDB.get_session() as session:
            return session.execute(select(cls)).scalars().all()

    @classmethod
    def from_dto(
        cls,
        config: CommonSourceConfig,
    ) -> "SourceConfigs":
        """Create a SourceConfigs instance from a Resolution DTO object."""
        # Create the SourceConfigs object
        return cls(
            location_type=str(config.location_config.type),
            location_name=str(config.location_config.name),
            extract_transform=config.extract_transform,
            key_field=SourceFields(
                index=0,
                name=config.key_field.name,
                type=config.key_field.type.value,
            ),
            index_fields=[
                SourceFields(
                    index=idx + 1,
                    name=field.name,
                    type=field.type.value,
                )
                for idx, field in enumerate(config.index_fields)
            ],
        )

    def to_dto(self) -> CommonSourceConfig:
        """Convert ORM source to a matchbox.common.SourceConfig object."""
        return CommonSourceConfig(
            location_config=LocationConfig(
                type=self.location_type, name=self.location_name
            ),
            extract_transform=self.extract_transform,
            key_field=CommonSourceField(
                name=self.key_field.name, type=self.key_field.type
            ),
            index_fields=[
                CommonSourceField(
                    name=field.name,
                    type=field.type,
                )
                for field in self.index_fields
            ],
        )


class ModelConfigs(CountMixin, MBDB.MatchboxBase):
    """Table of model configs for Matchbox."""

    __tablename__ = "model_configs"

    # Columns
    model_config_id: Mapped[int] = mapped_column(
        BIGINT, Identity(start=1), primary_key=True
    )
    resolution_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        nullable=False,
    )
    model_class: Mapped[str] = mapped_column(TEXT, nullable=False)
    model_settings: Mapped[dict] = mapped_column(JSONB, nullable=False)
    left_query: Mapped[dict] = mapped_column(JSONB, nullable=False)
    right_query: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    @property
    def name(self) -> str:
        """Get the name of the related resolution."""
        return self.model_resolution.name

    # Relationships
    model_resolution: Mapped["Resolutions"] = relationship(
        back_populates="model_config"
    )

    @classmethod
    def list_all(cls) -> list["SourceConfigs"]:
        """Returns all model_configs in the database."""
        with MBDB.get_session() as session:
            return session.execute(select(cls)).scalars().all()

    @classmethod
    def from_dto(
        cls,
        config: CommonModelConfig,
    ) -> "ModelConfigs":
        """Create a SourceConfigs instance from a Resolution DTO object."""
        # Create the SourceConfigs object
        return cls(
            model_class=config.model_class,
            model_settings=config.model_settings,
            left_query=config.left_query.model_dump_json(),
            right_query=(
                None if not config.right_query else config.right_query.model_dump_json()
            ),
        )

    def to_dto(self) -> CommonModelConfig:
        """Convert ORM source to a matchbox.common.ModelConfig object."""
        return CommonModelConfig(
            type=ModelType.LINKER if self.right_query else ModelType.DEDUPER,
            model_class=self.model_class,
            model_settings=self.model_settings,
            left_query=json.loads(self.left_query),
            right_query=json.loads(self.right_query) if self.right_query else None,
        )


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    __tablename__ = "contains"

    # Columns
    root: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    leaf: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )

    # Constraints and indices
    __table_args__ = (
        CheckConstraint("root != leaf", name="no_self_containment"),
        UniqueConstraint("root", "leaf"),
        Index("ix_contains_root_leaf", "root", "leaf"),
        Index("ix_contains_leaf_root", "leaf", "root"),
    )


class Clusters(CountMixin, MBDB.MatchboxBase):
    """Table of indexed data and clusters that match it."""

    __tablename__ = "clusters"

    # Columns
    cluster_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    cluster_hash: Mapped[bytes] = mapped_column(BYTEA, nullable=False)

    # Relationships
    keys: Mapped[list["ClusterSourceKey"]] = relationship(
        back_populates="cluster",
        passive_deletes=True,
    )
    probabilities: Mapped[list["Probabilities"]] = relationship(
        back_populates="proposes",
        passive_deletes=True,
    )
    leaves: Mapped[list["Clusters"]] = relationship(
        secondary=Contains.__table__,
        primaryjoin="Clusters.cluster_id == Contains.root",
        secondaryjoin="Clusters.cluster_id == Contains.leaf",
        backref="roots",
    )
    # Add relationship to SourceConfigs through ClusterSourceKey
    source_configs: Mapped[list["SourceConfigs"]] = relationship(
        secondary=ClusterSourceKey.__table__,
        primaryjoin="Clusters.cluster_id == ClusterSourceKey.cluster_id",
        secondaryjoin=(
            "ClusterSourceKey.source_config_id == SourceConfigs.source_config_id"
        ),
        viewonly=True,
    )

    # Constraints and indices
    __table_args__ = (UniqueConstraint("cluster_hash", name="clusters_hash_key"),)


class UserGroups(MBDB.MatchboxBase):
    """Association table for user-group membership."""

    __tablename__ = "user_groups"

    # Columns
    user_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    group_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("groups.group_id", ondelete="CASCADE"),
        primary_key=True,
    )


class Users(CountMixin, MBDB.MatchboxBase):
    """Table of user identities."""

    __tablename__ = "users"

    # Columns
    user_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    email: Mapped[str] = mapped_column(TEXT, nullable=True)

    # Relationships
    judgements: Mapped[list["EvalJudgements"]] = relationship(back_populates="user")
    groups: Mapped[list["Groups"]] = relationship(
        secondary=UserGroups.__table__,
        back_populates="members",
    )

    # Constraints and indices
    __table_args__ = (UniqueConstraint("name", name="user_name_unique"),)


class Groups(CountMixin, MBDB.MatchboxBase):
    """Groups for permission management."""

    __tablename__ = "groups"

    # Columns
    group_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(TEXT, nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    is_system: Mapped[bool] = mapped_column(BOOLEAN, default=False, nullable=False)

    # Relationships
    members: Mapped[list["Users"]] = relationship(
        secondary=UserGroups.__table__,
        back_populates="groups",
    )
    permissions: Mapped[list["Permissions"]] = relationship(
        back_populates="group",
        passive_deletes=True,
    )

    # Constraints and indices
    __table_args__ = (UniqueConstraint("name", name="groups_name_key"),)


class Permissions(CountMixin, MBDB.MatchboxBase):
    """Permissions granted to groups on resources.

    Each resource type should have one column. This creates lots of nulls,
    which are cheap in PostgreSQL and are on an ultimately small table,
    and avoids a polymorphic association.
    """

    __tablename__ = "permissions"

    # Columns
    permission_id: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=True
    )
    permission: Mapped[str] = mapped_column(TEXT, nullable=False)
    group_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("groups.group_id", ondelete="CASCADE"),
        nullable=False,
    )
    collection_id: Mapped[int | None] = mapped_column(
        BIGINT,
        ForeignKey("collections.collection_id", ondelete="CASCADE"),
        nullable=True,
    )
    is_system: Mapped[bool | None] = mapped_column(
        BOOLEAN,
        nullable=True,
    )

    # Relationships
    group: Mapped["Groups"] = relationship(back_populates="permissions")
    collection: Mapped["Collections | None"] = relationship(
        back_populates="permissions"
    )

    # Constraints and indices
    __table_args__ = (
        CheckConstraint(
            "permission IN ('read', 'write', 'admin')",
            name="valid_permission",
        ),
        CheckConstraint(
            "(collection_id IS NOT NULL AND is_system IS NULL) OR "
            "(collection_id IS NULL AND is_system = true)",
            name="exactly_one_resource",
        ),
        UniqueConstraint(
            "permission",
            "group_id",
            "collection_id",
            "is_system",
            name="unique_permission_grant",
        ),
    )


class EvalJudgements(CountMixin, MBDB.MatchboxBase):
    """Table of evaluation judgements produced by human validators."""

    __tablename__ = "eval_judgements"

    # Columns
    judgement_id: Mapped[int] = mapped_column(BIGINT, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    endorsed_cluster_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    shown_cluster_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    tag: Mapped[str] = mapped_column(TEXT, nullable=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Relationships
    user: Mapped["Users"] = relationship(back_populates="judgements")


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster is correct, according to a resolution."""

    __tablename__ = "probabilities"

    # Columns
    resolution_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    cluster_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    probability: Mapped[int] = mapped_column(SMALLINT, nullable=False)

    # Relationships
    proposed_by: Mapped["Resolutions"] = relationship(back_populates="probabilities")
    proposes: Mapped["Clusters"] = relationship(back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
        Index("ix_probabilities_resolution", "resolution_id"),
    )


class Results(CountMixin, MBDB.MatchboxBase):
    """Table of results for a resolution.

    Stores the raw left/right probabilities created by a model.
    """

    __tablename__ = "results"

    # Columns
    result_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    resolution_id: Mapped[int] = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        nullable=False,
    )
    left_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    right_id: Mapped[int] = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    probability: Mapped[int] = mapped_column(SMALLINT, nullable=False)

    # Relationships
    proposed_by: Mapped["Resolutions"] = relationship(back_populates="results")

    # Constraints
    __table_args__ = (
        Index("ix_results_resolution", "resolution_id"),
        CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
        UniqueConstraint("resolution_id", "left_id", "right_id"),
    )
