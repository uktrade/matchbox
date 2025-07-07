"""ORM classes for the Matchbox PostgreSQL database."""

from typing import Literal

from sqlalchemy import (
    BIGINT,
    BOOLEAN,
    INTEGER,
    SMALLINT,
    CheckConstraint,
    Column,
    ForeignKey,
    Identity,
    Index,
    UniqueConstraint,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import BYTEA, TEXT, insert
from sqlalchemy.orm import Session, relationship

from matchbox.common.dtos import ResolutionName
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
)
from matchbox.common.sources import SourceConfig as CommonSourceConfig
from matchbox.common.sources import SourceField as CommonSourceField
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class ResolutionFrom(CountMixin, MBDB.MatchboxBase):
    """Resolution lineage closure table with cached truth values."""

    __tablename__ = "resolution_from"

    # Columns
    parent = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    child = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    level = Column(INTEGER, nullable=False)
    truth_cache = Column(SMALLINT, nullable=True)

    # Constraints
    __table_args__ = (
        CheckConstraint("parent != child", name="no_self_reference"),
        CheckConstraint("level > 0", name="positive_level"),
    )


class Resolutions(CountMixin, MBDB.MatchboxBase):
    """Table of resolution points: models, sources and humans.

    Resolutions produce probabilities or own data in the clusters table.
    """

    __tablename__ = "resolutions"

    # Columns
    resolution_id = Column(BIGINT, primary_key=True, autoincrement=True)
    name = Column(TEXT, nullable=False)
    description = Column(TEXT, nullable=True)
    type = Column(TEXT, nullable=False)
    hash = Column(BYTEA, nullable=True)
    truth = Column(SMALLINT, nullable=True)

    # Relationships
    source_config = relationship(
        "SourceConfigs", back_populates="source_resolution", uselist=False
    )
    probabilities = relationship(
        "Probabilities",
        back_populates="proposed_by",
        passive_deletes=True,
    )
    results = relationship(
        "Results",
        back_populates="proposed_by",
        passive_deletes=True,
    )
    children = relationship(
        "Resolutions",
        secondary=ResolutionFrom.__table__,
        primaryjoin="Resolutions.resolution_id == ResolutionFrom.parent",
        secondaryjoin="Resolutions.resolution_id == ResolutionFrom.child",
        backref="parents",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "type IN ('model', 'source', 'human')",
            name="resolution_type_constraints",
        ),
        UniqueConstraint("name", name="resolutions_name_key"),
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
    def from_name(
        cls,
        name: ResolutionName,
        res_type: Literal["model", "source", "human"] | None = None,
        session: Session | None = None,
    ) -> "Resolutions":
        """Resolves a model resolution name to a Resolution object.

        Args:
            name: The name of the model to resolve.
            res_type: A resolution type to use as filter.
            session: A session to get the resolution for updates.

        Raises:
            MatchboxResolutionNotFoundError: If the model doesn't exist.
        """
        query = select(cls).where(cls.name == name)
        if res_type:
            query = query.where(cls.type == res_type)

        if session:
            resolution = session.execute(query).scalar()
        else:
            with MBDB.get_session() as session:
                resolution = session.execute(query).scalar()

        if resolution:
            return resolution

        res_type = res_type or "any"
        raise MatchboxResolutionNotFoundError(
            message=f"No resolution {name} of {res_type}."
        )


class PKSpace(MBDB.MatchboxBase):
    """Table used to reserve ranges of primary keys."""

    __tablename__ = "pk_space"

    id = Column(BIGINT, primary_key=True)
    next_cluster_id = Column(BIGINT, nullable=False)
    next_cluster_keys_id = Column(BIGINT, nullable=False)

    @classmethod
    def initialise(cls) -> None:
        """Create PKSpace tracking row if not exists."""
        with MBDB.get_session() as session:
            init_statement = (
                insert(cls)
                .values(id=1, next_cluster_id=1, next_cluster_keys_id=1)
                .on_conflict_do_nothing()
            )

            session.execute(init_statement)
            session.commit()

    @classmethod
    def reserve_block(
        cls, table: Literal["clusters", "cluster_keys"], block_size: int
    ) -> int:
        """Atomically get next available ID for table, and increment it."""
        if block_size < 1:
            raise ValueError("Block size must be at least 1.")

        match table:
            case "clusters":
                next_id_col = "next_cluster_id"
            case "cluster_keys":
                next_id_col = "next_cluster_keys_id"

        with MBDB.get_session() as session:
            try:
                next_id = session.execute(
                    update(cls)
                    .values(**{next_id_col: getattr(cls, next_id_col) + block_size})
                    .returning(getattr(cls, next_id_col) - block_size)
                ).scalar_one()
                session.commit()
            except:
                session.rollback()
                raise

            return next_id


class SourceFields(CountMixin, MBDB.MatchboxBase):
    """Table for storing column details for SourceConfigs."""

    __tablename__ = "source_fields"

    # Columns
    field_id = Column(BIGINT, primary_key=True)
    source_config_id = Column(
        BIGINT,
        ForeignKey("source_configs.source_config_id", ondelete="CASCADE"),
        nullable=False,
    )
    index = Column(INTEGER, nullable=False)
    name = Column(TEXT, nullable=False)
    type = Column(TEXT, nullable=False)
    is_key = Column(BOOLEAN, nullable=False)

    # Relationships
    source_config = relationship(
        "SourceConfigs",
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
    key_id = Column(BIGINT, primary_key=True)
    cluster_id = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    source_config_id = Column(
        BIGINT,
        ForeignKey("source_configs.source_config_id", ondelete="CASCADE"),
        nullable=False,
    )
    key = Column(TEXT, nullable=False)

    # Relationships
    cluster = relationship("Clusters", back_populates="keys")
    source_config = relationship("SourceConfigs", back_populates="cluster_keys")

    # Constraints and indices
    __table_args__ = (
        Index("ix_cluster_keys_cluster_id", "cluster_id"),
        Index("ix_cluster_keys_keys", "key"),
        UniqueConstraint("key_id", "source_config_id", name="unique_keys_source"),
    )


class SourceConfigs(CountMixin, MBDB.MatchboxBase):
    """Table of source_configs of data for Matchbox."""

    __tablename__ = "source_configs"

    # Columns
    source_config_id = Column(BIGINT, Identity(start=1), primary_key=True)
    resolution_id = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        nullable=False,
    )
    location_type = Column(TEXT, nullable=False)
    location_uri = Column(TEXT, nullable=False)
    extract_transform = Column(TEXT, nullable=False)

    @property
    def name(self) -> str:
        """Get the name of the related resolution."""
        return self.source_resolution.name

    # Relationships
    source_resolution = relationship("Resolutions", back_populates="source_config")
    fields = relationship(
        "SourceFields",
        back_populates="source_config",
        passive_deletes=True,
        cascade="all, delete-orphan",
    )
    key_field = relationship(
        "SourceFields",
        primaryjoin=(
            "and_(SourceConfigs.source_config_id == SourceFields.source_config_id, "
            "SourceFields.is_key == True)"
        ),
        viewonly=True,
        uselist=False,
    )
    index_fields = relationship(
        "SourceFields",
        primaryjoin=(
            "and_(SourceConfigs.source_config_id == SourceFields.source_config_id, "
            "SourceFields.is_key == False)"
        ),
        viewonly=True,
        order_by="SourceFields.index",
        collection_class=list,
    )
    cluster_keys = relationship(
        "ClusterSourceKey",
        back_populates="source_config",
        passive_deletes=True,
    )
    clusters = relationship(
        "Clusters",
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
        **kwargs,
    ):
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
            return session.query(cls).all()

    @classmethod
    def from_dto(
        cls,
        resolution: "Resolutions",
        source_config: CommonSourceConfig,
    ) -> "SourceConfigs":
        """Create a SourceConfigs instance from a CommonSource object."""
        return cls(
            resolution_id=resolution.resolution_id,
            location_type=source_config.location.type,
            location_uri=str(source_config.location.uri),
            extract_transform=source_config.extract_transform,
            key_field=SourceFields(
                index=0,
                name=source_config.key_field.name,
                type=source_config.key_field.type.value,
            ),
            index_fields=[
                SourceFields(
                    index=idx + 1,
                    name=field.name,
                    type=field.type.value,
                )
                for idx, field in enumerate(source_config.index_fields)
            ],
        )

    def to_dto(self) -> CommonSourceConfig:
        """Convert ORM source to a matchbox.common SourceConfig object."""
        return CommonSourceConfig(
            name=self.name,
            location={
                "type": self.location_type,
                "uri": self.location_uri,
            },
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


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    __tablename__ = "contains"

    # Columns
    root = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    leaf = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )

    # Constraints and indices
    __table_args__ = (
        CheckConstraint("root != leaf", name="no_self_containment"),
        Index("ix_contains_root_leaf", "root", "leaf"),
        Index("ix_contains_leaf_root", "leaf", "root"),
    )


class Clusters(CountMixin, MBDB.MatchboxBase):
    """Table of indexed data and clusters that match it."""

    __tablename__ = "clusters"

    # Columns
    cluster_id = Column(BIGINT, primary_key=True)
    cluster_hash = Column(BYTEA, nullable=False)

    # Relationships
    keys = relationship(
        "ClusterSourceKey",
        back_populates="cluster",
        passive_deletes=True,
    )
    probabilities = relationship(
        "Probabilities",
        back_populates="proposes",
        passive_deletes=True,
    )
    leaves = relationship(
        "Clusters",
        secondary=Contains.__table__,
        primaryjoin="Clusters.cluster_id == Contains.root",
        secondaryjoin="Clusters.cluster_id == Contains.leaf",
        backref="roots",
    )
    # Add relationship to SourceConfigs through ClusterSourceKey
    source_configs = relationship(
        "SourceConfigs",
        secondary=ClusterSourceKey.__table__,
        primaryjoin="Clusters.cluster_id == ClusterSourceKey.cluster_id",
        secondaryjoin=(
            "ClusterSourceKey.source_config_id == SourceConfigs.source_config_id"
        ),
        viewonly=True,
    )

    # Constraints and indices
    __table_args__ = (UniqueConstraint("cluster_hash", name="clusters_hash_key"),)


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster is correct, according to a resolution."""

    __tablename__ = "probabilities"

    # Columns
    resolution_id = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    cluster_id = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    probability = Column(SMALLINT, nullable=False)

    # Relationships
    proposed_by = relationship("Resolutions", back_populates="probabilities")
    proposes = relationship("Clusters", back_populates="probabilities")

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
    result_id = Column(BIGINT, primary_key=True, autoincrement=True)
    resolution_id = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        nullable=False,
    )
    left_id = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    right_id = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    probability = Column(SMALLINT, nullable=False)

    # Relationships
    proposed_by = relationship("Resolutions", back_populates="results")

    # Constraints
    __table_args__ = (
        Index("ix_results_resolution", "resolution_id"),
        CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
        UniqueConstraint("resolution_id", "left_id", "right_id"),
    )
