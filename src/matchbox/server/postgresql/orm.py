"""ORM classes for the Matchbox PostgreSQL database."""

from sqlalchemy import (
    BIGINT,
    INTEGER,
    SMALLINT,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import BYTEA, TEXT
from sqlalchemy.orm import relationship

from matchbox.common.graph import ResolutionNodeType
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
    """Table of resolution points: models, datasets and humans.

    Resolutions produce probabilities or own data in the clusters table.
    """

    __tablename__ = "resolutions"

    # Columns
    resolution_id = Column(BIGINT, primary_key=True)
    resolution_hash = Column(BYTEA, nullable=False)
    type = Column(TEXT, nullable=False)
    name = Column(TEXT, nullable=False)
    description = Column(TEXT)
    truth = Column(SMALLINT)

    # Relationships
    source = relationship("Sources", back_populates="dataset_resolution", uselist=False)
    probabilities = relationship(
        "Probabilities",
        back_populates="proposed_by",
        cascade="all, delete-orphan",
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
            "type IN ('model', 'dataset', 'human')",
            name="resolution_type_constraints",
        ),
        UniqueConstraint("resolution_hash", name="resolutions_hash_key"),
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

    def get_lineage(self) -> dict[int, float]:
        """Returns all ancestors and their cached truth values from this model."""
        with MBDB.get_session() as session:
            lineage_query = (
                select(ResolutionFrom.parent, ResolutionFrom.truth_cache)
                .where(ResolutionFrom.child == self.resolution_id)
                .order_by(ResolutionFrom.level.desc())
            )

            results = session.execute(lineage_query).all()

            lineage = {parent: truth for parent, truth in results}
            lineage[self.resolution_id] = self.truth

            return lineage

    def get_lineage_to_dataset(
        self, dataset: "Resolutions"
    ) -> tuple[bytes, dict[int, float]]:
        """Returns the resolution lineage and cached truth values to a dataset."""
        if dataset.type != ResolutionNodeType.DATASET.value:
            raise ValueError(
                f"Target resolution must be of type 'dataset', got {dataset.type}"
            )

        if self.resolution_id == dataset.resolution_id:
            return {dataset.resolution_id: None}

        with MBDB.get_session() as session:
            path_query = (
                select(ResolutionFrom.parent, ResolutionFrom.truth_cache)
                .join(Resolutions, Resolutions.resolution_id == ResolutionFrom.parent)
                .where(ResolutionFrom.child == self.resolution_id)
                .order_by(ResolutionFrom.level.desc())
            )

            results = session.execute(path_query).all()

            if not any(parent == dataset.resolution_id for parent, _ in results):
                raise ValueError(
                    f"No path between resolution {self.name}, dataset {dataset.name}"
                )

            lineage = {parent: truth for parent, truth in results}
            lineage[self.resolution_id] = self.truth

            return lineage

    @classmethod
    def next_id(cls) -> int:
        """Returns the next available resolution_id."""
        with MBDB.get_session() as session:
            result = session.execute(
                select(func.coalesce(func.max(cls.resolution_id), 0))
            ).scalar()
            return result + 1


class SourceColumns(CountMixin, MBDB.MatchboxBase):
    """Table for storing column details for Sources."""

    __tablename__ = "source_columns"

    # Columns
    column_id = Column(BIGINT, primary_key=True, autoincrement=True)
    source_id = Column(
        BIGINT,
        ForeignKey("sources.source_id", ondelete="CASCADE"),
        nullable=False,
    )
    column_index = Column(INTEGER, nullable=False)
    column_name = Column(TEXT, nullable=False)
    column_type = Column(TEXT, nullable=False)

    # Relationships
    source = relationship("Sources", back_populates="columns")

    # Constraints and indices
    __table_args__ = (
        UniqueConstraint("source_id", "column_index", name="unique_column_index"),
        Index("ix_source_columns_source_id", "source_id"),
    )


class ClusterSourcePK(CountMixin, MBDB.MatchboxBase):
    """Table for storing source primary keys for clusters."""

    __tablename__ = "cluster_source_pks"

    # Columns
    pk_id = Column(BIGINT, primary_key=True)
    cluster_id = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False
    )
    source_id = Column(
        BIGINT, ForeignKey("sources.source_id", ondelete="CASCADE"), nullable=False
    )
    source_pk = Column(TEXT, nullable=False)

    # Relationships
    cluster = relationship("Clusters", back_populates="source_pks")
    source = relationship("Sources", back_populates="cluster_source_pks")

    # Constraints and indices
    __table_args__ = (
        Index("ix_cluster_source_pks_cluster_id", "cluster_id"),
        Index("ix_cluster_source_pks_source_pk", "source_pk"),
        UniqueConstraint("pk_id", "source_id", name="unique_pk_source"),
    )

    @classmethod
    def next_id(cls) -> int:
        """Returns the next available cluster_id."""
        with MBDB.get_session() as session:
            result = session.execute(
                select(func.coalesce(func.max(cls.pk_id), 0))
            ).scalar()
            return result + 1


class Sources(CountMixin, MBDB.MatchboxBase):
    """Table of sources of data for Matchbox."""

    __tablename__ = "sources"

    # Columns
    source_id = Column(BIGINT, autoincrement=True, primary_key=True)
    resolution_id = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
    )
    resolution_name = Column(TEXT, nullable=False)
    full_name = Column(TEXT, nullable=False)
    warehouse_hash = Column(BYTEA, nullable=False)
    db_pk = Column(TEXT, nullable=False)

    # Relationships
    dataset_resolution = relationship("Resolutions", back_populates="source")
    columns = relationship(
        "SourceColumns",
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    cluster_source_pks = relationship(
        "ClusterSourcePK",
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    clusters = relationship(
        "Clusters",
        secondary=ClusterSourcePK.__table__,
        primaryjoin="Sources.resolution_id == ClusterSourcePK.source_id",
        secondaryjoin="ClusterSourcePK.cluster_id == Clusters.cluster_id",
        viewonly=True,
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint("full_name", "warehouse_hash", name="unique_source_address"),
    )

    @classmethod
    def list(cls) -> list["Sources"]:
        """Returns all sources in the database."""
        with MBDB.get_session() as session:
            return session.query(cls).all()


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    __tablename__ = "contains"

    # Columns
    parent = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    child = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )

    # Constraints and indices
    __table_args__ = (
        CheckConstraint("parent != child", name="no_self_containment"),
        Index("ix_contains_parent_child", "parent", "child"),
        Index("ix_contains_child_parent", "child", "parent"),
    )


class Clusters(CountMixin, MBDB.MatchboxBase):
    """Table of indexed data and clusters that match it."""

    __tablename__ = "clusters"

    # Columns
    cluster_id = Column(BIGINT, primary_key=True)
    cluster_hash = Column(BYTEA, nullable=False)

    # Relationships
    source_pks = relationship(
        "ClusterSourcePK",
        back_populates="cluster",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    probabilities = relationship(
        "Probabilities",
        back_populates="proposes",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    children = relationship(
        "Clusters",
        secondary=Contains.__table__,
        primaryjoin="Clusters.cluster_id == Contains.parent",
        secondaryjoin="Clusters.cluster_id == Contains.child",
        backref="parents",
    )
    # Add relationship to Sources through ClusterSourcePK
    sources = relationship(
        "Sources",
        secondary=ClusterSourcePK.__table__,
        primaryjoin="Clusters.cluster_id == ClusterSourcePK.cluster_id",
        secondaryjoin="ClusterSourcePK.source_id == Sources.resolution_id",
        viewonly=True,
    )

    # Constraints and indices
    __table_args__ = (UniqueConstraint("cluster_hash", name="clusters_hash_key"),)

    @classmethod
    def next_id(cls) -> int:
        """Returns the next available cluster_id."""
        with MBDB.get_session() as session:
            result = session.execute(
                select(func.coalesce(func.max(cls.cluster_id), 0))
            ).scalar()
            return result + 1


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster is correct, according to a resolution."""

    __tablename__ = "probabilities"

    # Columns
    resolution = Column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    cluster = Column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    probability = Column(SMALLINT, nullable=False)

    # Relationships
    proposed_by = relationship("Resolutions", back_populates="probabilities")
    proposes = relationship("Clusters", back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
    )
