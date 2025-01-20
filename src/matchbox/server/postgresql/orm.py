from sqlalchemy import (
    BIGINT,
    FLOAT,
    INTEGER,
    SMALLINT,
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, TEXT
from sqlalchemy.orm import Session, mapped_column, relationship

from matchbox.common.graph import ResolutionNodeType
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class ResolutionFrom(CountMixin, MBDB.MatchboxBase):
    """Resolution lineage closure table with cached truth values."""

    __tablename__ = "resolution_from"

    # Columns
    parent = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    child = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    level = mapped_column(INTEGER, nullable=False)
    truth_cache = mapped_column(FLOAT, nullable=True)

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
    resolution_id = mapped_column(BIGINT, primary_key=True)
    resolution_hash = mapped_column(BYTEA, nullable=False)
    type = mapped_column(TEXT, nullable=False)
    name = mapped_column(TEXT)
    description = mapped_column(TEXT)
    truth = mapped_column(FLOAT)

    # Relationships
    source = relationship("Sources", back_populates="dataset_resolution", uselist=False)
    probabilities = relationship(
        "Probabilities", back_populates="proposed_by", cascade="all, delete-orphan"
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
        CheckConstraint(
            "name IS NOT NULL OR type = 'dataset'",
            name="conditional_nullable_name_constraints",
        ),
        UniqueConstraint("resolution_hash", name="resolutions_hash_key"),
        UniqueConstraint("name", name="resolutions_name_key"),
    )

    @property
    def ancestors(self) -> set["Resolutions"]:
        """Returns all ancestors (parents, grandparents, etc.) of this resolution."""
        with Session(MBDB.get_engine()) as session:
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
        with Session(MBDB.get_engine()) as session:
            descendant_query = (
                select(Resolutions)
                .select_from(Resolutions)
                .join(ResolutionFrom, Resolutions.resolution_id == ResolutionFrom.child)
                .where(ResolutionFrom.parent == self.resolution_id)
            )
            return set(session.execute(descendant_query).scalars().all())

    def get_lineage(self) -> dict[int, float]:
        """Returns all ancestors and their cached truth values from this model."""
        with Session(MBDB.get_engine()) as session:
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

        with Session(MBDB.get_engine()) as session:
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
        with Session(MBDB.get_engine()) as session:
            result = session.execute(
                select(func.coalesce(func.max(cls.resolution_id), 0))
            ).scalar()
            return result + 1


class Sources(CountMixin, MBDB.MatchboxBase):
    """Table of sources of data for Matchbox."""

    __tablename__ = "sources"

    # Columns
    resolution_id = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    alias = mapped_column(TEXT, nullable=False)
    full_name = mapped_column(TEXT, nullable=False)
    warehouse_hash = mapped_column(BYTEA, nullable=False)
    id = mapped_column(TEXT, nullable=False)
    column_names = mapped_column(ARRAY(TEXT), nullable=False)
    column_aliases = mapped_column(ARRAY(TEXT), nullable=False)

    # Relationships
    dataset_resolution = relationship("Resolutions", back_populates="source")
    clusters = relationship("Clusters", back_populates="source")

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "alias", "full_name", "warehouse_hash", name="unique_alias_name_address"
        ),
    )

    @classmethod
    def list(cls) -> list["Sources"]:
        with Session(MBDB.get_engine()) as session:
            return session.query(cls).all()


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    __tablename__ = "contains"

    # Columns
    parent = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    child = mapped_column(
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
    cluster_id = mapped_column(BIGINT, primary_key=True)
    cluster_hash = mapped_column(BYTEA, nullable=False)
    dataset = mapped_column(BIGINT, ForeignKey("sources.resolution_id"), nullable=True)
    # Uses array as source data may have identical rows. We can't control this
    # Must be indexed or PostgreSQL incorrectly tries to use nested joins
    # when retrieving small datasets in query() -- extremely slow
    source_pk = mapped_column(ARRAY(TEXT), index=True, nullable=True)

    # Relationships
    source = relationship("Sources", back_populates="clusters")
    probabilities = relationship(
        "Probabilities", back_populates="proposes", cascade="all, delete-orphan"
    )
    children = relationship(
        "Clusters",
        secondary=Contains.__table__,
        primaryjoin="Clusters.cluster_id == Contains.parent",
        secondaryjoin="Clusters.cluster_id == Contains.child",
        backref="parents",
    )

    # Constraints and indices
    __table_args__ = (
        Index("ix_clusters_id_gin", source_pk, postgresql_using="gin"),
        UniqueConstraint("cluster_hash", name="clusters_hash_key"),
    )

    @classmethod
    def next_id(cls) -> int:
        """Returns the next available cluster_id."""
        with Session(MBDB.get_engine()) as session:
            result = session.execute(
                select(func.coalesce(func.max(cls.cluster_id), 0))
            ).scalar()
            return result + 1


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster is correct, according to a resolution."""

    __tablename__ = "probabilities"

    # Columns
    resolution = mapped_column(
        BIGINT,
        ForeignKey("resolutions.resolution_id", ondelete="CASCADE"),
        primary_key=True,
    )
    cluster = mapped_column(
        BIGINT, ForeignKey("clusters.cluster_id", ondelete="CASCADE"), primary_key=True
    )
    probability = mapped_column(SMALLINT, nullable=False)

    # Relationships
    proposed_by = relationship("Resolutions", back_populates="probabilities")
    proposes = relationship("Clusters", back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 100", name="valid_probability"),
    )
