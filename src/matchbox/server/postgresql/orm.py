from enum import Enum

from sqlalchemy import (
    FLOAT,
    INTEGER,
    VARCHAR,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA
from sqlalchemy.orm import Session, relationship

from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin


class ModelType(Enum):
    MODEL = "model"
    DATASET = "dataset"
    HUMAN = "human"


class ModelsFrom(CountMixin, MBDB.MatchboxBase):
    """Model lineage closure table with cached truth values."""

    __tablename__ = "models_from"

    # Columns
    parent = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )
    child = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )
    level = Column(INTEGER, nullable=False)
    truth_cache = Column(FLOAT, nullable=True)

    # Constraints
    __table_args__ = (
        CheckConstraint("parent != child", name="no_self_reference"),
        CheckConstraint("level > 0", name="positive_level"),
    )


class Models(CountMixin, MBDB.MatchboxBase):
    """Table of models and model-like objects: models, datasets and humans.

    By model-like objects, we mean objects that produce probabilities or own
    data in the clusters table.
    """

    __tablename__ = "models"

    # Columns
    hash = Column(BYTEA, primary_key=True)
    type = Column(VARCHAR, nullable=False)
    name = Column(VARCHAR, nullable=False, unique=True)
    description = Column(VARCHAR)
    truth = Column(FLOAT)

    # Relationships
    source = relationship("Sources", back_populates="dataset_model", uselist=False)
    probabilities = relationship(
        "Probabilities", back_populates="proposed_by", cascade="all, delete-orphan"
    )
    children = relationship(
        "Models",
        secondary=ModelsFrom.__table__,
        primaryjoin="Models.hash == ModelsFrom.parent",
        secondaryjoin="Models.hash == ModelsFrom.child",
        backref="parents",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "type IN ('model', 'dataset', 'human')",
            name="model_type_constraints",
        ),
    )

    @property
    def ancestors(self) -> set["Models"]:
        """Returns all ancestors (parents, grandparents, etc.) of this model."""
        with Session(MBDB.get_engine()) as session:
            ancestor_query = (
                select(Models)
                .select_from(Models)
                .join(ModelsFrom, Models.hash == ModelsFrom.parent)
                .where(ModelsFrom.child == self.hash)
            )
            return set(session.execute(ancestor_query).scalars().all())

    @property
    def descendants(self) -> set["Models"]:
        """Returns all descendants (children, grandchildren, etc.) of this model."""
        with Session(MBDB.get_engine()) as session:
            descendant_query = (
                select(Models)
                .select_from(Models)
                .join(ModelsFrom, Models.hash == ModelsFrom.child)
                .where(ModelsFrom.parent == self.hash)
            )
            return set(session.execute(descendant_query).scalars().all())

    def get_lineage(self) -> dict[bytes, float]:
        """Returns all ancestors and their cached truth values from this model."""
        with Session(MBDB.get_engine()) as session:
            lineage_query = (
                select(ModelsFrom.parent, ModelsFrom.truth_cache)
                .where(ModelsFrom.child == self.hash)
                .order_by(ModelsFrom.level.desc())
            )

            results = session.execute(lineage_query).all()

            lineage = {parent: truth for parent, truth in results}
            lineage[self.hash] = self.truth

            return lineage

    def get_lineage_to_dataset(
        self, model: "Models"
    ) -> tuple[bytes, dict[bytes, float]]:
        """Returns the model lineage and cached truth values to a dataset."""
        if model.type != ModelType.DATASET.value:
            raise ValueError(
                f"Target model must be of type 'dataset', got {model.type}"
            )

        if self.hash == model.hash:
            return {model.hash: None}

        with Session(MBDB.get_engine()) as session:
            path_query = (
                select(ModelsFrom.parent, ModelsFrom.truth_cache)
                .join(Models, Models.hash == ModelsFrom.parent)
                .where(ModelsFrom.child == self.hash)
                .order_by(ModelsFrom.level.desc())
            )

            results = session.execute(path_query).all()

            if not any(parent == model.hash for parent, _ in results):
                raise ValueError(
                    f"No path exists between model {self.name} and dataset {model.name}"
                )

            lineage = {parent: truth for parent, truth in results}
            lineage[self.hash] = self.truth

            return lineage


class Sources(CountMixin, MBDB.MatchboxBase):
    """Table of sources of data for Matchbox."""

    __tablename__ = "sources"

    # Columns
    model = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )
    schema = Column(VARCHAR, nullable=False)
    table = Column(VARCHAR, nullable=False)
    id = Column(VARCHAR, nullable=False)

    # Relationships
    dataset_model = relationship("Models", back_populates="source")
    clusters = relationship("Clusters", back_populates="source")

    @classmethod
    def list(cls) -> list["Sources"]:
        with Session(MBDB.get_engine()) as session:
            return session.query(cls).all()


class Contains(CountMixin, MBDB.MatchboxBase):
    """Cluster lineage table."""

    __tablename__ = "contains"

    # Columns
    parent = Column(
        BYTEA, ForeignKey("clusters.hash", ondelete="CASCADE"), primary_key=True
    )
    child = Column(
        BYTEA, ForeignKey("clusters.hash", ondelete="CASCADE"), primary_key=True
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
    hash = Column(BYTEA, primary_key=True)
    dataset = Column(BYTEA, ForeignKey("sources.model"), nullable=True)
    # Uses array as source data may have identical rows. We can't control this
    # Must be indexed or PostgreSQL incorrectly tries to use nested joins
    # when retrieving small datasets in query() -- extremely slow
    id = Column(ARRAY(VARCHAR(36)), index=True, nullable=True)

    # Relationships
    source = relationship("Sources", back_populates="clusters")
    probabilities = relationship(
        "Probabilities", back_populates="proposes", cascade="all, delete-orphan"
    )
    children = relationship(
        "Clusters",
        secondary=Contains.__table__,
        primaryjoin="Clusters.hash == Contains.parent",
        secondaryjoin="Clusters.hash == Contains.child",
        backref="parents",
    )

    # Constraints and indices
    __table_args__ = (Index("ix_clusters_id_gin", id, postgresql_using="gin"),)


class Probabilities(CountMixin, MBDB.MatchboxBase):
    """Table of probabilities that a cluster merge is correct, according to a model."""

    __tablename__ = "probabilities"

    # Columns
    model = Column(
        BYTEA, ForeignKey("models.hash", ondelete="CASCADE"), primary_key=True
    )
    cluster = Column(
        BYTEA, ForeignKey("clusters.hash", ondelete="CASCADE"), primary_key=True
    )
    probability = Column(FLOAT, nullable=False)

    # Relationships
    proposed_by = relationship("Models", back_populates="probabilities")
    proposes = relationship("Clusters", back_populates="probabilities")

    # Constraints
    __table_args__ = (
        CheckConstraint("probability BETWEEN 0 AND 1", name="valid_probability"),
    )
