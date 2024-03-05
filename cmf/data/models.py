from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from cmf.data.clusters import clusters_association
from cmf.data.db import CMFBase
from cmf.data.dedupe import DDupeProbabilities
from cmf.data.link import LinkProbabilities
from cmf.data.mixin import SHA1Mixin

if TYPE_CHECKING:
    from cmf.data import Clusters


class Models(SHA1Mixin, CMFBase):
    __tablename__ = "cmf__models"
    __table_args__ = (UniqueConstraint("name"),)

    name: Mapped[str]
    description: Mapped[str]
    deduplicates: Mapped[Optional[bytes]] = mapped_column(
        ForeignKey("cmf__source_dataset.uuid")
    )

    # ORM Many to Many pattern
    # https://docs.sqlalchemy.org/en/20/orm/
    # basic_relationships.html#many-to-many
    creates: WriteOnlyMapped["Clusters"] = relationship(
        secondary=clusters_association, back_populates="created_by"
    )

    # Association object pattern
    # https://docs.sqlalchemy.org/en/20/orm
    # /basic_relationships.html#association-object
    proposes_dedupes: WriteOnlyMapped["DDupeProbabilities"] = relationship(
        back_populates="proposed_by", passive_deletes=True
    )
    proposes_links: WriteOnlyMapped["LinkProbabilities"] = relationship(
        back_populates="proposed_by", passive_deletes=True
    )

    # This approach taken from the SQLAlchemy examples
    # https://github.com/sqlalchemy/sqlalchemy/
    # blob/main/examples/graphs/directed_graph.py
    child_edges: Mapped[List["ModelsFrom"]] = relationship(
        back_populates="child_model",
        primaryjoin="Models.sha1 == ModelsFrom.child",
        cascade="all, delete",
        passive_deletes=True,
    )
    parent_edges: Mapped[List["ModelsFrom"]] = relationship(
        back_populates="parent_model",
        primaryjoin="Models.sha1 == ModelsFrom.parent",
        cascade="all, delete",
        passive_deletes=True,
    )

    def parent_neighbours(self):
        return [x.parent_model for x in self.child_edges]

    def child_neighbours(self):
        return [x.child_model for x in self.parent_edges]


# From


class ModelsFrom(CMFBase):
    __tablename__ = "cmf__models_from"
    __table_args__ = (UniqueConstraint("parent", "child"),)

    # Using PostgreSQL delete cascade to handle model deletion correctly
    # https://docs.sqlalchemy.org/en/20/orm/
    # cascades.html#using-foreign-key-on-delete-cascade-with-orm-relationships
    parent: Mapped[bytes] = mapped_column(
        ForeignKey("cmf__models.sha1", ondelete="CASCADE"), primary_key=True
    )
    child: Mapped[bytes] = mapped_column(
        ForeignKey("cmf__models.sha1", ondelete="CASCADE"), primary_key=True
    )

    child_model = relationship(
        Models,
        primaryjoin="ModelsFrom.child == Models.sha1",
        back_populates="child_edges",
    )
    parent_model = relationship(
        Models,
        primaryjoin="ModelsFrom.parent == Models.sha1",
        back_populates="parent_edges",
    )

    def __init__(self, parent, child):
        self.parent_model = parent
        self.child_model = child
