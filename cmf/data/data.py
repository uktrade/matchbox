from typing import List

from sqlalchemy import ForeignKey, UniqueConstraint, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY

from uuid import UUID

from cmf.data.db import CMFBase
from cmf.data.mixin import UUIDMixin, SHA1Mixin


class SourceDataset(UUIDMixin, CMFBase):
    __tablename__ = "source_dataset"
    __table_args__ = (UniqueConstraint("db_schema", "db_table"),)

    db_schema: Mapped[str]
    db_id: Mapped[str]
    db_table: Mapped[str]

    data: Mapped[List["SourceData"]] = relationship(back_populates="parent_dataset")


class SourceData(SHA1Mixin, CMFBase):
    __tablename__ = "source_data"
    __table_args__ = (UniqueConstraint("id", "dataset"),)

    id: Mapped[List[str]] = mapped_column(
        ARRAY(String), index=True
    )  # Literally required for query() to work
    dataset: Mapped[UUID] = mapped_column(ForeignKey("source_dataset.uuid"))

    parent_dataset: Mapped["SourceDataset"] = relationship(back_populates="data")
