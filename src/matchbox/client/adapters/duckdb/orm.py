"""ORM classes for the local DuckDB backend.

"steps" and "source_configs" are local-only (no run_id/collection/
upload_stage - one file is one run).

The five shared cluster tables are copied onto this module's own
LOCAL_METADATA via Table.to_metadata() (schema=None), rather than reusing
tables.METADATA directly, so that:

- table names never collide with the Postgres ORM's tables in a process
  that imports both.
- all of local's tables live in one MetaData object, so FK references
  between them resolve without a cross-metadata lookup failure.

Query building still references the original tables.Clusters etc.
objects (schema "mb") - SQL generation only needs matching table and
column names, not object identity.
"""

from typing import Optional

from sqlalchemy import (
    BigInteger,
    FetchedValue,
    ForeignKey,
    LargeBinary,
    MetaData,
    Text,
    func,
    select,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    declarative_base,
    mapped_column,
    relationship,
)

from matchbox.common.adapters.sql import tables
from matchbox.common.dtos import StepType
from matchbox.common.exceptions import MatchboxStepNotFoundError, MatchboxStepTypeError

LOCAL_METADATA = MetaData()
LocalBase = declarative_base(metadata=LOCAL_METADATA)


class CountMixin:
    """Adds `.count(session)` to an ORM class mapped to a whole table.

    Unlike Postgres's CountMixin, count() takes an explicit session: local
    has no global singleton session to pull one from, since a process can
    hold several independent MatchboxLocalDuckDB instances at once.
    """

    @classmethod
    def count(cls, session: Session) -> int:
        """Counts the number of rows in the table."""
        return session.execute(select(func.count()).select_from(cls)).scalar_one()


class Steps(LocalBase):
    """Local step registry. No run_id/collection/upload_stage: one file is one run."""

    __tablename__ = "steps"

    step_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=False,
        server_default=FetchedValue(),
    )
    name: Mapped[str] = mapped_column(Text, unique=True)
    type: Mapped[str] = mapped_column(Text)
    fingerprint: Mapped[bytes] = mapped_column(LargeBinary)

    source_config: Mapped[Optional["SourceConfigs"]] = relationship(
        back_populates="step", uselist=False
    )

    @classmethod
    def from_name(
        cls, session: Session, name: str, expected_type: StepType | None = None
    ) -> "Steps":
        """Resolve a step by name, optionally validating its type.

        Args:
            session: Database session.
            name: The step's name.
            expected_type: If given, raise if the step isn't of this type.

        Raises:
            MatchboxStepNotFoundError: If no step has this name.
            MatchboxStepTypeError: If expected_type is given and doesn't match.
        """
        step = session.execute(select(cls).where(cls.name == name)).scalar_one_or_none()
        if step is None:
            raise MatchboxStepNotFoundError(name=name)
        if expected_type is not None and step.type != expected_type.value:
            raise MatchboxStepTypeError(
                step_name=name,
                step_type=StepType(step.type),
                expected_step_types=[expected_type],
            )
        return step


class SourceConfigs(LocalBase):
    """Degenerate one-row-per-source table.

    Keeps the join shape identical to the server's, without needing the
    rest of a real SourceConfig.
    """

    __tablename__ = "source_configs"

    source_config_id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=False,
        server_default=FetchedValue(),
    )
    step_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("steps.step_id"), unique=True
    )

    step: Mapped["Steps"] = relationship(back_populates="source_config")


class RawData(LocalBase):
    """Catalog of raw warehouse rows per source step.

    Canonical local state - expensive to re-fetch, so kept until
    explicitly replaced. One row per source step, FK'd 1:1 to Steps: the
    actual rows live in a physical DuckDB table named by
    adapter._raw_data_table_name(source_step_id), never stored here as
    data - see adapter.py for why.
    """

    __tablename__ = "raw_data"

    source_step_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("steps.step_id"), primary_key=True, autoincrement=False
    )


class QueryCache(LocalBase):
    """Catalog of cleaned query results.

    Disposable: cleared wholesale whenever any step's data changes, since
    recomputing is cheap. cache_key is the opaque identity of a query
    definition (query config plus upstream fingerprints); cache_id is a
    sequence-backed surrogate used only to name the physical table
    (adapter._query_cache_table_name(cache_id)) - see adapter.py.
    """

    __tablename__ = "query_cache"

    cache_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False, server_default=FetchedValue()
    )
    cache_key: Mapped[str] = mapped_column(Text, unique=True)


class QueryCacheStep(LocalBase):
    """Bridging table: which steps a cached query result depends on.

    A Query isn't itself a DAG step (unlike Source/Model/Resolver), so it
    has no step_id of its own to hang cache invalidation off. This table
    records the steps (sources + resolver) a cached result was built
    from, so _cascade_invalidate can drop only the cache entries that
    actually depend on a changed step.
    """

    __tablename__ = "query_cache_steps"

    cache_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("query_cache.cache_id"), primary_key=True
    )
    step_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("steps.step_id"), primary_key=True
    )


class Clusters(CountMixin, LocalBase):
    """Table of indexed data and clusters that match it."""

    __table__ = tables.Clusters.to_metadata(LOCAL_METADATA, schema=None)


class ClusterSourceKey(CountMixin, LocalBase):
    """Table for storing source primary keys for clusters."""

    __table__ = tables.ClusterSourceKey.to_metadata(LOCAL_METADATA, schema=None)


class Contains(CountMixin, LocalBase):
    """Cluster lineage table."""

    __table__ = tables.Contains.to_metadata(LOCAL_METADATA, schema=None)


class ModelEdges(CountMixin, LocalBase):
    """Table of results for a model step."""

    __table__ = tables.ModelEdges.to_metadata(LOCAL_METADATA, schema=None)


class ResolverClusters(CountMixin, LocalBase):
    """Association table linking resolver steps to cluster IDs."""

    __table__ = tables.ResolverClusters.to_metadata(LOCAL_METADATA, schema=None)


LOCAL_TABLES = [
    Steps.__table__,
    SourceConfigs.__table__,
    RawData.__table__,
    QueryCache.__table__,
    QueryCacheStep.__table__,
]
SHARED_TABLES = [
    Clusters.__table__,
    ClusterSourceKey.__table__,
    Contains.__table__,
    ModelEdges.__table__,
    ResolverClusters.__table__,
]
