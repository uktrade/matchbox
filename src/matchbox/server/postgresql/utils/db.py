import contextlib
import cProfile
import io
import pstats
from itertools import islice
from typing import Any, Callable, Iterable

from pg_bulk_ingest import Delete, Upsert, ingest
from sqlalchemy import Engine, Index, MetaData, Table, func
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import DeclarativeMeta, Session

from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeType,
)
from matchbox.server.postgresql.orm import (
    ResolutionFrom,
    Resolutions,
)

# Retrieval


def resolve_model_name(model: str, engine: Engine) -> Resolutions:
    """Resolves a model name to a Resolution object.

    Args:
        model: The name of the model to resolve.

    Raises:
        MatchboxResolutionNotFoundError: If the model doesn't exist.
    """
    with Session(engine) as session:
        if resolution := session.query(Resolutions).filter_by(name=model).first():
            return resolution
        raise MatchboxResolutionNotFoundError(resolution_name=model)


def get_resolution_graph(engine: Engine) -> ResolutionGraph:
    """Retrieves the resolution graph."""
    G = ResolutionGraph(nodes=set(), edges=set())
    with Session(engine) as session:
        for resolution in session.query(Resolutions).all():
            G.nodes.add(
                ResolutionNode(
                    id=resolution.resolution_id,
                    name=resolution.name,
                    type=ResolutionNodeType(resolution.type),
                )
            )

        for edge in (
            session.query(ResolutionFrom).filter(ResolutionFrom.level == 1).all()
        ):
            G.edges.add(ResolutionEdge(parent=edge.parent, child=edge.child))

    return G


# SQLAlchemy profiling


@contextlib.contextmanager
def sqa_profiled():
    """SQLAlchemy profiler.

    Taken directly from their docs:
    https://docs.sqlalchemy.org/en/20/faq/performance.html#query-profiling
    """
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    # uncomment this to see who's calling what
    # ps.print_callers()
    print(s.getvalue())


# Misc


def batched(iterable: Iterable, n: int) -> Iterable:
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def data_to_batch(
    records: list[tuple], table: Table, batch_size: int
) -> Callable[[str], tuple[Any]]:
    """Constructs a batches function for any dataframe and table."""

    def _batches(
        high_watermark,  # noqa ARG001 required for pg_bulk_ingest
    ) -> Iterable[tuple[None, None, Iterable[tuple[Table, tuple]]]]:
        for batch in batched(records, batch_size):
            yield None, None, ((table, t) for t in batch)

    return _batches


def isolate_table(table: DeclarativeMeta) -> tuple[MetaData, Table]:
    """Creates an isolated copy of a SQLAlchemy table.

    This is used to prevent pg_bulk_ingest from attempting to drop unrelated tables
    in the same schema. The function creates a new Table instance with:

    * A fresh MetaData instance
    * Copied columns
    * Recreated indices properly bound to the new table

    Args:
        table: The DeclarativeMeta class whose table should be isolated

    Returns:
        A tuple of:
            * The isolated SQLAlchemy MetaData
            * A new SQLAlchemy Table instance with all columns and indices
    """
    isolated_metadata = MetaData(schema=table.__table__.schema)

    isolated_table = Table(
        table.__table__.name,
        isolated_metadata,
        *[c._copy() for c in table.__table__.columns],
        schema=table.__table__.schema,
    )

    for idx in table.__table__.indexes:
        Index(
            idx.name,
            *[isolated_table.c[col.name] for col in idx.columns],
            **{k: v for k, v in idx.kwargs.items()},
        )

    return isolated_metadata, isolated_table


def hash_to_hex_decode(hash: bytes) -> bytes:
    """A workround for PostgreSQL so we can compile the query and use ConnectorX."""
    return func.decode(hash.hex(), "hex")


def batch_ingest(
    records: list[tuple[Any]],
    table: DeclarativeMeta,
    conn: Connection,
    batch_size: int,
) -> None:
    """Batch ingest records into a database table.

    We isolate the table and metadata as pg_bulk_ingest will try and drop unrelated
    tables if they're in the same schema.
    """
    isolated_metadata, isolated_table = isolate_table(table=table)

    fn_batch = data_to_batch(
        records=records,
        table=isolated_table,
        batch_size=batch_size,
    )

    ingest(
        conn=conn,
        metadata=isolated_metadata,
        batches=fn_batch,
        upsert=Upsert.IF_PRIMARY_KEY,
        delete=Delete.OFF,
    )
