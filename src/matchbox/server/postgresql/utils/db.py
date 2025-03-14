"""General utilities for the PostgreSQL backend."""

import base64
import contextlib
import cProfile
import io
import pstats
from itertools import islice
from typing import Any, Callable, Iterable

from pg_bulk_ingest import Delete, Upsert, ingest
from sqlalchemy import Engine, Index, MetaData, Table, func, inspect, select
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import DeclarativeMeta, Session

from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
)
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
    ResolutionNodeType,
)
from matchbox.server.base import MatchboxBackends, MatchboxSnapshot
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Sources,
)

# Retrieval


def resolve_model_name(model: str, engine: Engine) -> Resolutions:
    """Resolves a model name to a Resolution object.

    Args:
        model: The name of the model to resolve.
        engine: The database engine.

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


# Data management


def dump(engine: Engine) -> MatchboxSnapshot:
    """Dumps the entire database to a snapshot.

    Args:
        engine: The database engine.

    Returns:
        A MatchboxSnapshot object of type "postgres" with the database's
            current state.
    """
    tables = {
        "resolutions": Resolutions,
        "resolution_from": ResolutionFrom,
        "sources": Sources,
        "clusters": Clusters,
        "contains": Contains,
        "probabilities": Probabilities,
    }

    data = {}

    with Session(engine) as session:
        for table_name, model in tables.items():
            # Query all records from this table
            records = session.execute(select(model)).scalars().all()

            # Convert each record to a dictionary
            table_data = []
            for record in records:
                record_dict = {}
                for column in inspect(model).columns:
                    value = getattr(record, column.name)

                    # Store bytes as nested dictionary with encoding format
                    if isinstance(value, bytes):
                        record_dict[column.name] = {
                            "base64": base64.b64encode(value).decode("ascii")
                        }
                    else:
                        record_dict[column.name] = value

                table_data.append(record_dict)

            data[table_name] = table_data

    return MatchboxSnapshot(backend_type=MatchboxBackends.POSTGRES, data=data)


def restore(engine: Engine, snapshot: MatchboxSnapshot, batch_size: int) -> None:
    """Restores the database from a snapshot.

    Args:
        engine: The database engine.
        snapshot: A MatchboxSnapshot object of type "postgres" with the
            database's state
        batch_size: The number of records to insert in each batch

    Raises:
        ValueError: If the snapshot is missing data
    """
    table_map = {
        "resolutions": Resolutions,
        "resolution_from": ResolutionFrom,
        "sources": Sources,
        "clusters": Clusters,
        "contains": Contains,
        "probabilities": Probabilities,
    }

    table_order = [
        "resolutions",
        "resolution_from",
        "sources",
        "clusters",
        "contains",
        "probabilities",
    ]

    with Session(engine) as session:
        # Process tables in order
        for table_name in table_order:
            if table_name not in snapshot.data:
                raise ValueError(f"Invalid: Table {table_name} not found in snapshot.")

            model = table_map[table_name]
            records = snapshot.data[table_name]

            if not records:
                continue

            # Process records for insertion
            processed_records = []
            for record in records:
                processed_record = {}

                for key, value in record.items():
                    # Check if the value is a dictionary with encoding format
                    if isinstance(value, dict) and "base64" in value:
                        processed_record[key] = base64.b64decode(value["base64"])
                    else:
                        processed_record[key] = value

                processed_records.append(processed_record)

            # Insert
            for i in range(0, len(processed_records), batch_size):
                batch = processed_records[i : i + batch_size]
                session.bulk_insert_mappings(model, batch)
                session.flush()

        session.commit()


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
        high_watermark,  # noqa: ARG001
    ) -> Iterable[tuple[None, None, Iterable[tuple[Table, tuple]]]]:
        # high_watermark required for pg_bulk_ingest
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
