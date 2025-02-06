import contextlib
import cProfile
import io
import os
import pstats
from itertools import islice
from typing import Any, Callable, Iterable
from datetime import datetime

import pyarrow as pa
import adbc_driver_postgresql.dbapi
from adbc_driver_manager.dbapi import Connection as ADBCConnection
from adbc_driver_manager import DatabaseError as ADBCDatabaseError

from pg_bulk_ingest import Delete, Upsert, ingest
from sqlalchemy import Engine, Index, MetaData, Table, func, text
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import DeclarativeMeta, Session
from sqlalchemy.exc import DatabaseError as AlchemyDatabaseError

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


MB__POSTGRES__PASSWORD = os.environ["MB__POSTGRES__PASSWORD"]
MB__POSTGRES__PORT = os.environ["MB__POSTGRES__PORT"]
MB__POSTGRES__USER = os.environ["MB__POSTGRES__USER"]
MB__POSTGRES__DATABASE = os.environ["MB__POSTGRES__DATABASE"]
MB__POSTGRES__HOST = os.environ["MB__POSTGRES__HOST"]
MB__POSTGRES__SCHEMA = os.environ["MB__POSTGRES__DB_SCHEMA"]


POSTGRESQL_URI = f"postgresql://{MB__POSTGRES__USER}:{MB__POSTGRES__PASSWORD}@{MB__POSTGRES__HOST}:{MB__POSTGRES__PORT}/{MB__POSTGRES__DATABASE}"

def adbc_ingest_data(clusters:pa.Table, contains:pa.Table, probabilities:pa.Table, engine:Engine, resolution_id:int) -> bool:
    """ Ingest data from PostgreSQL using pyarrow adbc ingest.
    Args: clusters: pa.Table, contains: pa.Table, probabilities: pa.Table, engine: Engine
    """

    with engine.connect() as alchemy_conn:
        suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        if _adbc_insert_data(clusters, contains, probabilities, suffix, alchemy_conn, resolution_id):
            return _create_adbc_table_constraints(suffix, alchemy_conn)
        else:
            return False

def _create_adbc_table_constraints(db_schema:str, sufix:str, conn:Connection) -> bool:
    """ Creating primary and secondary keys indexes and constraints.
    Args: db_schema: str, the name of the schema
    """
    # Cluster
    _run_query(f"ALTER TABLE {db_schema}.clusters_{sufix} ADD PRIMARY KEY (cluster_id)", conn)
    _run_query(f"""ALTER TABLE {db_schema}.probabilities_{sufix} ADD PRIMARY KEY (resolution, "cluster")""", conn)
    _run_query(f"CREATE UNIQUE INDEX cluster_hash_index_{sufix} ON {db_schema}.clusters_{sufix} USING btree (cluster_hash)", conn)
   # _run_query(f"CREATE UNIQUE INDEX clusters_adbc_clusters_is_{sufix} ON {db_schema}.clusters_{sufix} USING btree (cluster_id)", conn)
    _run_query(f"CREATE INDEX ix_clusters_id_gin_{sufix} ON {db_schema}.clusters_{sufix} USING gin (source_pk)", conn)
    _run_query(f"CREATE INDEX ix_mb_clusters_source_pk_{sufix} ON {db_schema}.clusters_{sufix} USING btree (source_pk)", conn)

    # Contains
    _run_query(f"CREATE UNIQUE INDEX ix_contains_child_parent_{sufix} ON {db_schema}.contains_{sufix} USING btree (child, parent)", conn)
    _run_query(f"CREATE UNIQUE INDEX ix_contains_parent_child_{sufix} ON {db_schema}.contains_{sufix} USING btree (parent, child)", conn)

    # Foreign keys
    _run_query(f"ALTER TABLE {db_schema}.clusters_{sufix} ADD CONSTRAINT clusters_dataset_fkey FOREIGN KEY (dataset) REFERENCES {db_schema}.sources(resolution_id)", conn)
    _run_query(f"""ALTER TABLE {db_schema}."contains_{sufix}" ADD CONSTRAINT contains_child_fkey FOREIGN KEY (child) REFERENCES {db_schema}.clusters_{sufix}(cluster_id) ON DELETE CASCADE""", conn)
    _run_query(f"""ALTER TABLE {db_schema}."contains_{sufix}" ADD CONSTRAINT contains_parent_fkey FOREIGN KEY (parent) REFERENCES {db_schema}.clusters_{sufix}(cluster_id) ON DELETE CASCADE""", conn)
    _run_query(f"""ALTER TABLE {db_schema}.probabilities_{sufix} ADD CONSTRAINT probabilities_cluster_fkey FOREIGN KEY ("cluster") REFERENCES {db_schema}.clusters_{sufix}(cluster_id) ON DELETE CASCADE""", conn)
    _run_query(f"ALTER TABLE {db_schema}.probabilities_{sufix} ADD CONSTRAINT probabilities_resolution_fkey FOREIGN KEY (resolution) REFERENCES {db_schema}.resolutions(resolution_id) ON DELETE CASCADE", conn)

    _run_queries([
        f"""DROP TABLE IF EXISTS {db_schema}.clusters""",
        f"""DROP TABLE IF EXISTS {db_schema}.contains""",
        f"""DROP TABLE IF EXISTS {db_schema}.probabilities""",

        f"""ALTER TABLE {db_schema}.clusters_{sufix} RENAME TO clusters""",
        f"""ALTER TABLE {db_schema}.contains_{sufix} RENAME TO contains""",
        f"""ALTER TABLE {db_schema}.probabilities_{sufix} RENAME TO probabilities"""
    ], conn)
    return True

def _adbc_insert_data(clusters:pa.Table, contains:pa.Table, probabilities:pa.Table, suffix:str, alchemy_conn:Connection, resolution_id:int) -> bool:
    # TODO: try except proper exception type from adbc
    with adbc_driver_postgresql.dbapi.connect(POSTGRESQL_URI) as conn:
        try:
            _run_query(f"CREATE TABLE clusters_{suffix} AS SELECT * FROM clusters", alchemy_conn)
            _save_to_postgresql(
                table=clusters,
                conn=conn,
                schema=MB__POSTGRES__SCHEMA,
                table_name=f"clusters_{suffix}",
            )
            _run_query(f"CREATE TABLE contains_{suffix} AS SELECT * FROM contains", alchemy_conn)
            _save_to_postgresql(
                table=contains,
                conn=conn,
                schema=MB__POSTGRES__SCHEMA,
                table_name=f"contains_{suffix}",
            )
            _run_query(f"CREATE TABLE probabilities_{suffix} AS SELECT * FROM probabilities WHERE resolution != {resolution_id}", alchemy_conn)
            _save_to_postgresql(
                table=probabilities,
                conn=conn,
                schema=MB__POSTGRES__SCHEMA,
                table_name=f"probabilities_{suffix}",
            )
            conn.commit()
            return True
        except ADBCConnection as e:
            return False
        except AlchemyDatabaseError as e:
            return False

def _run_query(query: str,conn:Connection) -> None:
    conn.execute(text(query))
    conn.commit()


def _run_queries(queries: list[str], conn:Connection) -> None:
    conn.begin()
    for query in queries:
        conn.execute(text(query))
    conn.commit()

def _save_to_postgresql(
        table: pa.Table, conn: ADBCConnection, schema: str, table_name: str
):
    """
    Saves a PyArrow Table to PostgreSQL using ADBC.
    """
    with conn.cursor() as cursor:
        # Convert PyArrow Table to Arrow RecordBatchStream for efficient transfer
        batch_reader = pa.RecordBatchReader.from_batches(
            table.schema, table.to_batches()
        )
        cursor.adbc_ingest(
            table_name=table_name,
            data=batch_reader,
            mode="append",
            db_schema_name=schema,
        )
