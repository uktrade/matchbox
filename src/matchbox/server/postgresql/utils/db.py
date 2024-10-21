import contextlib
import cProfile
import io
import pstats
from itertools import islice
from typing import Any, Callable, Iterable, Tuple

import rustworkx as rx
from pg_bulk_ingest import Delete, Upsert, ingest
from sqlalchemy import Engine, Table
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import DeclarativeMeta, Session

from matchbox.server.postgresql.data import SourceDataset
from matchbox.server.postgresql.models import Models, ModelsFrom

# Retrieval


def get_model_subgraph(engine: Engine) -> rx.PyDiGraph:
    """Retrieves the model subgraph as a PyDiGraph."""
    G = rx.PyDiGraph()
    models = {}
    datasets = {}

    with Session(engine) as session:
        for dataset in session.query(SourceDataset).all():
            dataset_idx = G.add_node(
                {
                    "id": str(dataset.uuid),
                    "name": f"{dataset.db_schema}.{dataset.db_table}",
                    "type": "dataset",
                }
            )
            datasets[dataset.uuid] = dataset_idx

        for model in session.query(Models).all():
            model_idx = G.add_node(
                {"id": str(model.sha1), "name": model.name, "type": "model"}
            )
            models[model.sha1] = model_idx
            if model.deduplicates is not None:
                dataset_idx = datasets.get(model.deduplicates)
                _ = G.add_edge(model_idx, dataset_idx, {"type": "deduplicates"})

        for edge in session.query(ModelsFrom).all():
            parent_idx = models.get(edge.parent)
            child_idx = models.get(edge.child)
            _ = G.add_edge(parent_idx, child_idx, {"type": "from"})

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
    "Batch data into lists of length n. The last batch may be shorter."
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def data_to_batch(
    records: list[tuple], table: Table, batch_size: int
) -> Callable[[str], Tuple[Any]]:
    """Constructs a batches function for any dataframe and table."""

    def _batches(
        high_watermark,  # noqa ARG001 required for pg_bulk_ingest
    ) -> Iterable[Tuple[None, None, Iterable[Tuple[Table, tuple]]]]:
        for batch in batched(records, batch_size):
            yield None, None, ((table, t) for t in batch)

    return _batches


def batch_ingest(
    records: list[tuple],
    table: Table | DeclarativeMeta,
    conn: Connection,
    batch_size: int,
) -> None:
    """Batch ingest records into a database table."""

    if isinstance(table, DeclarativeMeta):
        table = table.__table__

    fn_batch = data_to_batch(
        records=records,
        table=table,
        batch_size=batch_size,
    )

    ingest(
        conn=conn,
        metadata=table.metadata,
        batches=fn_batch,
        upsert=Upsert.IF_PRIMARY_KEY,
        delete=Delete.OFF,
    )