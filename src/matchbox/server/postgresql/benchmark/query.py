from sqlalchemy.orm import Session

from matchbox.common.sources import SourceAddress
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Resolutions,
    Sources,
)
from matchbox.server.postgresql.utils.query import (
    _resolve_cluster_hierarchy,
)


def compile_query_sql(point_of_truth: str, source_address: SourceAddress) -> str:
    """Compiles a the SQL for query() based on a single point of truth and dataset.

    Args:
        point_of_truth (string): The name of the resolution to use, like "linker_1"
        dataset_name (string): The name of the dataset to retrieve, like "dbt.companies"

    Returns:
        A compiled PostgreSQL query, including semicolon, ready to run on Matchbox
    """
    engine = MBDB.get_engine()

    with Session(engine) as session:
        point_of_truth_resolution = (
            session.query(Resolutions)
            .filter(Resolutions.name == point_of_truth)
            .first()
        )
        dataset_id = (
            session.query(Resolutions.resolution_id)
            .join(Sources, Sources.resolution_id == Resolutions.resolution_id)
            .filter(
                Sources.full_name == source_address.full_name,
                Sources.warehouse_hash == source_address.warehouse_hash,
            )
            .scalar()
        )

        id_query = _resolve_cluster_hierarchy(
            dataset_id=dataset_id,
            resolution=point_of_truth_resolution,
            threshold=None,
            engine=engine,
        )

    compiled_stmt = id_query.compile(
        dialect=engine.dialect, compile_kwargs={"literal_binds": True}
    )

    return str(compiled_stmt) + ";"
