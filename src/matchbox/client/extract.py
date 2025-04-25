"""Functions to extract data out of the Matchbox server."""

from pyarrow import Table as ArrowTable
from sqlalchemy import Engine, String, func, select, text

from matchbox.client import _handler
from matchbox.common.sources import Source, SourceAddress


def _create_view_definition(
    column_names: list[dict[str, str]],
    db_pks: dict[str, str],
    mapping_table: str,
    engine: Engine,
) -> str:
    selection = [text("id")]

    for cn in column_names:
        selection.append(
            text(f"{cn['table_name']}.{cn['column_name']} as {cn['combined_name']}")
        )

    query = select(*selection).select_from(text(mapping_table))

    for table_name, db_pk in db_pks.items():
        query = query.join(
            text(table_name),
            func.cast(text(f"{table_name}.{db_pk}"), String)
            == func.cast(text(f"{mapping_table}.{table_name}_{db_pk}"), String),
            isouter=True,
        )

    return str(query.compile(dialect=engine.dialect))


def _combined_colname(source: Source, col_name: str):
    return source.address.full_name.replace(".", "_") + "_" + col_name


def sql_interface(
    resolution_name: str,
    engine: Engine,
    mapping_table: str,
) -> tuple[ArrowTable, ArrowTable, str]:
    """Generate snapshot and DDL implmenting SQL interface to results.

    Args:
        resolution_name: Name of the resolution from which to generate results
        engine: SQLAlchemy engine of warehouse running SQL interface
        mapping_table: The name (+ schema) where the mapping table will be written.

    Returns:
        A tuple of 3 items containing:

            * A PyArrow table mapping from Matchbox IDs to all source PKs
            * A PyArrow table mapping sources and columns to column names in the view
            * The DQL definition of a view to use for querying results in SQL

    """
    # Get all sources in scope of the resolution
    res_sources = _handler.get_resolution_sources(resolution_name=resolution_name)

    # Filter only sources compatible with engine
    warehouse_hash_b64 = SourceAddress.compose(
        full_name="", engine=engine
    ).warehouse_hash_b64

    sources = [
        s.set_engine(engine)
        for s in res_sources
        if s.address.warehouse_hash_b64 == warehouse_hash_b64
    ]

    source_mb_ids: list[ArrowTable] = []
    column_names: list[dict[str, str]] = []
    db_pks: dict[str, str] = {}

    for s in sources:
        # Get Matchbox IDs from backend
        source_mb_ids.append(
            _handler.query(
                source_address=s.address,
                resolution_name=resolution_name,
            )
        )
        # Get remote columns from warehouse
        cols = s.get_remote_columns()
        column_names += [
            {
                "table_name": s.address.full_name,
                "column_name": c,
                "combined_name": _combined_colname(s, c),
                "db_pk": s.db_pk,
            }
            for c in cols
        ]

        db_pks[s.address.full_name] = s.db_pk

    # Join Matchbox IDs to form mapping table
    mapping = source_mb_ids[0]
    mapping = mapping.rename_columns(
        {
            "source_pk": _combined_colname(
                sources[0], db_pks[sources[0].address.full_name]
            )
        }
    )
    for s, mb_ids in zip(sources[1:], source_mb_ids[1:], strict=True):
        mapping = mapping.join(right_table=mb_ids, keys="id", join_type="full outer")
        mapping = mapping.rename_columns(
            {"source_pk": _combined_colname(s, db_pks[s.address.full_name])}
        )

    # Get SQL query for view
    view_definition = _create_view_definition(
        column_names=column_names,
        db_pks=db_pks,
        mapping_table=mapping_table,
        engine=engine,
    )

    # Convert column names dataframe
    column_names = ArrowTable.from_pylist(column_names)

    return mapping, column_names, view_definition
