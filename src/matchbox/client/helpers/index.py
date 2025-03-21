"""Functions to index data sources to the Matchbox server."""

from sqlalchemy import Engine

from matchbox.client import _handler
from matchbox.common.sources import Source, SourceAddress, SourceColumn


def _process_columns(
    columns: list[str] | list[dict[str, dict[str, str]]] | None,
) -> tuple[SourceColumn]:
    if columns is None:
        return []

    if isinstance(columns[0], str):
        return (SourceColumn(name=column) for column in columns)

    return (
        SourceColumn(name=column["name"], alias=column["alias"], type=column["type"])
        for column in columns
    )


def index(
    full_name: str,
    db_pk: str,
    engine: Engine,
    resolution_name: str | None = None,
    columns: list[str] | list[dict[str, dict[str, str]]] | None = None,
    batch_size: int | None = None,
) -> None:
    """Indexes data in Matchbox.

    Args:
        full_name: the full name of the source
        db_pk: the primary key of the source
        engine: the engine to connect to a data warehouse
        resolution_name: a custom resolution name
            If missing, will use the default name for a `Source`
        columns: the columns to index
        batch_size: the size of each batch when fetching data from the warehouse,
            which helps reduce the load on the database. Default is None.

    Examples:
        ```python
        index("mb.test_orig", "id", engine=engine)
        ```
        ```python
        index("mb.test_cl2", "id", engine=engine, columns=["name", "age"])
        ```
        ```python
        index(
            "mb.test_cl2",
            "id",
            engine=engine,
            columns=[
                {"name": "name", "alias": "person_name", "type": "TEXT"},
                {"name": "age", "alias": "person_age", "type": "BIGINT"},
            ],
        )
        ```
        ```python
        index("mb.test_orig", "id", engine=engine, batch_size=10_000)
        ```
    """
    columns = _process_columns(columns)

    address = SourceAddress.compose(engine=engine, full_name=full_name)
    if resolution_name:
        source = Source(
            address=address,
            resolution_name=resolution_name,
            columns=columns,
            db_pk=db_pk,
        )
    else:
        source = Source(
            address=address,
            columns=columns,
            db_pk=db_pk,
        )

    source.set_engine(engine)
    if not columns:
        source = source.default_columns()

    _handler.index(source=source, batch_size=batch_size)
