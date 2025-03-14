"""Functions to index data sources to the Matchbox server."""

from sqlalchemy import Engine

from matchbox.client import _handler
from matchbox.common.sources import Source, SourceAddress, SourceColumn


def _process_columns(
    columns: list[str] | list[dict[str, dict[str, str]]] | None,
) -> list[SourceColumn]:
    if columns is None:
        return []

    if isinstance(columns[0], str):
        return [SourceColumn(name=column) for column in columns]

    return [
        SourceColumn(name=column["name"], alias=column["alias"], type=column["type"])
        for column in columns
    ]


def index(
    full_name: str,
    db_pk: str,
    engine: Engine,
    columns: list[str] | list[dict[str, dict[str, str]]] | None = None,
) -> None:
    """Indexes data in Matchbox.

    Args:
        full_name: the full name of the source
        db_pk: the primary key of the source
        engine: the engine to connect to a data warehouse
        columns: the columns to index

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
    """
    columns = _process_columns(columns)

    source = Source(
        address=SourceAddress.compose(engine=engine, full_name=full_name),
        columns=columns,
        db_pk=db_pk,
    ).set_engine(engine)

    if not columns:
        source = source.default_columns()

    _handler.index(source=source, data_hashes=source.hash_data())
