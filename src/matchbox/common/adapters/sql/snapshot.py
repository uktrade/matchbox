"""Generic table dump/restore, shared across relational backends.

Handles only data movement: selecting every row and base64-encoding bytes
columns for JSON-safety, or decoding and bulk-inserting them back.
Engine-specific concerns stay with the caller.
"""

import base64
from typing import Any

from sqlalchemy import Table, insert, select
from sqlalchemy.orm import Session


def dump_tables(
    session: Session, tables: list[Table]
) -> dict[str, list[dict[str, Any]]]:
    """Dump every row of the given tables to a JSON-safe dict, keyed by table name.

    Args:
        session: Database session.
        tables: The tables to dump, in any order.

    Returns:
        A dict mapping table name to its rows, each row a dict of column
            name to value. Bytes values are wrapped as {"base64": ...}.
    """
    data: dict[str, list[dict[str, Any]]] = {}

    for table in tables:
        records = session.execute(select(table)).mappings().all()

        table_data = []
        for record in records:
            record_dict = dict(record)
            for key, value in record_dict.items():
                if isinstance(value, bytes):
                    record_dict[key] = {
                        "base64": base64.b64encode(value).decode("ascii")
                    }
            table_data.append(record_dict)

        data[table.name] = table_data

    return data


def restore_tables(
    session: Session,
    tables: list[Table],
    data: dict[str, list[dict[str, Any]]],
    batch_size: int = 10_000,
) -> None:
    """Restore rows into the given tables, in order, from a dump_tables() dict.

    Args:
        session: Database session.
        tables: The tables to restore, in dependency order (parents first).
        data: A dict as produced by dump_tables().
        batch_size: The number of records to insert per batch.

    Raises:
        ValueError: If a table isn't present in data.
    """
    for table in tables:
        if table.name not in data:
            raise ValueError(f"Invalid: Table {table.name} not found in snapshot.")

        records = data[table.name]
        if not records:
            continue

        processed_records = []
        for record in records:
            processed_record = {}
            for key, value in record.items():
                if isinstance(value, dict) and "base64" in value:
                    processed_record[key] = base64.b64decode(value["base64"])
                else:
                    processed_record[key] = value
            processed_records.append(processed_record)

        for i in range(0, len(processed_records), batch_size):
            batch = processed_records[i : i + batch_size]
            session.execute(insert(table), batch)
            session.flush()
