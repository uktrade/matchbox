import pyarrow as pa
import pytest
from sqlalchemy import BIGINT, TEXT, Column, MetaData
from sqlalchemy.orm import declarative_base

from matchbox.common.exceptions import MatchboxDatabaseWriteError
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin
from matchbox.server.postgresql.orm import PKSpace
from matchbox.server.postgresql.utils.db import (
    ingest_to_temporary_table,
    large_append,
)


@pytest.mark.docker
def test_reserve_id_block(
    matchbox_postgres: MatchboxPostgres,  # Reset DB
):
    """Test that we can atomically reserve ID blocks."""
    first_cluster_id = PKSpace.reserve_block("clusters", 42)
    second_cluster_id = PKSpace.reserve_block("clusters", 42)

    assert first_cluster_id == second_cluster_id - 42

    first_cluster_keys_id = PKSpace.reserve_block("cluster_keys", 42)
    second_cluster_keys_id = PKSpace.reserve_block("cluster_keys", 42)

    assert first_cluster_keys_id == second_cluster_keys_id - 42

    with pytest.raises(ValueError):
        PKSpace.reserve_block("clusters", 0)


@pytest.mark.docker
def test_large_append(
    matchbox_postgres: MatchboxPostgres,  # will drop dummy table
):
    """Test appending large data to a table."""
    engine = MBDB.get_engine()
    metadata = MetaData(schema=MBDB.MatchboxBase.metadata.schema)

    # Initialise DummyTable to which we'll ingest
    class DummyTable(CountMixin, declarative_base(metadata=metadata)):
        __tablename__ = "dummytable"
        key = Column(BIGINT, primary_key=True)
        foo = Column(TEXT, nullable=False)

    metadata.create_all(engine, tables=[DummyTable.__table__])
    metadata.reflect(engine)
    original_tables = len(metadata.tables)

    # No auto-commit
    with MBDB.get_adbc_connection() as adbc_connection:
        large_append(
            data=pa.Table.from_pylist([{"foo": "val"}]),
            table_class=DummyTable,
            adbc_connection=adbc_connection,
        )
    assert DummyTable.count() == 0

    # Ingest data with manual keys and chunking
    with MBDB.get_adbc_connection() as adbc_connection:
        large_append(
            data=pa.Table.from_pylist(
                [{"key": 0, "foo": "val1"}],
            ),
            table_class=DummyTable,
            adbc_connection=adbc_connection,
            max_chunksize=100,
        )
        adbc_connection.commit()

    # Ingest data without keys and no chunking
    with MBDB.get_adbc_connection() as adbc_connection:
        large_append(
            data=pa.Table.from_pylist([{"foo": "val2"}]),
            table_class=DummyTable,
            adbc_connection=adbc_connection,
        )
        adbc_connection.commit()

    # Both rows were fine
    assert DummyTable.count() == 2
    with MBDB.get_session() as session:
        second_id = (
            session.query(DummyTable.key).filter(DummyTable.foo == "val2").scalar()
        )
    assert 0 < second_id

    # Upserting not allowed
    with (
        pytest.raises(MatchboxDatabaseWriteError),
        MBDB.get_adbc_connection() as adbc_connection,
    ):
        large_append(
            data=pa.Table.from_pylist([{"key": 0, "foo": "val3"}]),
            table_class=DummyTable,
            adbc_connection=adbc_connection,
        )

    # Failed ingestion has no effect
    assert DummyTable.count() == 2
    assert len(metadata.tables) == original_tables

    # Columns not available in the target table are rejected
    with (
        pytest.raises(ValueError, match="does not have columns"),
        MBDB.get_adbc_connection() as adbc_connection,
    ):
        large_append(
            data=pa.Table.from_pylist([{"key": 10, "bar": "val3"}]),
            table_class=DummyTable,
            adbc_connection=adbc_connection,
        )


@pytest.mark.docker
def test_ingest_to_temporary_table(
    matchbox_postgres: MatchboxPostgres,  # will drop dummy table
):
    """Test temporary table creation, data ingestion, and automatic cleanup."""
    from sqlalchemy.dialects.postgresql import BIGINT, TEXT

    # Create sample arrow data
    data = pa.Table.from_pylist(
        [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"},
        ]
    )

    schema_name = MBDB.MatchboxBase.metadata.schema
    table_name = "test_temp_ingest"

    # Define the column types for the temporary table
    column_types = {
        "id": BIGINT,
        "value": TEXT,
    }

    # Use the context manager to create and populate a temporary table
    with ingest_to_temporary_table(
        table_name=table_name,
        schema_name=schema_name,
        data=data,
        column_types=column_types,
    ) as temp_table:
        # Verify the table exists and has the expected data
        with MBDB.get_session() as session:
            # Check that the table exists using SQLAlchemy syntax
            from sqlalchemy import func, select

            result = session.execute(
                select(func.count()).select_from(temp_table)
            ).scalar()
            assert result == 2

            # Check a specific value using SQLAlchemy syntax
            value = session.execute(
                select(temp_table.c.value).where(temp_table.c.id == 1)
            ).scalar()
            assert value == "test1"

    # After context exit, verify the table no longer exists
    with MBDB.get_session() as session:
        with pytest.raises(Exception):  # Should fail as table is dropped # noqa: B017
            session.execute(select(func.count()).select_from(temp_table))
