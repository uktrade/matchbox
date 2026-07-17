import pyarrow as pa
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import BIGINT, TEXT

from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.utils.db import ingest_to_temporary_table


@pytest.mark.docker
def test_adbc_pool_state(matchbox_postgres: MatchboxPostgres) -> None:
    """Ensure pooled ADBC connections do not retain open transactions.

    ADBC starts a new transaction after commit or rollback when autocommit is
    disabled. PostgreSQL can terminate a connection returned in that state when
    it reaches the idle-in-transaction timeout, so pooled connections must be idle.
    """
    with (
        MBDB.get_adbc_connection() as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute("SELECT pg_backend_pid()")
        row = cursor.fetchone()
        assert row is not None
        backend_pid = row[0]

    with MBDB.get_session() as session:
        state = session.execute(
            text("SELECT state FROM pg_stat_activity WHERE pid = :pid"),
            {"pid": backend_pid},
        ).scalar_one()

    assert state == "idle"


@pytest.mark.docker
def test_ingest_to_temporary_table(
    matchbox_postgres: MatchboxPostgres,  # will drop dummy table
) -> None:
    """Test temporary table creation, data ingestion, and automatic cleanup."""
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
        "id": BIGINT(),
        "value": TEXT(),
    }

    # Use the context manager to create and populate a temporary table
    with (
        ingest_to_temporary_table(
            table_name=table_name,
            schema_name=schema_name,
            data=data,
            column_types=column_types,
        ) as temp_table,
        MBDB.get_session() as session,
    ):
        # Verify the table exists and has the expected data
        # Check that the table exists using SQLAlchemy syntax
        result = session.execute(select(func.count()).select_from(temp_table)).scalar()
        assert result == 2

        # Check a specific value using SQLAlchemy syntax
        value = session.execute(
            select(temp_table.c.value).where(temp_table.c.id == 1)
        ).scalar()
        assert value == "test1"

    # After context exit, verify the table no longer exists
    with (
        MBDB.get_session() as session,
        pytest.raises(Exception),  # Should fail as table is dropped # noqa: B017
    ):
        session.execute(select(func.count()).select_from(temp_table))
