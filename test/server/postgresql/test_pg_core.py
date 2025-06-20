import pyarrow as pa
import pytest
from sqlalchemy import BIGINT, TEXT, Column, MetaData, UniqueConstraint
from sqlalchemy.orm import declarative_base

from matchbox.common.exceptions import MatchboxDatabaseWriteError
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.mixin import CountMixin
from matchbox.server.postgresql.orm import PKSpace
from matchbox.server.postgresql.utils.db import ingest_to_temporary_table, large_ingest


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
def test_large_ingest_simple(
    matchbox_postgres: MatchboxPostgres,  # will drop dummy table
):
    """Test append-only mode of large ingest."""
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

    # Ingest data with manual keys and chunking
    large_ingest(
        data=pa.Table.from_pylist(
            [{"key": 0, "foo": "val1"}],
        ),
        table_class=DummyTable,
        max_chunksize=100,
    )

    # Ingest data without keys and no chunking
    large_ingest(data=pa.Table.from_pylist([{"foo": "val2"}]), table_class=DummyTable)

    # Both rows were fine
    assert DummyTable.count() == 2
    with MBDB.get_session() as session:
        second_id = (
            session.query(DummyTable.key).filter(DummyTable.foo == "val2").scalar()
        )
    assert 0 < second_id

    # By default, upserting not allowed
    with pytest.raises(MatchboxDatabaseWriteError):
        large_ingest(
            data=pa.Table.from_pylist([{"key": 0, "foo": "val3"}]),
            table_class=DummyTable,
        )

    # Failed ingestion has no effect
    assert DummyTable.count() == 2
    assert len(metadata.tables) == original_tables

    # Columns not available in the target table are rejected
    with pytest.raises(ValueError, match="does not have columns"):
        large_ingest(
            data=pa.Table.from_pylist([{"key": 10, "bar": "val3"}]),
            table_class=DummyTable,
        )


@pytest.mark.docker
def test_large_ingest_upsert_custom_update(
    matchbox_postgres_dropped: MatchboxPostgres,  # will drop dummy table
):
    """Test large ingest with upsertion and custom columns to update."""
    engine = MBDB.get_engine()
    metadata = MetaData(schema=MBDB.MatchboxBase.metadata.schema)

    # Initialise DummyTable to which we'll ingest
    class DummyTable(CountMixin, declarative_base(metadata=metadata)):
        __tablename__ = "dummytable"
        key = Column(BIGINT, primary_key=True)
        foo = Column(TEXT, nullable=False)
        bar = Column(TEXT, nullable=False)

        __table_args__ = (UniqueConstraint("foo", name="unique_foo"),)

    metadata.create_all(engine, tables=[DummyTable.__table__])
    metadata.reflect(engine)
    original_tables = len(metadata.tables)

    # Initialise with one original row
    with MBDB.get_session() as session:
        row1 = DummyTable(key=1, foo="original foo", bar="original bar")
        session.add(row1)
        session.commit()

    # Some choices of parameters are not allowed
    with pytest.raises(ValueError, match="Cannot update a custom upsert key"):
        large_ingest(
            data=pa.Table.from_pylist([{"key": 1, "foo": "new foo", "bar": "new bar"}]),
            table_class=DummyTable,
            update_columns=["foo"],
            upsert_keys=["foo"],
        )

    with pytest.raises(ValueError, match="different custom upsert key"):
        large_ingest(
            data=pa.Table.from_pylist([{"key": 1, "foo": "new foo", "bar": "new bar"}]),
            table_class=DummyTable,
            update_columns=["key"],
        )

    # Ingest updated data
    large_ingest(
        data=pa.Table.from_pylist([{"key": 1, "foo": "new foo", "bar": "new bar"}]),
        table_class=DummyTable,
        update_columns=["foo"],
    )

    # Number of rows unchanged
    assert DummyTable.count() == 1

    # Only foo has changed
    with MBDB.get_session() as session:
        new_foo = session.query(DummyTable.foo).filter(DummyTable.key == 1).scalar()
        new_bar = session.query(DummyTable.bar).filter(DummyTable.key == 1).scalar()
    assert "new" in new_foo
    assert "original" in new_bar

    # No lingering temp tables
    metadata.clear()  # clear all Table objects from this MetaData, doesn't touch DB
    metadata.reflect(engine)
    assert len(metadata.tables) == original_tables

    # Cannot update column when constraints violated
    with pytest.raises(MatchboxDatabaseWriteError):
        large_ingest(
            pa.Table.from_pylist([{"key": 2, "foo": "new foo", "bar": "new bar"}]),
            DummyTable,
            update_columns=["foo"],
        )

    # Failed ingestion has no effect
    assert DummyTable.count() == 1
    metadata.clear()
    metadata.reflect(engine)
    assert len(metadata.tables) == original_tables

    # Constraints are not violated when upserting
    large_ingest(
        pa.Table.from_pylist([{"key": 1, "foo": "new foo", "bar": "new bar"}]),
        DummyTable,
        update_columns=["foo"],
    )

    # Nothing changed still
    assert DummyTable.count() == 1


@pytest.mark.docker
def test_large_ingest_upsert_custom_key(
    matchbox_postgres_dropped: MatchboxPostgres,  # will drop dummy table
):
    """Test large ingest with upsertion on custom keys."""
    engine = MBDB.get_engine()
    metadata = MetaData(schema=MBDB.MatchboxBase.metadata.schema)

    # Initialise DummyTable to which we'll ingest
    class DummyTable(CountMixin, declarative_base(metadata=metadata)):
        __tablename__ = "dummytable"
        key = Column(BIGINT, primary_key=True)
        other_key = Column(TEXT, nullable=False)
        foo = Column(TEXT, nullable=False)

        __table_args__ = (UniqueConstraint("other_key", name="unique_other_key"),)

    metadata.create_all(engine, tables=[DummyTable.__table__])

    # Initialise with one original row
    with MBDB.get_session() as session:
        metadata.reflect(engine)
        original_tables = len(metadata.tables)

        row1 = DummyTable(key=1, other_key="a", foo="original foo")
        session.add(row1)
        session.commit()

    # Ingest updated data
    large_ingest(
        data=pa.Table.from_pylist([{"key": 2, "other_key": "a", "foo": "new foo"}]),
        table_class=DummyTable,
        upsert_keys=["other_key"],
    )

    # Number of rows unchanged
    assert DummyTable.count() == 1

    with MBDB.get_session() as session:
        new_foo = (
            session.query(DummyTable.foo).filter(DummyTable.other_key == "a").scalar()
        )
        new_keys = (
            session.query(DummyTable.key).filter(DummyTable.other_key == "a").scalar()
        )

    # We can update standard columns and primary keys
    assert "new" in new_foo
    assert new_keys == 2

    # No lingering temp tables
    metadata.clear()  # clear all Table objects from this MetaData, doesn't touch DB
    metadata.reflect(engine)
    assert len(metadata.tables) == original_tables


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
