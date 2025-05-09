import asyncio
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import UploadFile

from matchbox.common.dtos import BackendUploadType
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.factories.sources import source_factory
from matchbox.server.api.arrow import s3_to_recordbatch, table_to_s3
from matchbox.server.api.cache import MetadataStore, heartbeat

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


@pytest.mark.asyncio
async def test_file_to_s3(s3: S3Client):
    """Test that a file can be uploaded to S3."""
    # Create a mock bucket
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )
    source_testkit = source_factory(
        features=[
            {"name": "company_name", "base_generator": "company"},
            {"name": "address", "base_generator": "address"},
            {
                "name": "crn",
                "base_generator": "bothify",
                "parameters": (("text", "???-###-???-###"),),
            },
            {
                "name": "duns",
                "base_generator": "bothify",
                "parameters": (("text", "??######"),),
            },
        ],
    )
    all_companies = source_testkit.query.to_pandas()

    # Test 1: Upload a parquet file
    # Create a mock UploadFile
    all_companies["id"] = all_companies["id"].astype(str)
    table = pa.Table.from_pandas(all_companies)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    file_content = sink.getvalue().to_pybytes()

    parquet_file = UploadFile(filename="test.parquet", file=BytesIO(file_content))

    # Call the function
    key = "foo.parquet"
    upload_id = await table_to_s3(
        client=s3,
        bucket="test-bucket",
        key=key,
        file=parquet_file,
        expected_schema=table.schema,
    )
    # Validate response
    assert key == upload_id

    response_table = pa.Table.from_batches(
        [
            batch
            for batch in s3_to_recordbatch(
                client=s3, bucket="test-bucket", key=upload_id
            )
        ]
    )

    assert response_table.equals(table)

    # Test 2: Upload a non-parquet file
    text_file = UploadFile(filename="test.txt", file=BytesIO(b"test"))

    with pytest.raises(MatchboxServerFileError):
        await table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=text_file,
            expected_schema=table.schema,
        )

    # Test 3: Upload a parquet file with a different schema
    corrupted_schema = table.schema.remove(0)
    with pytest.raises(MatchboxServerFileError):
        await table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=parquet_file,
            expected_schema=corrupted_schema,
        )


def test_basic_cache_and_retrieve():
    """Test basic caching and retrieval functionality."""
    store = MetadataStore()
    source = source_factory().config

    # Cache the source
    cache_id = store.cache_source(source)
    assert isinstance(cache_id, str)

    # Retrieve and verify
    entry = store.get(cache_id)
    assert entry is not None
    assert entry.metadata == source
    assert entry.upload_type.schema == BackendUploadType.INDEX.schema
    assert isinstance(entry.update_timestamp, datetime)

    # Verify initial status
    assert entry.status.status == "awaiting_upload"
    assert entry.status.entity == BackendUploadType.INDEX
    assert entry.status.id == cache_id


@patch("matchbox.server.api.cache.datetime")
def test_expiration(mock_datetime: Mock):
    """Test that entries expire correctly after period of inactivity."""
    store = MetadataStore(expiry_minutes=30)
    source = source_factory().config

    # Cache the source
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    cache_id = store.cache_source(source)

    # Should still be valid after 29 minutes
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 29)
    assert store.get(cache_id) is not None

    # Status update should refresh timestamp
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 30)
    store.update_status(cache_id, "processing")

    # Should still be valid 29 minutes after status update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 59)
    assert store.get(cache_id) is not None

    # Should expire 31 minutes after status update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 13, 30)
    assert store.get(cache_id) is None


@pytest.mark.asyncio
@patch("matchbox.server.api.cache.datetime")
async def test_cleanup(mock_datetime: Mock):
    """Test that cleanup removes expired entries."""
    store = MetadataStore(expiry_minutes=30)
    source = source_factory().config

    # Create two entries
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    id1 = store.cache_source(source)
    id2 = store.cache_source(source)

    # Update status of one entry
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 15)
    store.update_status(id2, "processing")

    # Move time forward past expiration for first entry but not second
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 31)

    # Adding a new entry should trigger cleanup
    id3 = store.cache_source(source)

    # First should be gone, second should exist due to status update
    assert store.get(id1) is None
    assert store.get(id2) is not None
    assert store.get(id3) is not None


def test_remove():
    """Test manual removal of cache entries."""
    store = MetadataStore()
    source = source_factory().config

    # Cache and remove
    cache_id = store.cache_source(source)
    assert store.remove(cache_id) is True
    assert store.get(cache_id) is None

    # Try removing non-existent entry
    assert store.remove("nonexistent") is False


@pytest.mark.asyncio
async def test_status_management():
    """Test status update functionality."""
    store = MetadataStore()
    source = source_factory().config

    # Create entry and verify initial status
    cache_id = store.cache_source(source)
    entry = store.get(cache_id)
    assert entry.status.status == "awaiting_upload"

    # Update status
    assert store.update_status(cache_id, "processing") is True
    entry = store.get(cache_id)
    assert entry.status.status == "processing"

    # Update with details
    assert store.update_status(cache_id, "failed", "Error details") is True
    entry = store.get(cache_id)
    assert entry.status.status == "failed"
    assert entry.status.details == "Error details"

    # Try updating non-existent entry
    with pytest.raises(KeyError):
        store.update_status("nonexistent", "processing")


@pytest.mark.asyncio
@patch("matchbox.server.api.cache.datetime")
async def test_timestamp_updates(mock_datetime: Mock):
    """Test that timestamps update correctly on different operations."""
    store = MetadataStore()
    source = source_factory().config

    # Initial creation
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    cache_id = store.cache_source(source)
    entry = store.get(cache_id)
    assert entry.update_timestamp == datetime(2024, 1, 1, 12, 0)

    # Status update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 15)
    store.update_status(cache_id, "processing")
    entry = store.get(cache_id)
    assert entry.update_timestamp == datetime(2024, 1, 1, 12, 15)

    # Get operation
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 30)
    entry = store.get(cache_id)
    assert entry.update_timestamp == datetime(2024, 1, 1, 12, 30)


@pytest.mark.asyncio
async def test_heartbeat_updates_status():
    """Test that heartbeat updates status periodically."""
    store = MetadataStore()
    source = source_factory().config

    # Create initial entry
    upload_id = store.cache_source(source)

    async with heartbeat(store, upload_id, interval_seconds=0.1):
        # Wait long enough for at least one heartbeat
        await asyncio.sleep(0.15)

    # Verify the status was updated with heartbeat details
    entry = store.get(upload_id)
    assert entry.status.status == "processing"
    assert "Still processing... Last heartbeat:" in entry.status.details


@pytest.mark.asyncio
@patch("matchbox.server.api.cache.datetime")
async def test_heartbeat_timestamp_updates(mock_datetime: Mock):
    """Test that heartbeat updates timestamps correctly."""
    store = MetadataStore()
    source = source_factory().config

    # Create initial entry
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    upload_id = store.cache_source(source)

    # Start heartbeat and advance time
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 5)
    async with heartbeat(store, upload_id, interval_seconds=0.1):
        await asyncio.sleep(0.15)

        entry = store.get(upload_id)
        assert entry.update_timestamp == datetime(2024, 1, 1, 12, 5)
        assert "Last heartbeat:" in entry.status.details


@pytest.mark.asyncio
@patch("matchbox.server.api.cache.datetime")
async def test_heartbeat_with_expiry(mock_datetime: Mock):
    """Test heartbeat behavior with cache expiry."""
    store = MetadataStore(expiry_minutes=30)
    source = source_factory().config

    # Create entry
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0)
    upload_id = store.cache_source(source)

    # Start heartbeat and let it update
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 25)
    async with heartbeat(store, upload_id, interval_seconds=0.1):
        await asyncio.sleep(0.15)

        # Entry should still exist and have updated timestamp
        entry = store.get(upload_id)
        assert entry is not None
        assert entry.update_timestamp == datetime(2024, 1, 1, 12, 25)

    # Move past expiry time
    mock_datetime.now.return_value = datetime(2024, 1, 1, 13, 0)
    assert store.get(upload_id) is None


@pytest.mark.asyncio
async def test_heartbeat_errors_on_removed_entry():
    """Test heartbeat behavior when entry is removed during processing."""
    store = MetadataStore()
    source = source_factory().config

    # Create entry
    upload_id = store.cache_source(source)

    with pytest.raises(KeyError):
        # Remove entry while heartbeat is running
        async with heartbeat(store, upload_id, interval_seconds=0.1):
            store.remove(upload_id)
            await asyncio.sleep(0.15)  # Wait for next heartbeat attempt


@pytest.mark.asyncio
async def test_multiple_heartbeats():
    """Test multiple concurrent heartbeats on different entries."""
    store = MetadataStore()
    source = source_factory().config

    # Create two entries
    id1 = store.cache_source(source)
    id2 = store.cache_source(source)

    async def run_heartbeat(upload_id):
        async with heartbeat(store, upload_id, interval_seconds=0.1):
            await asyncio.sleep(0.15)

    # Run heartbeats concurrently
    await asyncio.gather(run_heartbeat(id1), run_heartbeat(id2))

    # Verify both entries were updated
    entry1 = store.get(id1)
    entry2 = store.get(id2)
    assert entry1.status.status == "processing"
    assert entry2.status.status == "processing"
    assert "Last heartbeat:" in entry1.status.details
    assert "Last heartbeat:" in entry2.status.details
