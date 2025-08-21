from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError
from fastapi import UploadFile

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import BackendUploadType, ModelConfig, ModelType, UploadStage
from matchbox.common.exceptions import (
    MatchboxServerFileError,
)
from matchbox.common.factories.sources import source_factory
from matchbox.server.uploads import (
    InMemoryUploadTracker,
    UploadTracker,
    process_upload,
    s3_to_recordbatch,
    table_to_s3,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_file_to_s3(s3: S3Client):
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
    upload_id = table_to_s3(
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
        table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=text_file,
            expected_schema=table.schema,
        )

    # Test 3: Upload a parquet file with a different schema
    corrupted_schema = table.schema.remove(0)
    with pytest.raises(MatchboxServerFileError):
        table_to_s3(
            client=s3,
            bucket="test-bucket",
            key=key,
            file=parquet_file,
            expected_schema=corrupted_schema,
        )


@pytest.fixture(scope="function")
def tracker_instance(request: pytest.FixtureRequest, tracker: str):
    """Create a fresh tracker instance for each test."""
    tracker_obj = request.getfixturevalue(tracker)
    return tracker_obj


@pytest.mark.parametrize(
    "tracker",
    [
        pytest.param("upload_tracker_in_memory", id="in_memory"),
        pytest.param("upload_tracker_redis", id="redis"),
    ],
)
@pytest.mark.docker
class TestUploadTracker:
    @pytest.fixture(autouse=True)
    def setup(self, tracker_instance: str):
        self.tracker: UploadTracker = tracker_instance

    def test_basic_upload_tracking(self):
        """Test adding upload to tracker and retrieving."""
        source = source_factory().source_config
        model = ModelConfig(
            name="name",
            description="description",
            type=ModelType.DEDUPER,
            left_resolution="resolution",
        )

        # Add the source and the model
        source_upload_id = self.tracker.add_source(source)
        assert isinstance(source_upload_id, str)

        model_upload_id = self.tracker.add_model(model)
        assert isinstance(source_upload_id, str)

        # Retrieve and verify
        source_entry = self.tracker.get(source_upload_id)
        assert source_entry is not None
        assert source_entry.metadata == source
        assert source_entry.status.stage == UploadStage.AWAITING_UPLOAD
        assert source_entry.status.entity == BackendUploadType.INDEX
        assert source_entry.status.id == source_upload_id
        assert isinstance(source_entry.status.update_timestamp, datetime)

        model_entry = self.tracker.get(model_upload_id)
        assert model_entry is not None
        assert model_entry.metadata == model
        assert model_entry.status.stage == UploadStage.AWAITING_UPLOAD
        assert model_entry.status.entity == BackendUploadType.RESULTS
        assert model_entry.status.id == model_upload_id
        assert isinstance(model_entry.status.update_timestamp, datetime)

    def test_status_management(self):
        """Test status update functionality."""
        source = source_factory().source_config

        # Create entry and verify initial status
        upload_id = self.tracker.add_source(source)
        entry = self.tracker.get(upload_id)
        assert entry.status.stage == UploadStage.AWAITING_UPLOAD

        # Update status
        self.tracker.update(upload_id, UploadStage.PROCESSING)
        entry = self.tracker.get(upload_id)
        assert entry.status.stage == UploadStage.PROCESSING

        # Update with details
        self.tracker.update(upload_id, UploadStage.FAILED, "Error details")
        entry = self.tracker.get(upload_id)
        assert entry.status.stage == UploadStage.FAILED
        assert entry.status.details == "Error details"

        # Try updating non-existent entry
        with pytest.raises(KeyError):
            self.tracker.update("nonexistent", UploadStage.PROCESSING)

    @patch("matchbox.server.uploads.datetime")
    def test_timestamp_updates(self, mock_datetime: Mock):
        """Test that timestamps update correctly on different operations."""
        source = source_factory().source_config

        creation_timestamp = datetime(2024, 1, 1, 12, 0)
        get_timestamp = datetime(2024, 1, 1, 12, 15)
        update_timestamp = datetime(2024, 1, 1, 12, 30)

        # Initial creation
        mock_datetime.now.return_value = creation_timestamp
        upload_id = self.tracker.add_source(source)
        entry = self.tracker.get(upload_id)
        assert entry.status.update_timestamp == datetime(2024, 1, 1, 12, 0)

        # Get operation does not update timestamp
        mock_datetime.now.return_value = get_timestamp
        entry = self.tracker.get(upload_id)
        assert entry.status.update_timestamp == creation_timestamp

        # Status update updates timestamp
        mock_datetime.now.return_value = update_timestamp
        self.tracker.update(upload_id, UploadStage.PROCESSING)
        entry = self.tracker.get(upload_id)
        assert entry.status.update_timestamp == update_timestamp


def test_process_upload_deletes_file_on_failure(s3: S3Client):
    """Test that files are deleted from S3 even when processing fails.

    Other behaviours of this task are captured in the API integration tests for adding a
    source or a model.
    """
    # Setup
    tracker = InMemoryUploadTracker()
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.index = Mock(side_effect=ValueError("Simulated processing failure"))

    bucket = "test-bucket"
    test_key = "test-upload-id.parquet"

    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Add parquet to S3 and verify
    source_testkit = source_factory()
    buffer = table_to_buffer(source_testkit.data_hashes)
    s3.put_object(Bucket=bucket, Key=test_key, Body=buffer)

    assert s3.head_object(Bucket=bucket, Key=test_key)

    # Setup metadata store with test data
    upload_id = tracker.add_source(source_testkit.source_config)
    tracker.update(upload_id, UploadStage.AWAITING_UPLOAD)

    # Run the process, expecting it to fail
    with pytest.raises(MatchboxServerFileError) as excinfo:
        process_upload(
            backend=mock_backend,
            tracker=tracker,
            s3_client=s3,
            upload_id=upload_id,
            bucket=bucket,
            filename=test_key,
            upload_type="type",
            resolution_name="name",
        )

    assert "Simulated processing failure" in str(excinfo.value)

    # Check that the status was updated to failed
    status = tracker.get(upload_id).status
    assert status.stage == UploadStage.FAILED, (
        f"Expected status 'failed', got '{status.stage}'"
    )

    # Verify file was deleted despite the failure
    with pytest.raises(ClientError) as excinfo:
        s3.head_object(Bucket=bucket, Key=test_key)
    assert "404" in str(excinfo.value) or "NoSuchKey" in str(excinfo.value), (
        f"File was not deleted: {str(excinfo.value)}"
    )
