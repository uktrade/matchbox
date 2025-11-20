from io import BytesIO
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError
from fastapi import UploadFile

from matchbox.common.arrow import table_to_buffer
from matchbox.common.dtos import ResolutionPath, ResolutionType
from matchbox.common.exceptions import (
    MatchboxServerFileError,
)
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import source_factory
from matchbox.server.uploads import (
    InMemoryUploadTracker,
    UploadTracker,
    file_to_s3,
    process_upload,
    s3_to_recordbatch,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


def test_file_to_s3(s3: S3Client) -> None:
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
                "name": "dh",
                "base_generator": "bothify",
                "parameters": (("text", "??######"),),
            },
        ],
    )
    all_companies = source_testkit.data.to_pandas()

    # Create a mock UploadFile
    all_companies["id"] = all_companies["id"].astype(str)
    table = pa.Table.from_pandas(all_companies)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    file_content = sink.getvalue().to_pybytes()

    parquet_file = UploadFile(filename="test.parquet", file=BytesIO(file_content))

    # Call the function
    key = "foo.parquet"
    upload_id = file_to_s3(client=s3, bucket="test-bucket", key=key, file=parquet_file)
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


@pytest.fixture(scope="function")
def tracker_instance(request: pytest.FixtureRequest, tracker: str) -> UploadTracker:
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
    def setup(self, tracker_instance: str) -> None:
        self.tracker: UploadTracker = tracker_instance

    def test_getting_and_setting(self) -> None:
        assert self.tracker.get("upload_id") is None
        self.tracker.set("upload_id", "error_message")
        assert self.tracker.get("upload_id") == "error_message"


@pytest.mark.parametrize(
    "resolution_type",
    [
        pytest.param(ResolutionType.SOURCE, id="source"),
        pytest.param(ResolutionType.MODEL, id="model"),
    ],
)
def test_process_upload_success(s3: S3Client, resolution_type: ResolutionType) -> None:
    """Test that upload process hands data to backend."""
    # Prepare data
    if resolution_type == ResolutionType.SOURCE:
        testkit = source_factory().fake_run()
        resolution = testkit.source.to_resolution()
        data = testkit.data_hashes
    else:
        testkit = model_factory().fake_run()
        resolution = testkit.model.to_resolution()
        data = testkit.probabilities.to_arrow()

    # Setup mock backend and tracker
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.get_resolution = Mock(return_value=resolution)

    # Create bucket
    bucket = "test-bucket"
    test_key = "test-upload-id.parquet"

    s3.create_bucket(
        Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": "eu-west-2"}
    )

    # Add parquet to S3 and verify
    buffer = table_to_buffer(data)
    s3.put_object(Bucket=bucket, Key=test_key, Body=buffer)

    assert s3.head_object(Bucket=bucket, Key=test_key)

    process_upload(
        backend=mock_backend,
        tracker=InMemoryUploadTracker(),
        s3_client=s3,
        upload_id="upload_id",
        bucket=bucket,
        filename=test_key,
        resolution_path=ResolutionPath(
            collection="collection", run=1, name="resolution"
        ),
    )

    # Verify file was deleted
    with pytest.raises(ClientError) as excinfo:
        s3.head_object(Bucket=bucket, Key=test_key)
    assert "404" in str(excinfo.value) or "NoSuchKey" in str(excinfo.value), (
        f"File was not deleted: {str(excinfo.value)}"
    )

    # Ensure data was inserted
    if resolution_type == ResolutionType.SOURCE:
        assert mock_backend.insert_source_data.called
    else:
        assert mock_backend.insert_model_data.called


def test_process_upload_empty_table(s3: S3Client) -> None:
    """Test that files representing empty table can be handled."""
    # Setup mock backend
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3

    # Create bucket
    bucket = "test-bucket"
    test_key = "test-upload-id.parquet"
    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Add empty parquet to S3 and verify
    buffer = table_to_buffer(pa.Table.from_arrays([]))
    s3.put_object(Bucket=bucket, Key=test_key, Body=buffer)

    assert s3.head_object(Bucket=bucket, Key=test_key)

    # Trigger upload processing
    process_upload(
        backend=mock_backend,
        tracker=InMemoryUploadTracker(),
        s3_client=s3,
        upload_id="upload_id",
        bucket=bucket,
        filename=test_key,
        resolution_path=ResolutionPath(
            collection="collection", run=1, name="resolution"
        ),
    )

    # Verify file was deleted
    with pytest.raises(ClientError) as excinfo:
        s3.head_object(Bucket=bucket, Key=test_key)
    assert "404" in str(excinfo.value) or "NoSuchKey" in str(excinfo.value), (
        f"File was not deleted: {str(excinfo.value)}"
    )


def test_process_upload_deletes_file_on_failure(s3: S3Client) -> None:
    """Test that files are deleted from S3 even when processing fails.

    Other behaviours of this task are captured in the API integration tests for adding a
    source or a model.
    """
    # Prepare data
    source_testkit = source_factory().fake_run()

    # Setup mock backend and tracker
    tracker = InMemoryUploadTracker()
    mock_backend = Mock()
    mock_backend.settings.datastore.get_client.return_value = s3
    mock_backend.get_resolution = Mock(
        return_value=source_testkit.source.to_resolution()
    )
    mock_backend.insert_source_data = Mock(
        side_effect=ValueError("Simulated processing failure")
    )

    # Create bucket
    bucket = "test-bucket"
    test_key = "test-upload-id.parquet"

    s3.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
    )

    # Add parquet to S3 and verify
    buffer = table_to_buffer(source_testkit.data_hashes)
    s3.put_object(Bucket=bucket, Key=test_key, Body=buffer)

    assert s3.head_object(Bucket=bucket, Key=test_key)

    # Run the process, expecting it to fail
    with pytest.raises(MatchboxServerFileError) as excinfo:
        process_upload(
            backend=mock_backend,
            tracker=tracker,
            s3_client=s3,
            upload_id="upload_id",
            bucket=bucket,
            filename=test_key,
            resolution_path=ResolutionPath(
                collection="collection", run=1, name="resolution"
            ),
        )

    assert "Simulated processing failure" in str(excinfo.value)

    # Check that the error was added to tracker
    error = tracker.get("upload_id")
    assert "Simulated processing failure" in error

    # Verify file was deleted despite the failure
    with pytest.raises(ClientError) as excinfo:
        s3.head_object(Bucket=bucket, Key=test_key)
    assert "404" in str(excinfo.value) or "NoSuchKey" in str(excinfo.value), (
        f"File was not deleted: {str(excinfo.value)}"
    )

    # Ensure resolution is marked as ready again
    assert mock_backend.unlock_resolution_data.called
