"""Worker logic to process user uploads."""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Generator

import pyarrow as pa
import redis
from celery import Celery
from fastapi import (
    UploadFile,
)
from pyarrow import parquet as pq
from pydantic import BaseModel

from matchbox.common.dtos import (
    BackendUploadType,
    ModelConfig,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.logging import logger
from matchbox.common.sources import SourceConfig
from matchbox.server.base import (
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

# -- Upload trackers --


class UploadEntry(BaseModel):
    """Entry in upload tracker, combining private metadata and public upload status."""

    status: UploadStatus
    metadata: SourceConfig | ModelConfig


class UploadTracker(ABC):
    """Abstract class for upload tracker."""

    @staticmethod
    def _create_entry(
        metadata: SourceConfig | ModelConfig, upload_type: BackendUploadType
    ) -> UploadEntry:
        """Create initial UploadEntry object."""
        upload_id = str(uuid.uuid4())

        return UploadEntry(
            metadata=metadata,
            status=UploadStatus(
                id=upload_id,
                stage=UploadStage.AWAITING_UPLOAD,
                update_timestamp=datetime.now(),
                entity=upload_type,
            ),
        )

    def _get_updated_entry(
        self, upload_id: str, stage: str, details: str | None
    ) -> UploadEntry:
        """Create new UploadEntry object as update on previous entry."""
        entry = self.get(upload_id)
        if not entry:
            raise KeyError(f"Entry {upload_id} not found.")

        status = entry.status.model_copy(
            update={"stage": stage, "update_timestamp": datetime.now()}
        )
        if details:
            status.details = details

        return UploadEntry(status=status, metadata=entry.metadata)

    def add_source(self, metadata: SourceConfig) -> str:
        """Register source metadata and return ID."""
        entry = self._create_entry(metadata, BackendUploadType.INDEX)
        self._register_entry(entry)

        return entry.status.id

    def add_model(self, metadata: ModelConfig) -> str:
        """Register model results metadata and return ID."""
        entry = self._create_entry(metadata, BackendUploadType.RESULTS)
        self._register_entry(entry)

        return entry.status.id

    @abstractmethod
    def _register_entry(self, UploadEntry) -> str:
        """Register UploadEntry object to tracker and return its ID."""
        ...

    @abstractmethod
    def get(self, upload_id: str) -> UploadEntry | None:
        """Retrieve metadata by ID if not expired."""
        ...

    @abstractmethod
    def update(self, upload_id: str, stage: str, details: str | None = None) -> None:
        """Update the stage and details for an upload.

        Raises:
            KeyError: If entry not found.
        """
        ...


class InMemoryUploadTracker(UploadTracker):
    """In-memory upload tracker, only usable with single server instance."""

    def __init__(self):
        """Initialise tracker data structure."""
        self._tracker = {}

    def _register_entry(self, entry: UploadEntry) -> None:
        self._tracker[entry.status.id] = entry

    def get(self, upload_id: str) -> UploadEntry | None:  # noqa: D102
        return self._tracker.get(upload_id)

    def update(  # noqa: D102
        self, upload_id: str, stage: str, details: str | None = None
    ) -> None:
        self._tracker[upload_id] = self._get_updated_entry(
            upload_id=upload_id, stage=stage, details=details
        )


class RedisUploadTracker(UploadTracker):
    """Upload tracker backed by Redis."""

    def __init__(self, redis_url: str, expiry_minutes: int, key_space: str = "upload"):
        """Connect Redis and initialise tracker object."""
        self.expiry_minutes = expiry_minutes
        self.redis = redis.Redis.from_url(redis_url)
        self.key_prefix = f"{key_space}:"

    def _to_redis(self, key: str, value: str):
        expiry_seconds = self.expiry_minutes * 60
        self.redis.setex(f"{self.key_prefix}{key}", expiry_seconds, value)

    def _register_entry(self, entry: UploadEntry) -> str:  # noqa: D102
        self._to_redis(entry.status.id, entry.model_dump_json())

        return entry.status.id

    def get(self, upload_id: str) -> UploadEntry | None:  # noqa: D102
        data = self.redis.get(f"{self.key_prefix}{upload_id}")
        if not data:
            return None

        entry = UploadEntry.model_validate_json(data)

        return entry

    def update(  # noqa: D102
        self, upload_id: str, stage: str, details: str | None = None
    ) -> None:
        entry = self._get_updated_entry(
            upload_id=upload_id, stage=stage, details=details
        )

        self._to_redis(upload_id, entry.model_dump_json())


_IN_MEMORY_TRACKER = InMemoryUploadTracker()


def settings_to_upload_tracker(settings: MatchboxServerSettings) -> UploadTracker:
    """Initialise an upload tracker from server settings."""
    match settings.task_runner:
        case "api":
            return _IN_MEMORY_TRACKER
        case "celery":
            return RedisUploadTracker(
                redis_url=settings.redis_uri,
                expiry_minutes=settings.uploads_expiry_minutes,
            )
        case _:
            raise RuntimeError("Unsupported task runner.")


# -- S3 functions --


def table_to_s3(
    client: S3Client,
    bucket: str,
    key: str,
    file: UploadFile,
    expected_schema: pa.Schema,
) -> str:
    """Upload a PyArrow Table to S3 and return the key.

    Args:
        client: The S3 client to use.
        bucket: The S3 bucket to upload to.
        key: The key to upload to.
        file: The file to upload.
        expected_schema: The schema that the file should match.

    Raises:
        MatchboxServerFileError: If the file is not a valid Parquet file or the schema
            does not match the expected schema.

    Returns:
        The key of the uploaded file.
    """
    try:
        table = pq.read_table(file.file)

        if not table.schema.equals(expected_schema):
            raise MatchboxServerFileError(
                message=(
                    "Schema mismatch. "
                    f"Expected:\n{expected_schema}\nGot:\n{table.schema}"
                )
            )

        file.file.seek(0)

        client.put_object(Bucket=bucket, Key=key, Body=file.file)

    except Exception as e:
        if isinstance(e, MatchboxServerFileError):
            raise
        raise MatchboxServerFileError(message=f"Invalid Parquet file: {str(e)}") from e

    return key


def s3_to_recordbatch(
    client: S3Client, bucket: str, key: str, batch_size: int = 1000
) -> Generator[pa.RecordBatch, None, None]:
    """Download a PyArrow Table from S3 and stream it as RecordBatches."""
    response = client.get_object(Bucket=bucket, Key=key)
    buffer = pa.BufferReader(response["Body"].read())

    parquet_file = pq.ParquetFile(buffer)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch


# -- Upload tasks --


CELERY_SETTINGS = get_backend_settings(MatchboxServerSettings().backend_type)()
CELERY_BACKEND: MatchboxDBAdapter | None = None
CELERY_TRACKER: UploadTracker | None = None

celery = Celery("matchbox", broker=CELERY_SETTINGS.redis_uri)
celery.conf.update(
    # Only acknowledge task (remove it from queue) after task completion
    task_acks_late=True,
    # Reduce pre-fetching (workers reserving tasks while they're still busy)
    # as it's not ideal for long-running tasks
    prefetch_multiplier=1,
)


def initialise_celery_worker():
    """Initialise backend and tracker for celery worker."""
    global CELERY_SETTINGS
    global CELERY_BACKEND
    global CELERY_TRACKER

    if not CELERY_BACKEND:
        CELERY_BACKEND = settings_to_backend(CELERY_SETTINGS)
    if not CELERY_TRACKER:
        CELERY_TRACKER = settings_to_upload_tracker(CELERY_SETTINGS)


def process_upload(
    backend: MatchboxDBAdapter,
    tracker: UploadTracker,
    s3_client: S3Client,
    upload_type: str,
    resolution_name: str,
    upload_id: str,
    bucket: str,
    filename: str,
) -> None:
    """Generic task to process uploaded file, usable by FastAPI BackgroundTasks."""
    tracker.update(upload_id, UploadStage.PROCESSING)
    upload = tracker.get(upload_id)

    try:
        data = pa.Table.from_batches(
            [
                batch
                for batch in s3_to_recordbatch(
                    client=s3_client, bucket=bucket, key=filename
                )
            ]
        )

        if upload.status.entity == BackendUploadType.INDEX:
            backend.index(source_config=upload.metadata, data_hashes=data)
        elif upload.status.entity == BackendUploadType.RESULTS:
            backend.set_model_results(name=upload.metadata.name, results=data)
        else:
            raise ValueError(f"Unknown upload type: {upload.status.entity}")

        tracker.update(upload_id, UploadStage.COMPLETE)

    except Exception as e:
        error_context = {
            "upload_id": upload_id,
            "upload_type": upload_type,
            "resolution_name": resolution_name,
            "bucket": bucket,
            "key": filename,
        }
        logger.error(
            f"Upload processing failed with context: {error_context}", exc_info=True
        )
        details = (
            f"Error: {e}. Context: "
            f"Upload type: {getattr(upload.status, 'entity', 'unknown')}, "
            f"SourceConfig: {getattr(upload, 'metadata', 'unknown')}"
        )
        tracker.update(
            upload_id,
            UploadStage.FAILED,
            details=details,
        )
        raise MatchboxServerFileError(message=details) from e
    finally:
        try:
            s3_client.delete_object(Bucket=bucket, Key=filename)
        except Exception as delete_error:
            logger.error(
                f"Failed to delete S3 file {bucket}/{filename}: {delete_error}"
            )


@celery.task(ignore_result=True)
def process_upload_celery(
    upload_type: str,
    resolution_name: str,
    upload_id: str,
    bucket: str,
    filename: str,
) -> None:
    """Celery task to process uploaded file, with only serialisable arguments."""
    initialise_celery_worker()

    partial(
        process_upload,
        backend=CELERY_BACKEND,
        tracker=CELERY_TRACKER,
        s3_client=CELERY_BACKEND.settings.datastore.get_client(),
    )(
        upload_type=upload_type,
        resolution_name=resolution_name,
        upload_id=upload_id,
        bucket=bucket,
        filename=filename,
    )
