"""Worker logic to process user uploads."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import redis
from celery import Celery, Task
from fastapi import UploadFile
from pyarrow import parquet as pq
from pydantic import BaseModel

from matchbox.common.dtos import (
    BackendUploadType,
    ResolutionPath,
    UploadStage,
    UploadStatus,
)
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.logging import logger
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

from celery.exceptions import MaxRetriesExceededError
from celery.utils.log import get_task_logger

celery_logger = get_task_logger(__name__)

# -- Upload trackers --


class UploadEntry(BaseModel):
    """Entry in upload tracker, combining private metadata and public upload status."""

    path: ResolutionPath
    status: UploadStatus


class UploadTracker(ABC):
    """Abstract class for upload tracker."""

    @staticmethod
    def _create_entry(
        path: ResolutionPath, upload_type: BackendUploadType
    ) -> UploadEntry:
        """Create initial UploadEntry object."""
        return UploadEntry(
            path=path,
            status=UploadStatus(
                full_path=str(path),
                stage=UploadStage.AWAITING_UPLOAD,
                update_timestamp=datetime.now(),
                entity=upload_type,
            ),
        )

    def _get_updated_entry(
        self, full_path: str, stage: str, details: str | None
    ) -> UploadEntry:
        """Create new UploadEntry object as update on previous entry."""
        entry = self.get(full_path)
        if not entry:
            raise KeyError(f"Entry {(full_path)} not found.")

        status = entry.status.model_copy(
            update={"stage": stage, "update_timestamp": datetime.now()}
        )
        if details:
            status.details = details

        return UploadEntry(path=entry.path, status=status)

    def add_source(self, path: ResolutionPath) -> str:
        """Register source resolution and return path."""
        entry = self._create_entry(path, BackendUploadType.INDEX)
        self._register_entry(entry)

        return entry.status.full_path

    def add_model(self, path: ResolutionPath) -> str:
        """Register model resolution and return path."""
        entry = self._create_entry(path, BackendUploadType.RESULTS)
        self._register_entry(entry)

        return entry.status.full_path

    @abstractmethod
    def _register_entry(self, UploadEntry) -> str:
        """Register UploadEntry object to tracker and return its path."""
        ...

    @abstractmethod
    def get(self, full_path: str) -> UploadEntry | None:
        """Retrieve entry by ID if not expired."""
        ...

    @abstractmethod
    def update(self, full_path: str, stage: str, details: str | None = None) -> None:
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
        self._tracker[str(entry.path)] = entry

    def get(self, full_path) -> UploadEntry | None:  # noqa: D102
        return self._tracker.get(full_path)

    def update(  # noqa: D102
        self, full_path: str, stage: str, details: str | None = None
    ) -> None:
        self._tracker[full_path] = self._get_updated_entry(
            full_path=full_path, stage=stage, details=details
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
        self._to_redis(str(entry.path), entry.model_dump_json())

        return str(entry.path)

    def get(self, full_path) -> UploadEntry | None:  # noqa: D102
        data = self.redis.get(f"{self.key_prefix}{full_path}")
        if not data:
            return None

        entry = UploadEntry.model_validate_json(data)

        return entry

    def update(  # noqa: D102
        self, full_path: str, stage=str, details: str | None = None
    ) -> None:
        entry = self._get_updated_entry(
            full_path=full_path, stage=stage, details=details
        )

        self._to_redis(full_path, entry.model_dump_json())


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

    yield from parquet_file.iter_batches(batch_size=batch_size)


# -- Upload tasks --


CELERY_SETTINGS = get_backend_settings(MatchboxServerSettings().backend_type)()
CELERY_BACKEND: MatchboxDBAdapter | None = None
CELERY_TRACKER: UploadTracker | None = None

celery = Celery("matchbox", broker=CELERY_SETTINGS.redis_uri)
celery.conf.update(
    # Hard time limit for tasks (in seconds)
    task_time_limit=CELERY_SETTINGS.uploads_expiry_minutes * 60,
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
    full_path: str,
    bucket: str,
    filename: str,
) -> None:
    """Generic task to process uploaded file, usable by FastAPI BackgroundTasks."""
    tracker.update(full_path, UploadStage.PROCESSING)
    upload = tracker.get(full_path)

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
            backend.insert_source_data(path=upload.path, data_hashes=data)
        elif upload.status.entity == BackendUploadType.RESULTS:
            backend.insert_model_data(path=upload.path, results=data)
        else:
            raise ValueError(f"Unknown upload type: {upload.status.entity}")

        tracker.update(full_path, UploadStage.COMPLETE)

    except Exception as e:
        error_context = {
            "resolution_path": full_path,
            "upload_type": upload_type,
            "bucket": bucket,
            "key": filename,
        }
        logger.error(
            f"Upload processing failed with context: {error_context}", exc_info=True
        )
        details = (
            f"Error: {e}. Context: "
            f"Upload type: {getattr(upload.status, 'entity', 'unknown')}, "
            f"Resolution path: {getattr(upload, 'path', 'unknown')}"
        )
        tracker.update(
            full_path,
            UploadStage.FAILED,
            details=details,
        )
        raise MatchboxServerFileError(message=details) from e
    finally:
        try:
            s3_client.delete_object(Bucket=bucket, Key=filename)
        except Exception as delete_error:  # noqa: BLE001
            logger.error(
                f"Failed to delete S3 file {bucket}/{filename}: {delete_error}"
            )


@celery.task(ignore_result=True, bind=True, max_retries=3)
def process_upload_celery(
    self: Task,
    upload_type: str,
    full_path: str,
    bucket: str,
    filename: str,
) -> None:
    """Celery task to process uploaded file, with only serialisable arguments."""
    initialise_celery_worker()

    celery_logger.info(f"Uploading data for resolution {full_path}")

    upload_function = partial(
        process_upload,
        backend=CELERY_BACKEND,
        tracker=CELERY_TRACKER,
        s3_client=CELERY_BACKEND.settings.datastore.get_client(),
    )

    try:
        upload_function(
            upload_type=upload_type,
            full_path=full_path,
            bucket=bucket,
            filename=filename,
        )
    except Exception as exc:  # noqa: BLE001
        celery_logger.error(
            f"Upload failed for resolution resolution {full_path}. Retrying..."
        )
        try:
            raise self.retry(exc=exc) from None
        except MaxRetriesExceededError:
            if CELERY_TRACKER:
                CELERY_TRACKER.update(
                    full_path, UploadStage.FAILED, f"Max retries exceeded: {exc}"
                )
            raise

    celery_logger.info(f"Upload complete for resolution {full_path}")
