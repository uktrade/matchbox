"""Worker logic to process user uploads."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from functools import partial
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import redis
from celery import Celery, Task
from celery.exceptions import MaxRetriesExceededError
from celery.utils.log import get_task_logger
from fastapi import UploadFile
from pyarrow import parquet as pq

from matchbox.common.arrow import SCHEMA_INDEX, SCHEMA_RESULTS
from matchbox.common.dtos import (
    ResolutionPath,
    ResolutionType,
)
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.logging import log_mem_usage, logger
from matchbox.server.base import (
    MatchboxBackends,
    MatchboxDBAdapter,
    MatchboxServerSettings,
    get_backend_settings,
    settings_to_backend,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any

celery_logger = get_task_logger(__name__)

# -- Upload trackers --


class UploadTracker(ABC):
    """Upload error tracker."""

    @abstractmethod
    def get(self, upload_id: str) -> str | None:
        """Retrieve error message from tracker."""
        ...

    @abstractmethod
    def set(self, upload_id: str, message: str) -> None:
        """Add error message to tracker."""
        ...


class InMemoryUploadTracker:
    """In-memory error tracker."""

    def __init__(self) -> None:
        """Initialise dictionary."""
        self.tracker: dict[str, str] = {}

    def get(self, upload_id: str) -> str | None:  # noqa: D102
        return self.tracker.get(upload_id)

    def set(self, upload_id: str, message: str) -> None:  # noqa: D102
        self.tracker[upload_id] = message


class RedisUploadTracker:
    """Error tracker backed by Redis."""

    def __init__(
        self, redis_url: str, expiry_minutes: int, key_space: str = "upload"
    ) -> None:
        """Connect Redis and initialise tracker object."""
        self.expiry_minutes = expiry_minutes
        self.redis = redis.Redis.from_url(redis_url)
        self.key_prefix = f"{key_space}:"

    def get(self, upload_id: str) -> str | None:  # noqa: D102
        results = self.redis.get(f"{self.key_prefix}{upload_id}")
        if results:
            return results.decode("utf-8")
        return None

    def set(self, upload_id: str, message: str) -> None:  # noqa: D102
        expiry_seconds = self.expiry_minutes * 60
        self.redis.setex(f"{self.key_prefix}{upload_id}", expiry_seconds, message)


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


def file_to_s3(client: S3Client, bucket: str, key: str, file: UploadFile) -> str:
    """Upload a PyArrow Table to S3 and return the key.

    Args:
        client: The S3 client to use.
        bucket: The S3 bucket to upload to.
        key: The key to upload to.
        file: The file to upload.

    Raises:
        MatchboxServerFileError: If the file is not a valid Parquet file or the schema
            does not match the expected schema.

    Returns:
        The key of the uploaded file.
    """
    try:
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


CELERY_SETTINGS: MatchboxServerSettings = get_backend_settings(
    MatchboxServerSettings().backend_type
)()
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
    # Limit number of tasks a worker can complete before restarting
    # This may prevent OOM errors caused by memory not being deallocated properly
    worker_max_tasks_per_child=5,
)


def initialise_celery_worker() -> None:
    """Initialise backend and tracker for celery worker and set up logging."""
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
    resolution_path: ResolutionPath,
    upload_id: str,
    bucket: str,
    filename: str,
) -> None:
    """Generic task to process uploaded file, usable by FastAPI BackgroundTasks."""
    try:
        resolution = backend.get_resolution(path=resolution_path)

        batches = [
            batch
            for batch in s3_to_recordbatch(
                client=s3_client, bucket=bucket, key=filename
            )
        ]

        # If successful, either backend method marks resolution as complete
        if resolution.resolution_type == ResolutionType.SOURCE:
            backend.insert_source_data(
                path=resolution_path,
                data_hashes=pa.Table.from_batches(batches, schema=SCHEMA_INDEX),
            )
        else:
            backend.insert_model_data(
                path=resolution_path,
                results=pa.Table.from_batches(batches, schema=SCHEMA_RESULTS),
            )

    except Exception as e:
        # After failure, signal to clients they can try again
        backend.unlock_resolution_data(path=resolution_path)

        error_context = {
            "upload_id": upload_id,
            "resolution_path": str(resolution_path),
            "bucket": bucket,
            "key": filename,
        }
        logger.error(
            f"Upload processing failed with context: {error_context}", exc_info=True
        )

        # Attach error to upload ID to inform clients
        tracker.set(upload_id=upload_id, message=str(e))

        raise MatchboxServerFileError(
            message=f"Error: {e}. Context: {error_context}"
        ) from e
    finally:
        try:
            s3_client.delete_object(Bucket=bucket, Key=filename)
        except Exception as delete_error:  # noqa: BLE001
            logger.error(
                f"Failed to delete S3 file {bucket}/{filename}: {delete_error}"
            )


@celery.task(ignore_result=True, bind=True, max_retries=3)
def process_upload_celery(
    self: Task, resolution_path_json: str, upload_id: str, bucket: str, filename: str
) -> None:
    """Celery task to process uploaded file, with only serialisable arguments."""
    initialise_celery_worker()
    resolution_path = ResolutionPath.model_validate_json(resolution_path_json)

    # If using Postgres, we must reset the global database connections
    # to avoid using inherited C-pointers from the parent process (ADBC).
    if CELERY_SETTINGS.backend_type == MatchboxBackends.POSTGRES:
        from matchbox.server.postgresql.db import MBDB  # noqa: PLC0415

        MBDB._disconnect_adbc()

    celery_logger.info(
        "Uploading data for resolution %s, ID %s", str(resolution_path), upload_id
    )
    log_mem_usage()

    upload_function = partial(
        process_upload,
        backend=CELERY_BACKEND,
        tracker=CELERY_TRACKER,
        s3_client=CELERY_BACKEND.settings.datastore.get_client(),
    )

    try:
        upload_function(
            resolution_path=resolution_path,
            upload_id=upload_id,
            bucket=bucket,
            filename=filename,
        )
        celery_logger.info(
            "Upload complete for %s, ID %s", str(resolution_path), upload_id
        )

    except Exception as exc:  # noqa: BLE001
        celery_logger.error(
            "Upload failed for resolution %s, ID %s. Retrying...",
            str(resolution_path),
            upload_id,
        )
        try:
            raise self.retry(exc=exc) from None
        except MaxRetriesExceededError:
            CELERY_TRACKER.set(upload_id, f"Max retries exceeded: {exc}")
            raise

    finally:
        log_mem_usage()

        # Cleanup connections
        if CELERY_SETTINGS.backend_type == MatchboxBackends.POSTGRES:
            from matchbox.server.postgresql.db import MBDB  # noqa: PLC0415

            MBDB._disconnect()
            MBDB._disconnect_adbc()
