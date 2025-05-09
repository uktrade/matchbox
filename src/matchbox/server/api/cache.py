"""A simple in-memory cache of uploaded metadata and processing status."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from matchbox.common.dtos import BackendUploadType, ModelMetadata, UploadStatus
from matchbox.common.exceptions import MatchboxServerFileError
from matchbox.common.logging import logger
from matchbox.common.sources import SourceConfig
from matchbox.server.api.arrow import s3_to_recordbatch
from matchbox.server.base import MatchboxDBAdapter


class MetadataCacheEntry(BaseModel):
    """Cache entry for uploaded metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SourceConfig | ModelMetadata
    upload_type: BackendUploadType
    update_timestamp: datetime
    status: UploadStatus


class MetadataStore:
    """A simple in-memory cache of uploaded metadata and processing status.

    Uses a janitor pattern for lazy cleanup. Entries expire after a period of
    inactivity, where activity is defined as any update to the entry's status
    or access via the upload endpoint.
    """

    def __init__(self, expiry_minutes: int = 30):
        """Initialise the cache with an expiry time in minutes."""
        self._store: dict[str, MetadataCacheEntry] = {}
        self.expiry_minutes = expiry_minutes

    def _is_expired(self, entry: MetadataCacheEntry) -> bool:
        """Check if a cache entry has expired due to inactivity."""
        return datetime.now() - entry.update_timestamp > timedelta(
            minutes=self.expiry_minutes
        )

    def _cleanup_if_needed(self) -> None:
        """Lazy cleanup - remove all expired entries."""
        expired = [id for id, entry in self._store.items() if self._is_expired(entry)]
        for id in expired:
            del self._store[id]

    def _update_timestamp(self, cache_id: str) -> None:
        """Update the timestamp on an entry to prevent expiry."""
        if entry := self._store.get(cache_id):
            entry.update_timestamp = datetime.now()

    def cache_source(self, metadata: SourceConfig) -> str:
        """Cache source metadata and return ID."""
        self._cleanup_if_needed()
        cache_id = str(uuid.uuid4())

        self._store[cache_id] = MetadataCacheEntry(
            metadata=metadata,
            upload_type=BackendUploadType.INDEX,
            update_timestamp=datetime.now(),
            status=UploadStatus(
                id=cache_id, status="awaiting_upload", entity=BackendUploadType.INDEX
            ),
        )
        return cache_id

    def cache_model(self, metadata: ModelMetadata) -> str:
        """Cache model results metadata and return ID."""
        self._cleanup_if_needed()
        cache_id = str(uuid.uuid4())

        self._store[cache_id] = MetadataCacheEntry(
            metadata=metadata,
            upload_type=BackendUploadType.RESULTS,
            update_timestamp=datetime.now(),
            status=UploadStatus(
                id=cache_id, status="awaiting_upload", entity=BackendUploadType.RESULTS
            ),
        )
        return cache_id

    def get(self, cache_id: str) -> MetadataCacheEntry | None:
        """Retrieve metadata by ID if not expired. Updates timestamp on access."""
        self._cleanup_if_needed()

        entry = self._store.get(cache_id)
        if not entry:
            return None

        if self._is_expired(entry):
            del self._store[cache_id]
            return None

        self._update_timestamp(cache_id)
        return entry

    def update_status(
        self, cache_id: str, status: str, details: str | None = None
    ) -> bool:
        """Update the status of an entry.

        Raises:
            KeyError: If entry not found.
        """
        if entry := self._store.get(cache_id):
            entry.status.status = status
            if details is not None:
                entry.status.details = details
            entry.update_timestamp = datetime.now()
            return True
        raise KeyError(f"Cache entry {cache_id} not found.")

    def remove(self, cache_id: str) -> bool:
        """Remove an entry from the store."""
        self._cleanup_if_needed()
        if cache_id in self._store:
            del self._store[cache_id]
            return True
        return False


@asynccontextmanager
async def heartbeat(
    metadata_store: MetadataStore, upload_id: str, interval_seconds: int = 300
):
    """Context manager that updates status with a heartbeat.

    Args:
        metadata_store: Store for updating status
        upload_id: ID of the upload being processed
        interval_seconds: How often to send heartbeat (default 5 minutes)
    """
    heartbeat_task = None

    async def _heartbeat():
        while True:
            await asyncio.sleep(interval_seconds)
            timestamp = datetime.now().isoformat()
            metadata_store.update_status(
                cache_id=upload_id,
                status="processing",
                details=f"Still processing... Last heartbeat: {timestamp}",
            )

    try:
        heartbeat_task = asyncio.create_task(_heartbeat())
        yield
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


def process_upload(
    backend: MatchboxDBAdapter,
    upload_id: str,
    bucket: str,
    key: str,
    metadata_store: MetadataStore,
) -> None:
    """Background task to process uploaded file."""
    metadata_store.update_status(upload_id, "processing")
    client = backend.settings.datastore.get_client()
    upload = metadata_store.get(upload_id)

    try:
        data = pa.Table.from_batches(
            [
                batch
                for batch in s3_to_recordbatch(client=client, bucket=bucket, key=key)
            ]
        )

        if upload.upload_type == BackendUploadType.INDEX:
            backend.index(source=upload.metadata, data_hashes=data)
        elif upload.upload_type == BackendUploadType.RESULTS:
            backend.set_model_results(model=upload.metadata.name, results=data)
        else:
            raise ValueError(f"Unknown upload type: {upload.upload_type}")

        metadata_store.update_status(upload_id, "complete")

    except Exception as e:
        error_context = {
            "upload_id": upload_id,
            "upload_type": getattr(upload, "upload_type", "unknown"),
            "metadata": getattr(upload, "metadata", "unknown"),
            "bucket": bucket,
            "key": key,
        }
        logger.error(
            f"Upload processing failed with context: {error_context}", exc_info=True
        )
        details = (
            f"Error: {e}. Context: "
            f"Upload type: {getattr(upload, 'upload_type', 'unknown')}, "
            f"SourceConfig: {getattr(upload, 'metadata', 'unknown')}"
        )
        metadata_store.update_status(
            upload_id,
            "failed",
            details=details,
        )
        raise MatchboxServerFileError(message=details) from e
    finally:
        try:
            client.delete_object(Bucket=bucket, Key=key)
        except Exception as delete_error:
            logger.error(f"Failed to delete S3 file {bucket}/{key}: {delete_error}")
