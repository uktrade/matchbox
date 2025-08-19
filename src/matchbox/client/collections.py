"""Imperative interface to the Matchbox server."""

from typing import Self

from pydantic import BaseModel

from matchbox.client import _handler
from matchbox.client.models.models import Model
from matchbox.common.sources import SourceConfig


class Version(BaseModel):
    """Version of a collection.

    A version owns a set of resolutions, and is the main interface for reading and
    writing data on the Matchbox server. A version also points to the top resolution in
    a DAG.
    """

    id: int

    def add_source(
        self,
        source_config: SourceConfig,
        batch_size: int | None = None,
    ) -> None:
        """Indexes data on the server.

        Args:
            source_config: A SourceConfig with client set
            batch_size: the size of each batch when fetching data from the warehouse,
                which helps reduce the load on the database.
        """
        if not source_config.location.client:
            raise ValueError("Source client not set.")

        data_hashes = source_config.hash_data(batch_size=batch_size)
        _handler.index(
            version_id=self.id, source_config=source_config, data_hashes=data_hashes
        )

    def add_model(self, model: Model) -> None:
        """Stores results on the server.

        Args:
            model: A Model that has been run
        """
        if not model.results:
            raise ValueError("Model not run.")

        _handler.create_model(version_id=self.id, model_config=self.model_config)

    def set_current(self, resolution_name: str) -> None:
        """Mark version as current for its collection and point to a resolution."""
        _handler.set_current_version(
            version_id=self.id, resolution_name=resolution_name
        )

    def delete(self) -> None:
        """Delete a version and all its resolutions."""
        _handler.delete_version(self.id)

    def delete_resolution(self, resolution_name: str) -> None:
        """Delete a resolution within this version.

        Will delete:

        * The resolution itself
        * All descendants of the resolution, in this or other versions.
        * All endorsements of clusters made by those resolutions, either
        probabilities for models, or keys for sources.

        Will not delete:

        * The clusters themselves.
        """
        _handler.delete_resolution(resolution_name=resolution_name, version_id=self.id)


class Collection(BaseModel):
    """A named bookmark to a resolution.

    A collection can have a current and a next version.
    """

    _id: int | None = None
    name: str

    def get(self) -> Self:
        """Retrieve collection, fail if not exists."""
        self._id = _handler.get_collection(self.name, exists=True)
        return self

    def create(self) -> Self:
        """Create collection, fail if exists."""
        self._id = _handler.get_collection(self.name, exists=False)
        return self

    def get_id(self) -> int:
        """Verifies ID exists and returns it."""
        if not self._id:
            return RuntimeError("This collection hasn't been retrieved or created.")
        return self._id

    def current_version(self) -> Version | None:
        """Retrieve current version for this collection, if there is one."""
        if version_id := _handler.get_version(collection_id=self.get_id()):
            return Version(id=version_id)
        return None

    def new_version(self) -> Version:
        """Create a new version for this collection."""
        return Version(id=_handler.create_version(collection_id=self.get_id()))

    def delete(self) -> None:
        """Delete a collection and all its versions."""
        _handler.delete_collection(self.get_id())
