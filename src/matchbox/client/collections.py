"""Imperative interface to the Matchbox server."""

from typing import Self

from pyarrow import Table
from pydantic import BaseModel

from matchbox.client import _handler
from matchbox.common.dtos import ModelConfig
from matchbox.common.sources import SourceConfig

# TODO: how do we return a DAG from the server?
# Improve model
# Remove visualisation


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
        data_hashes: Table,
    ):
        """Indexes data on the server.

        Args:
            source_config: A SourceConfig with client set
            data_hashes: Result of hashing data source
        """
        if not source_config.location.client:
            raise ValueError("Source client not set.")

        _handler.create_source(
            version_id=self.id, source_config=source_config, data_hashes=data_hashes
        )

    def add_model(self, model_config: ModelConfig, results: Table) -> None:
        """Stores results on the server.

        Args:
            model_config: a model configuration
            results: output from the model
        """
        _handler.create_model(
            version_id=self.id, model_config=self.model_config, results=results
        )

    def set_model_threshold(self, threshold: int):
        """Set probability threshold for model."""
        _handler.set_model_threshold(version_id=self.id, threshold=threshold)

    def get_source(self, resolution_name: str) -> SourceConfig | None:
        """Retrieve source config by resolution name."""
        return _handler.get_source_config(name=resolution_name)

    def get_model(self, resolution_name: str) -> ModelConfig | None:
        """Retrieve model config by resolution name."""
        return _handler.get_model(name=resolution_name)

    def get_dag(self):
        """Return complete DAG from a current version."""
        return _handler.get_dag(version_id=self.id)

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
        return _handler.get_version(collection_id=self.get_id())

    def new_version(self) -> Version:
        """Create a new version for this collection."""
        return Version(id=_handler.create_version(collection_id=self.get_id()))

    def delete(self) -> None:
        """Delete a collection and all its versions."""
        _handler.delete_collection(self.get_id())
