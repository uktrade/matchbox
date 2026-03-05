"""Base classes for resolver methodologies."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import ClassVar

import polars as pl
from pydantic import BaseModel, ConfigDict

from matchbox.common.dtos import ModelResolutionName, ResolverType


class ResolverSettings(BaseModel, ABC):
    """Base settings type for resolver methodologies."""

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def validate_inputs(self, model_names: Iterable[ModelResolutionName]) -> None:
        """Validates whether the models' clusters can be computed with this object.

        Should be used in conjunction with ResolverMethod.compute_clusters().

        Args:
            model_names: A list of model names that will be processed using
                these settings

        Raises:
            RuntimeError: if supplied model names don't match the settings
        """
        ...


class ResolverMethod(BaseModel, ABC):
    """Base class for resolver methodologies."""

    resolver_type: ClassVar[ResolverType]
    settings: ResolverSettings

    @abstractmethod
    def compute_clusters(
        self, model_edges: Mapping[ModelResolutionName, pl.DataFrame]
    ) -> pl.DataFrame:
        """Compute cluster assignments from model edges.

        Args:
            model_edges: A mapping of model names to model edges which conform
                to SCHEMA_MODEL_EDGES

        Returns:
            A Polars DataFrame which conforms to SCHEMA_CLUSTERS

        Raises:
            RuntimeError: if supplied model names don't match the Resolver's settings
        """
        ...
