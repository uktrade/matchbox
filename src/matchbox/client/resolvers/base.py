"""Base classes for resolver methodologies."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TypeAlias

import polars as pl
from pydantic import BaseModel, ConfigDict

ResolutionName: TypeAlias = str


class ResolverSettings(BaseModel):
    """Base settings type for resolver methodologies."""

    model_config = ConfigDict(extra="forbid")


class ResolverMethod(BaseModel, ABC):
    """Base class for resolver methodologies."""

    settings: ResolverSettings

    @abstractmethod
    def compute_clusters(
        self,
        model_edges: Mapping[ResolutionName, pl.DataFrame],
        resolver_assignments: Mapping[ResolutionName, pl.DataFrame],
    ) -> pl.DataFrame:
        """Compute cluster assignments from model edges and resolver inputs."""
        ...
