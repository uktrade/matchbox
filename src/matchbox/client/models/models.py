"""Functions and classes to define, run and register models."""

from typing import Any, ParamSpec, TypeVar, overload

import polars as pl

from matchbox.client import _handler
from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.results import Results
from matchbox.common.dtos import ModelConfig, ModelType, Resolution
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.graph import ModelResolutionName, ResolutionName, ResolutionType
from matchbox.common.logging import logger

P = ParamSpec("P")
R = TypeVar("R")


class Model:
    """Unified model class for both linking and deduping operations."""

    @overload
    def __init__(
        self,
        name: str,
        description: str | None,
        model_instance: Deduper,
        left_resolution: ResolutionName,
        left_data: pl.DataFrame,
        right_resolution: None = None,
        right_data: None = None,
        truth: float = 1.0,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        description: str | None,
        model_instance: Linker,
        left_resolution: ResolutionName,
        left_data: pl.DataFrame,
        right_resolution: ResolutionName,
        right_data: pl.DataFrame,
        truth: float = 1.0,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        description: str | None,
        model_instance: Linker | Deduper,
        left_resolution: ResolutionName,
        left_data: pl.DataFrame,
        right_resolution: ResolutionName | None = None,
        right_data: pl.DataFrame | None = None,
        truth: float = 1.0,
    ):
        """Create a new model instance.

        Args:
            name: Unique name for the model
            description: Optional description of the model
            truth: Truth threshold. Defaults to 1.0. Can be set later after analysis.
            model_instance: Instance of Linker or Deduper
            left_resolution: The name of the resolution that produced left_data. This is
                the only resolution for deduping.
            left_data: Primary data for the model. This is the only data for deduping.
            right_resolution: The name of the resolution that produced right_data.
                Required for linking
            right_data: Secondary data for the model. Required for linking
        """
        self.name = name
        self.description = description
        self.model_instance = model_instance
        self.left_data = left_data
        self.right_data = right_data

        self._truth: int = _truth_float_to_int(truth)

        model_type: ModelType = (
            ModelType.LINKER
            if isinstance(model_instance, Linker)
            else ModelType.DEDUPER
        )

        self.config = ModelConfig(
            type=model_type,
            left_resolution=left_resolution,
            right_resolution=right_resolution,
        )

    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        return Resolution(
            name=self.name,
            description=self.description,
            truth=self._truth,
            resolution_type=ResolutionType.MODEL,
            config=self.config,
        )

    @classmethod
    def from_resolution(
        cls,
        resolution: Resolution,
        model_instance: Linker | Deduper,
        left_data: pl.DataFrame,
        right_data: pl.DataFrame | None = None,
    ) -> "Model":
        """Reconstruct from Resolution."""
        assert resolution.resolution_type == ResolutionType.MODEL, (
            "Resolution must be of type 'model'"
        )
        assert isinstance(resolution.config, ModelConfig), "Config must be ModelConfig"
        return cls(
            name=resolution.name,
            description=resolution.description,
            truth=resolution.truth,
            metadata=resolution.config,
            model_instance=model_instance,
            left_data=left_data,
            right_data=right_data,
        )

    def insert_model(self) -> None:
        """Insert the model into the backend database."""
        resolution = self.to_resolution()
        if existing_resolution := _handler.get_resolution(name=self.name):
            # Check if config matches
            if existing_resolution.config != self.config:
                raise ValueError(
                    f"Resolution {self.name} already exists with different "
                    "configuration. Please delete the existing resolution "
                    "or use a different name. "
                )
            log_prefix = f"Resolution {self.name}"
            logger.warning("Already exists. Passing.", prefix=log_prefix)
        else:
            _handler.create_resolution(resolution=resolution)

    @property
    def results(self) -> Results:
        """Retrieve results associated with the model from the database."""
        results = _handler.get_results(name=self.name)
        return Results(probabilities=results, metadata=self.config)

    @results.setter
    def results(self, results: Results) -> None:
        """Write results associated with the model to the database."""
        if results.probabilities.shape[0] > 0:
            _handler.set_results(name=self.name, results=results.probabilities)

    @property
    def truth(self) -> float | None:
        """Returns the truth threshold for the model as a float."""
        if self._truth is not None:
            return _truth_int_to_float(self._truth)
        return None

    @truth.setter
    def truth(self, truth: float) -> None:
        """Set the truth threshold for the model."""
        self._truth = _truth_float_to_int(truth)
        _handler.set_truth(name=self.name, truth=self._truth)

    def delete(self, certain: bool = False) -> bool:
        """Delete the model from the database."""
        result = _handler.delete_resolution(name=self.name, certain=certain)
        return result.success

    def run(self) -> Results:
        """Execute the model pipeline and return results."""
        if self.config.type == ModelType.LINKER:
            if self.right_data is None:
                raise MatchboxResolutionNotFoundError("Right data required for linking")

            results = self.model_instance.link(
                left=self.left_data, right=self.right_data
            )
        else:
            results = self.model_instance.dedupe(data=self.left_data)

        return Results(
            probabilities=results,
            model=self,
            metadata=self.config,
        )


def make_model(
    name: ModelResolutionName,
    description: str,
    model_class: type[Linker] | type[Deduper],
    model_settings: dict[str, Any],
    left_data: pl.DataFrame,
    left_resolution: ResolutionName,
    right_data: pl.DataFrame | None = None,
    right_resolution: ResolutionName | None = None,
) -> Model:
    """Create a unified model instance for either linking or deduping operations.

    Args:
        name: Your unique identifier for the model
        description: Description of the model run
        model_class: Either Linker or Deduper class
        model_settings: Configuration settings for the model
        left_data: Primary data
        left_resolution: Resolution name for primary model or source
        right_data: Secondary data (linking only)
        right_resolution: Resolution name for secondary model or source (linking only)

    Returns:
        Model: Configured model instance ready for execution
    """
    model_type = (
        ModelType.LINKER if issubclass(model_class, Linker) else ModelType.DEDUPER
    )

    model_instance = model_class.from_settings(**model_settings)

    if model_type == ModelType.LINKER:
        model_instance.prepare(left=left_data, right=right_data)
    else:
        model_instance.prepare(data=left_data)

    return Model(
        name=name,
        description=description,
        model_instance=model_instance,
        left_resolution=left_resolution,
        left_data=left_data,
        right_resolution=right_resolution,
        right_data=right_data,
    )


def _truth_float_to_int(truth: float) -> int:
    """Convert user input float truth values to int."""
    if isinstance(truth, float) and 0.0 <= truth <= 1.0:
        return round(truth * 100)
    else:
        raise ValueError(f"Truth value {truth} not a valid probability")


def _truth_int_to_float(truth: int) -> float:
    """Convert backend int truth values to float."""
    if isinstance(truth, int) and 0 <= truth <= 100:
        return float(truth / 100)
    else:
        raise ValueError(f"Truth value {truth} not valid")
