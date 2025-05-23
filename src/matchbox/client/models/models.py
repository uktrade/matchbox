"""Functions and classes to define, run and register models."""

from typing import Any, ParamSpec, TypeVar

from pandas import DataFrame

from matchbox.client import _handler
from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.results import Results
from matchbox.common.dtos import (
    ModelAncestor,
    ModelConfig,
    ModelResolutionName,
    ModelType,
    ResolutionName,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.logging import logger

P = ParamSpec("P")
R = TypeVar("R")


class Model:
    """Unified model class for both linking and deduping operations."""

    def __init__(
        self,
        metadata: ModelConfig,
        model_instance: Linker | Deduper,
        left_data: DataFrame,
        right_data: DataFrame | None = None,
    ):
        """Create a new model instance."""
        self.model_config = metadata
        self.model_instance = model_instance
        self.left_data = left_data
        self.right_data = right_data

    def insert_model(self) -> None:
        """Insert the model into the backend database."""
        if model_config := _handler.get_model(name=self.model_config.name):
            if model_config != self.model_config:
                raise ValueError(
                    f"Model {self.model_config.name} already exists with "
                    "different configuration. Please delete the existing model "
                    "or use a different name. "
                )
            logger.warning(f"Model {model_config.name} already exists. Passing.")
        else:
            _handler.insert_model(model_config=self.model_config)

    @property
    def results(self) -> Results:
        """Retrieve results associated with the model from the database."""
        results = _handler.get_model_results(name=self.model_config.name)
        return Results(probabilities=results, metadata=self.model_config)

    @results.setter
    def results(self, results: Results) -> None:
        """Write results associated with the model to the database."""
        if results.probabilities.shape[0] > 0:
            _handler.add_model_results(
                name=self.model_config.name, results=results.probabilities
            )

    @property
    def truth(self) -> float:
        """Retrieve the truth threshold for the model."""
        truth = _handler.get_model_truth(name=self.model_config.name)
        return _truth_int_to_float(truth)

    @truth.setter
    def truth(self, truth: float) -> None:
        """Set the truth threshold for the model."""
        _handler.set_model_truth(
            name=self.model_config.name, truth=_truth_float_to_int(truth)
        )

    @property
    def ancestors(self) -> dict[str, float]:
        """Retrieve the ancestors of the model."""
        return {
            ancestor.name: _truth_int_to_float(ancestor.truth)
            for ancestor in _handler.get_model_ancestors(name=self.model_config.name)
        }

    @property
    def ancestors_cache(self) -> dict[str, float]:
        """Retrieve the ancestors cache of the model."""
        return {
            ancestor.name: _truth_int_to_float(ancestor.truth)
            for ancestor in _handler.get_model_ancestors_cache(
                name=self.model_config.name
            )
        }

    @ancestors_cache.setter
    def ancestors_cache(self, ancestors_cache: dict[str, float]) -> None:
        """Set the ancestors cache of the model."""
        _handler.set_model_ancestors_cache(
            name=self.model_config.name,
            ancestors=[
                ModelAncestor(name=k, truth=_truth_float_to_int(v))
                for k, v in ancestors_cache.items()
            ],
        )

    def delete(self, certain: bool = False) -> bool:
        """Delete the model from the database."""
        result = _handler.delete_resolution(
            name=self.model_config.name, certain=certain
        )
        return result.success

    def run(self) -> Results:
        """Execute the model pipeline and return results."""
        if self.model_config.type == ModelType.LINKER:
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
            metadata=self.model_config,
        )


def make_model(
    name: ModelResolutionName,
    description: str,
    model_class: type[Linker] | type[Deduper],
    model_settings: dict[str, Any],
    left_data: DataFrame,
    left_resolution: ResolutionName,
    right_data: DataFrame | None = None,
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

    if model_type == ModelType.LINKER and (
        right_data is None or right_resolution is None
    ):
        raise ValueError("Linking requires both right_data and right_resolution")

    model_instance = model_class.from_settings(**model_settings)

    if model_type == ModelType.LINKER:
        model_instance.prepare(left=left_data, right=right_data)
    else:
        model_instance.prepare(data=left_data)

    metadata = ModelConfig(
        name=name,
        description=description,
        type=model_type.value,
        left_resolution=left_resolution,
        right_resolution=right_resolution,
    )

    return Model(
        metadata=metadata,
        model_instance=model_instance,
        left_data=left_data,
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
