"""Functions and classes to define, run and register models."""

from typing import Any, ParamSpec, TypeVar

from pandas import DataFrame

from matchbox.client import _handler
from matchbox.client.models.dedupers.base import Deduper
from matchbox.client.models.linkers.base import Linker
from matchbox.client.results import Results
from matchbox.common.dtos import ModelAncestor, ModelMetadata, ModelType
from matchbox.common.exceptions import MatchboxResolutionNotFoundError

P = ParamSpec("P")
R = TypeVar("R")


class Model:
    """Unified model class for both linking and deduping operations."""

    def __init__(
        self,
        metadata: ModelMetadata,
        model_instance: Linker | Deduper,
        left_data: DataFrame,
        right_data: DataFrame | None = None,
    ):
        self.metadata = metadata
        self.model_instance = model_instance
        self.left_data = left_data
        self.right_data = right_data

    def insert_model(self) -> None:
        """Insert the model into the backend database."""
        _handler.insert_model(model=self.metadata)

    @property
    def results(self) -> Results:
        """Retrieve results associated with the model from the database."""
        results = _handler.get_model_results(name=self.metadata.name)
        return Results(probabilities=results, metadata=self.metadata)

    @results.setter
    def results(self, results: Results) -> None:
        """Write results associated with the model to the database."""
        if results.probabilities.shape[0] > 0:
            _handler.add_model_results(
                name=self.metadata.name, results=results.probabilities
            )

    @property
    def truth(self) -> int:
        """Retrieve the truth threshold for the model."""
        return _handler.get_model_truth(name=self.metadata.name)

    @truth.setter
    def truth(self, truth: float) -> None:
        """Set the truth threshold for the model."""
        # Convert float to int
        _handler.set_model_truth(name=self.metadata.name, truth=int(truth * 100))

    @property
    def ancestors(self) -> dict[str, float]:
        """Retrieve the ancestors of the model."""
        return {
            ancestor.name: ancestor.truth
            for ancestor in _handler.get_model_ancestors(name=self.metadata.name)
        }

    @property
    def ancestors_cache(self) -> dict[str, float]:
        """Retrieve the ancestors cache of the model."""
        return {
            ancestor.name: ancestor.truth
            for ancestor in _handler.get_model_ancestors_cache(name=self.metadata.name)
        }

    @ancestors_cache.setter
    def ancestors_cache(self, ancestors_cache: dict[str, float]) -> None:
        """Set the ancestors cache of the model."""
        _handler.set_model_ancestors_cache(
            name=self.metadata.name,
            ancestors=[
                ModelAncestor(name=k, truth=v) for k, v in ancestors_cache.items()
            ],
        )

    def delete(self, certain: bool = False) -> bool:
        """Delete the model from the database."""
        result = _handler.delete_model(name=self.metadata.name, certain=certain)
        return result.success

    def run(self) -> Results:
        """Execute the model pipeline and return results."""
        if self.metadata.type == ModelType.LINKER:
            if self.right_data is None:
                raise MatchboxResolutionNotFoundError(
                    "Right dataset required for linking"
                )

            results = self.model_instance.link(
                left=self.left_data, right=self.right_data
            )
        else:
            results = self.model_instance.dedupe(data=self.left_data)

        return Results(
            probabilities=results,
            model=self,
            metadata=self.metadata,
        )


def make_model(
    model_name: str,
    description: str,
    model_class: type[Linker] | type[Deduper],
    model_settings: dict[str, Any],
    left_data: DataFrame,
    left_resolution: str,
    right_data: DataFrame | None = None,
    right_resolution: str | None = None,
) -> Model:
    """Create a unified model instance for either linking or deduping operations.

    Args:
        model_name: Your unique identifier for the model
        description: Description of the model run
        model_class: Either Linker or Deduper class
        model_settings: Configuration settings for the model
        left_data: Primary dataset
        left_resolution: Resolution name for primary model or dataset
        right_data: Secondary dataset (linking only)
        right_resolution: Resolution name for secondary model or dataset (linking only)

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

    metadata = ModelMetadata(
        name=model_name,
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
