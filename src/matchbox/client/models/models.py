"""Functions and classes to define, run and register models."""

import inspect
from typing import ParamSpec, TypeVar, overload

import polars as pl

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.models import dedupers, linkers
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.queries import Query
from matchbox.client.results import Results
from matchbox.common.dtos import ModelConfig, ModelType, Resolution
from matchbox.common.graph import ResolutionType
from matchbox.common.logging import logger

P = ParamSpec("P")
R = TypeVar("R")

default_dedupers = {
    name: obj for name, obj in inspect.getmembers(dedupers, inspect.isclass)
}
default_linkers = {
    name: obj for name, obj in inspect.getmembers(linkers, inspect.isclass)
}


_MODEL_CLASSES = {**default_dedupers, **default_linkers}


def add_model_class(ModelClass: type[Linker] | type[Deduper]) -> None:
    """Add custom deduper or linker."""
    if issubclass(ModelClass, Linker) or issubclass(ModelClass, Deduper):
        _MODEL_CLASSES[ModelClass.__name__] = ModelClass
    else:
        raise ValueError("The argument is not a proper subclass of Deduper or Linker.")


class Model:
    """Unified model class for both linking and deduping operations."""

    @overload
    def __init__(
        self,
        name: str,
        description: str | None,
        model_class: type[Deduper],
        model_settings: DeduperSettings | dict,
        left_query: Query,
        right_query: None = None,
        truth: float = 1.0,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        description: str | None,
        model_class: type[Linker],
        model_settings: LinkerSettings | dict,
        left_query: Query,
        right_query: Query,
        truth: float = 1.0,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        description: str | None,
        model_class: type[Deduper] | type[Linker] | str,
        model_settings: DeduperSettings | LinkerSettings | dict,
        left_query: Query,
        right_query: Query | None = None,
        truth: float = 1.0,
    ):
        """Create a new model instance.

        Args:
            name: Unique name for the model
            description: Optional description of the model
            truth: Truth threshold. Defaults to 1.0. Can be set later after analysis.
            model_class: Class of Linker or Deduper, or its name.
            model_settings: Appropriate settings object to pass to model class.
            left_query: The query that will get the data to deduplicate, or the data to
                link on the left.
            right_query: The query that will get the data to link on the right.
        """
        self.name = name
        self.description = description
        self._truth: int = _truth_float_to_int(truth)
        self.left_query = left_query
        self.right_query = right_query
        self.results: Results | None = None

        if isinstance(model_class, str):
            model_class: type[Linker | Deduper] = _MODEL_CLASSES[model_class]
        self.model_instance = model_class(settings=model_settings)

        model_type: ModelType = (
            ModelType.LINKER if issubclass(model_class, Linker) else ModelType.DEDUPER
        )

        self.config = ModelConfig(
            type=model_type,
            model_class=model_class.__name__,
            model_settings=model_settings.model_dump_json(),
            left_query=left_query.config,
            right_query=right_query.config,
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

    def from_resolution(cls, resolution: Resolution) -> "Model":
        """Reconstruct from Resolution."""
        assert resolution.resolution_type == "model", (
            "Resolution must be of type 'model'"
        )
        assert isinstance(resolution.config, ModelConfig), "Config must be ModelConfig"
        return cls(
            name=resolution.name,
            description=resolution.description,
            model_class=resolution.config.model_class,
            model_settings=resolution.config.model_settings,
            left_query=resolution.config.left_query,
            right_query=resolution.config.right_query,
            truth=resolution.truth,
        )

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

    def delete(self, certain: bool = False) -> bool:
        """Delete the model from the database."""
        result = _handler.delete_resolution(name=self.name, certain=certain)
        return result.success

    def run(self, for_validation: bool = False) -> Results:
        """Execute the model pipeline and return results.

        Args:
            for_validation: Whether to download and store extra data to explore and
                    score results.
        """
        left_df: pl.DataFrame = self.left_query.run(
            return_leaf_id=for_validation, batch_size=settings.batch_size
        )

        if self.config.type == ModelType.LINKER:
            right_df: pl.DataFrame = self.right_query.run(return_leaf_id=for_validation)

            self.model_instance.prepare(left_df, right_df)
            results = self.model_instance.link(left=left_df, right=right_df)
        else:
            self.model_instance.prepare(left_df)
            results = self.model_instance.dedupe(data=self.left_data)

        if not for_validation:
            self.results = Results(probabilities=results)
        else:
            self.results = Results(
                probabilities=results,
                left_data=left_df.to_arrow(),
                right_data=right_df.to_arrow(),
            )

        return self.results

    def sync(self) -> None:
        """Send the model config, truth and results to the server."""
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

        _handler.set_truth(name=self.name, truth=self._truth)

        if self.results:
            _handler.set_results(name=self.name, results=self.results.probabilities)

    def download_results(self) -> Results:
        """Retrieve results associated with the model from the database."""
        results = _handler.get_results(name=self.name)
        return Results(probabilities=results, metadata=self.config)


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
