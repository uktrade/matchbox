"""Functions and classes to define, run and register models."""

import inspect
import json
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from polars import DataFrame

from matchbox.client import _handler
from matchbox.client.models import dedupers, linkers
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.queries import Query
from matchbox.client.results import ModelResults
from matchbox.common.dtos import (
    ModelConfig,
    ModelResolutionName,
    ModelResolutionPath,
    ModelType,
    Resolution,
    ResolutionType,
    SourceResolutionName,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.hash import hash_model_results
from matchbox.common.logging import logger, profile_time
from matchbox.common.transform import threshold_float_to_int, threshold_int_to_float

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.sources import Source
else:
    DAG = Any
    Source = Any

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


_MODEL_CLASSES = {
    **{name: obj for name, obj in inspect.getmembers(dedupers, inspect.isclass)},
    **{name: obj for name, obj in inspect.getmembers(linkers, inspect.isclass)},
}


def add_model_class(ModelClass: type[Linker] | type[Deduper]) -> None:
    """Add custom deduper or linker."""
    if issubclass(ModelClass, Linker) or issubclass(ModelClass, Deduper):
        _MODEL_CLASSES[ModelClass.__name__] = ModelClass
    else:
        raise ValueError("The argument is not a proper subclass of Deduper or Linker.")


def post_run(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that a method is called after model run.

    Raises:
        RuntimeError: If run hasn't happened.
    """

    @wraps(method)
    def wrapper(self: "Model", *args: Any, **kwargs: Any) -> T:
        if self.results is None:
            raise RuntimeError(
                "The model must be run before attempting this operation."
            )
        return method(self, *args, **kwargs)

    return wrapper


class Model:
    """Unified model class for both linking and deduping operations."""

    @overload
    def __init__(
        self,
        name: str,
        dag: DAG,
        model_class: type[Deduper],
        model_settings: DeduperSettings | dict,
        left_query: Query,
        right_query: None = None,
        truth: float = 1.0,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        dag: DAG,
        name: str,
        model_class: type[Linker],
        model_settings: LinkerSettings | dict,
        left_query: Query,
        right_query: Query,
        truth: float = 1.0,
        description: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        dag: DAG,
        name: str,
        model_class: type[Deduper] | type[Linker] | str,
        model_settings: DeduperSettings | LinkerSettings | dict,
        left_query: Query,
        right_query: Query | None = None,
        truth: float = 1.0,
        description: str | None = None,
    ):
        """Create a new model instance.

        Args:
            dag: DAG containing this model.
            name: Unique name for the model
            truth: Truth threshold. Defaults to 1.0. Can be set later after analysis.
            model_class: Class of Linker or Deduper, or its name.
            model_settings: Appropriate settings object to pass to model class.
            left_query: The query that will get the data to deduplicate, or the data to
                link on the left.
            right_query: The query that will get the data to link on the right.
            description: Optional description of the model
        """
        self.dag = dag
        self.name = name
        self.description = description
        self._truth: int = threshold_float_to_int(truth)
        self.left_query = left_query
        self.right_query = right_query
        self.results: ModelResults | None = None

        if isinstance(model_class, str):
            self.model_class: type[Linker | Deduper] = _MODEL_CLASSES[model_class]
        else:
            self.model_class = model_class
        self.model_instance = self.model_class(settings=model_settings)

        self.model_type: ModelType = (
            ModelType.LINKER
            if issubclass(self.model_class, Linker)
            else ModelType.DEDUPER
        )

        if isinstance(model_settings, dict):
            SettingsClass = self.model_instance.__annotations__["settings"]
            self.model_settings = SettingsClass(**model_settings)
        else:
            self.model_settings = model_settings

    @property
    def config(self) -> ModelConfig:
        """Generate config DTO from Model."""
        return ModelConfig(
            type=self.model_type,
            model_class=self.model_class.__name__,
            model_settings=self.model_settings.model_dump_json(),
            left_query=self.left_query.config,
            right_query=self.right_query.config if self.right_query else None,
        )

    @property
    def sources(self) -> set[SourceResolutionName]:
        """Set of source names upstream of this node."""
        left_input = self.dag.nodes[self.left_query.config.point_of_truth]
        model_sources = left_input.sources
        if self.right_query:
            right_input = self.dag.nodes[self.right_query.config.point_of_truth]
            model_sources.update(right_input.sources)

        return model_sources

    @post_run
    def to_resolution(self) -> Resolution:
        """Convert to Resolution for API calls."""
        return Resolution(
            description=self.description,
            truth=self._truth,
            resolution_type=ResolutionType.MODEL,
            config=self.config,
            fingerprint=hash_model_results(self.results.probabilities),
        )

    @classmethod
    def from_resolution(
        cls,
        resolution: Resolution,
        resolution_name: str,
        dag: DAG,
    ) -> "Model":
        """Reconstruct from Resolution."""
        if resolution.resolution_type != ResolutionType.MODEL:
            raise ValueError("Resolution must be of type 'model'")

        return cls(
            dag=dag,
            name=ModelResolutionName(resolution_name),
            description=resolution.description,
            model_class=resolution.config.model_class,
            model_settings=json.loads(resolution.config.model_settings),
            left_query=Query.from_config(resolution.config.left_query, dag=dag),
            right_query=Query.from_config(resolution.config.right_query, dag=dag)
            if resolution.config.right_query
            else None,
            truth=threshold_int_to_float(resolution.truth),
        )

    @property
    def resolution_path(self) -> ModelResolutionPath:
        """Returns the model resolution path."""
        return ModelResolutionPath(
            collection=self.dag.name,
            run=self.dag.run,
            name=self.name,
        )

    @property
    def truth(self) -> float | None:
        """Returns the truth threshold for the model as a float."""
        if self._truth is not None:
            return threshold_int_to_float(self._truth)
        return None

    @truth.setter
    def truth(self, truth: float) -> None:
        """Set the truth threshold for the model."""
        self._truth = threshold_float_to_int(truth)

    def delete(self, certain: bool = False) -> bool:
        """Delete the model from the database."""
        logger.info(f"Deleting {self.name}")
        result = _handler.delete_resolution(path=self.resolution_path, certain=certain)
        return result.success

    @profile_time(attr="name")
    def compute_probabilities(
        self, left_df: DataFrame, right_df: DataFrame | None = None
    ) -> DataFrame:
        """Run model instance against data."""
        if self.config.type == ModelType.LINKER:
            self.model_instance.prepare(left_df, right_df)
            probabilities = self.model_instance.link(left=left_df, right=right_df)
        else:
            self.model_instance.prepare(left_df)
            probabilities = self.model_instance.dedupe(data=left_df)

        return probabilities

    def run(
        self,
        left_data: DataFrame | None = None,
        right_data: DataFrame | None = None,
        for_validation: bool = False,
    ) -> ModelResults:
        """Execute the model pipeline and return results.

        Args:
            left_data (optional): Pre-fetched query data to deduplicate if the model is
                a deduper, or link on the left if the model is a linker.
            right_data (optional): Pre-fetched query data to link on the right, if the
                model is a linker. If the model is a deduper, this argument is ignored.
            for_validation: Whether to download and store extra data to explore and
                score results.
        """
        log_prefix = f"Run {self.name}"
        logger.info("Executing left query", prefix=log_prefix)

        left_df = (
            left_data
            if left_data is not None
            else self.left_query.data(return_leaf_id=for_validation)
        )
        right_df = None

        if self.config.type == ModelType.LINKER:
            logger.info("Executing right query", prefix=log_prefix)
            right_df = (
                right_data
                if right_data is not None
                else self.right_query.data(return_leaf_id=for_validation)
            )

        logger.info("Running model logic", prefix=log_prefix)
        probabilities = self.compute_probabilities(left_df, right_df)

        if for_validation:
            self.results = ModelResults(
                probabilities=probabilities,
                left_root_leaf=self.left_query.leaf_id,
                right_root_leaf=self.right_query.leaf_id
                if right_df is not None
                else None,
            )
        else:
            self.results = ModelResults(probabilities=probabilities)

        return self.results

    @post_run
    @profile_time(attr="name")
    def sync(self) -> None:
        """Send the model config and results to the server.

        Not resistant to race conditions: only one client should call sync at a time.
        """
        log_prefix = f"Sync {self.name}"
        resolution = self.to_resolution()
        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
            logger.info("Found existing resolution", prefix=log_prefix)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None

        if existing_resolution:
            if (existing_resolution.fingerprint == resolution.fingerprint) and (
                existing_resolution.config.parents == resolution.config.parents
            ):
                logger.info("Updating existing resolution", prefix=log_prefix)
                # Assumes that resolution hasn't been deleted or made incompatible
                # Else, server will error
                _handler.update_resolution(
                    resolution=resolution, path=self.resolution_path
                )
            else:
                logger.info(
                    "Update not possible. Deleting existing resolution",
                    prefix=log_prefix,
                )
                # Assumes that resolution hasn't been deleted, else server will error
                _handler.delete_resolution(path=self.resolution_path, certain=True)
                existing_resolution = None

        if not existing_resolution:
            logger.info("Creating new resolution", prefix=log_prefix)
            # Assumes that resolution hasn't since been re-created.
            # Else, server will error
            _handler.create_resolution(resolution=resolution, path=self.resolution_path)
            logger.info("Setting data for new resolution", prefix=log_prefix)
            # Assumes resolution has not been deleted or made incompatible
            _handler.set_data(
                path=self.resolution_path, data=self.results.probabilities
            )

    def download_results(self) -> ModelResults:
        """Retrieve results associated with the model from the database."""
        results = _handler.get_results(name=self.name)
        return ModelResults(probabilities=results)

    def query(self, *sources: Source, **kwargs: Any) -> Query:
        """Generate a query for this model."""
        return Query(*sources, **kwargs, model=self, dag=self.dag)

    def clear_data(self) -> None:
        """Deletes data computed for node."""
        self.results = None
