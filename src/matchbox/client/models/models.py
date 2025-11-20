"""Functions and classes to define, run and register models."""

import inspect
import json
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.models import dedupers, linkers
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.queries import CacheMode, Query
from matchbox.client.results import ModelResults
from matchbox.common.dtos import (
    ModelConfig,
    ModelResolutionName,
    ModelResolutionPath,
    ModelType,
    Resolution,
    ResolutionType,
    SourceResolutionName,
    UploadStage,
)
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.hash import hash_model_results
from matchbox.common.logging import logger
from matchbox.common.transform import truth_float_to_int, truth_int_to_float

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
        self._truth: int = truth_float_to_int(truth)
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
            truth=truth_int_to_float(resolution.truth),
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
            return truth_int_to_float(self._truth)
        return None

    @truth.setter
    def truth(self, truth: float) -> None:
        """Set the truth threshold for the model."""
        self._truth = truth_float_to_int(truth)

    def delete(self, certain: bool = False) -> bool:
        """Delete the model from the database."""
        logger.info(f"Deleting {self.name}")
        result = _handler.delete_resolution(path=self.resolution_path, certain=certain)
        return result.success

    def run(
        self, for_validation: bool = False, cache_queries: bool = False
    ) -> ModelResults:
        """Execute the model pipeline and return results.

        Args:
            for_validation: Whether to download and store extra data to explore and
                    score results.
            cache_queries: Whether to cache query results on first run and re-use them
                subsequently.
        """
        log_prefix = f"Run {self.name}"
        logger.info("Executing left query", prefix=log_prefix)
        cache_mode = CacheMode.CLEAN if cache_queries else CacheMode.OFF
        left_df = self.left_query.set_cache_mode(cache_mode).run(
            return_leaf_id=for_validation,
            batch_size=settings.batch_size,
            reuse_cache=cache_queries,
        )
        right_df = None

        if self.config.type == ModelType.LINKER:
            logger.info("Executing right query", prefix=log_prefix)
            right_df = self.right_query.set_cache_mode(cache_mode).run(
                return_leaf_id=for_validation,
                batch_size=settings.batch_size,
                reuse_cache=cache_queries,
            )

            self.model_instance.prepare(left_df, right_df)
            results = self.model_instance.link(left=left_df, right=right_df)
        else:
            self.model_instance.prepare(left_df)
            results = self.model_instance.dedupe(data=left_df)

        if for_validation:
            self.results = ModelResults(
                probabilities=results,
                left_root_leaf=self.left_query.leaf_id,
                right_root_leaf=self.right_query.leaf_id
                if right_df is not None
                else None,
            )
        else:
            self.results = ModelResults(probabilities=results)

        return self.results

    @post_run
    def sync(self) -> None:
        """Send the model config and results to the server."""
        log_prefix = f"Sync {self.name}"
        resolution = self.to_resolution()
        try:
            existing_resolution = _handler.get_resolution(path=self.resolution_path)
        except MatchboxResolutionNotFoundError:
            logger.info("Found existing resolution", prefix=log_prefix)
            existing_resolution = None

        if existing_resolution:
            if (existing_resolution.fingerprint == resolution.fingerprint) and (
                existing_resolution.config.parents == resolution.config.parents
            ):
                logger.info("Updating existing resolution", prefix=log_prefix)
                _handler.update_resolution(
                    resolution=resolution, path=self.resolution_path
                )
            else:
                logger.info(
                    "Update not possible. Deleting existing resolution",
                    prefix=log_prefix,
                )
                _handler.delete_resolution(path=self.resolution_path, certain=True)
                existing_resolution = None

        if not existing_resolution:
            logger.info("Creating new resolution", prefix=log_prefix)
            _handler.create_resolution(resolution=resolution, path=self.resolution_path)

        upload_stage = _handler.get_resolution_stage(self.resolution_path)
        if upload_stage == UploadStage.READY:
            logger.info("Setting data for new resolution", prefix=log_prefix)
            _handler.set_data(
                path=self.resolution_path, data=self.results.probabilities
            )

    def download_results(self) -> ModelResults:
        """Retrieve results associated with the model from the database."""
        results = _handler.get_results(name=self.name)
        return ModelResults(probabilities=results, metadata=self.config)

    def query(self, *sources: Source, **kwargs: Any) -> Query:
        """Generate a query for this model."""
        return Query(*sources, **kwargs, model=self, dag=self.dag)
