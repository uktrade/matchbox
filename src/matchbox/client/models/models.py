"""Functions and classes to define, run and register models."""

import inspect
from datetime import datetime
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.models import dedupers, linkers
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.results import Results
from matchbox.common.dtos import ModelConfig, ModelType, Resolution
from matchbox.common.exceptions import MatchboxResolutionNotFoundError
from matchbox.common.graph import ResolutionType
from matchbox.common.logging import logger

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.queries import Query
else:
    DAG = Any
    Query = Any

P = ParamSpec("P")
R = TypeVar("R")


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
        self.last_run: datetime | None = None
        self.dag = dag
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

        if isinstance(model_settings, dict):
            SettingsClass = self.model_instance.__annotations__["settings"]
            model_settings = SettingsClass(**model_settings)

        serialised_settings = model_settings.model_dump_json()

        self.config = ModelConfig(
            type=model_type,
            model_class=model_class.__name__,
            model_settings=serialised_settings,
            left_query=left_query.config,
            right_query=right_query.config if right_query else None,
        )

    @property
    def dependencies(self) -> list[str]:
        """Returns all resolution names this model needs as implied by the queries."""
        if self.right_query:
            return (
                self.left_query.config.dependencies
                + self.right_query.config.dependencies
            )
        return self.left_query.config.dependencies

    @property
    def parents(self) -> list[str]:
        """Returns all points of truth input to this model."""
        if self.right_query:
            return [
                self.left_query.config.point_of_truth,
                self.right_query.config.point_of_truth,
            ]
        return [self.left_query.config.point_of_truth]

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
    def from_resolution(cls, resolution: Resolution, dag: DAG) -> "Model":
        """Reconstruct from Resolution."""
        if resolution.resolution_type != ResolutionType.MODEL:
            raise ValueError("Resolution must be of type 'model'")

        return cls(
            dag=dag,
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
        left_df = self.left_query.run(
            return_leaf_id=for_validation, batch_size=settings.batch_size
        )
        right_df = None

        if self.config.type == ModelType.LINKER:
            right_df = self.right_query.run(
                return_leaf_id=for_validation, batch_size=settings.batch_size
            )

            self.model_instance.prepare(left_df, right_df)
            results = self.model_instance.link(left=left_df, right=right_df)
        else:
            self.model_instance.prepare(left_df)
            results = self.model_instance.dedupe(data=left_df)

        if for_validation:
            self.results = Results(
                probabilities=results,
                left_root_leaf=self.left_query.leaf_id.to_arrow(),
                right_root_leaf=self.right_query.leaf_id.to_arrow()
                if right_df is not None
                else None,
            )
        else:
            self.results = Results(probabilities=results)

        self.last_run = datetime.now()
        return self.results

    def sync(self) -> None:
        """Send the model config, truth and results to the server."""
        resolution = self.to_resolution()
        try:
            existing_resolution = _handler.get_resolution(name=self.name)
        except MatchboxResolutionNotFoundError:
            existing_resolution = None
        # Check if config matches
        if existing_resolution:
            if existing_resolution.config != self.config:
                raise ValueError(
                    f"Resolution {self.name} already exists with different "
                    "configuration. Please delete the existing resolution "
                    "or use a different name. "
                )
            else:
                log_prefix = f"Resolution {self.name}"
                logger.warning("Already exists. Passing.", prefix=log_prefix)
        else:
            _handler.create_resolution(resolution=resolution)

        _handler.set_truth(name=self.name, truth=self._truth)

        if self.results and len(self.results.probabilities):
            _handler.set_data(
                name=self.name,
                data=self.results.probabilities,
                validate_type=ResolutionType.MODEL,
            )

    def download_results(self) -> Results:
        """Retrieve results associated with the model from the database."""
        results = _handler.get_results(name=self.name)
        return Results(probabilities=results, metadata=self.config)

    def query(self, *sources, **kwargs) -> Query:
        """Generate a query for this model."""
        return self.dag.query(*sources, **kwargs, model=self)


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
