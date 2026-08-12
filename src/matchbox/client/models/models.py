"""Functions and classes to define, run and register models."""

import inspect
from typing import TYPE_CHECKING, Any, ClassVar, ParamSpec, TypeVar, overload

import polars as pl
import pyarrow as pa
from polars import DataFrame

from matchbox.client.models import dedupers, linkers
from matchbox.client.models.dedupers.base import Deduper, DeduperSettings
from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.client.queries import Query
from matchbox.client.results import normalise_model_scores
from matchbox.client.steps import StepABC, post_run
from matchbox.common.arrow import SCHEMA_MODEL_EDGES
from matchbox.common.dtos import (
    ModelConfig,
    ModelStepName,
    ModelStepPath,
    ModelType,
    SourceStepName,
    Step,
    StepType,
)
from matchbox.common.hash import hash_arrow_table
from matchbox.common.logging import logger
from matchbox.common.stats import DAGStats

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
    from matchbox.client.resolvers import Resolver
    from matchbox.client.resolvers.base import ResolverMethod, ResolverSettings
else:
    DAG = Any

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


class Model(StepABC):
    """Unified model class for both linking and deduping operations."""

    _local_data_schema: ClassVar[pa.Schema] = SCHEMA_MODEL_EDGES

    @overload
    def __init__(
        self,
        name: str,
        dag: DAG,
        model_class: type[Deduper],
        model_settings: DeduperSettings | dict,
        left_query: Query,
        right_query: None = None,
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
        description: str | None = None,
    ):
        """Create a new model instance.

        Args:
            dag: DAG containing this model.
            name: Unique name for the model.
            model_class: Class of Linker or Deduper, or its name.
            model_settings: Appropriate settings object to pass to model class.
            left_query: The query that will get the data to deduplicate, or the data to
                link on the left.
            right_query: The query that will get the data to link on the right.
            description: Optional description of the model.
        """
        super().__init__(dag=dag, name=name, description=description)

        self.left_query = left_query
        self.right_query = right_query

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
    def results(self) -> pl.DataFrame | None:
        """The locally computed model scores. Alias for local_data."""
        return self._local_data

    @results.setter
    def results(self, value: pl.DataFrame | None) -> None:
        self._local_data = value

    @property
    def config(self) -> ModelConfig:
        """Generate config DTO from Model."""
        return ModelConfig(
            type=self.model_type,
            model_class=self.model_class.__name__,
            model_settings=self.model_settings.model_dump(mode="json"),
            left_query=self.left_query.config,
            right_query=self.right_query.config if self.right_query else None,
        )

    @property
    def sources(self) -> set[SourceStepName]:
        """Set of source names upstream of this node."""
        left_input = self.dag.nodes[self.left_query.config.resolves_from]
        model_sources = left_input.sources
        if self.right_query:
            right_input = self.dag.nodes[self.right_query.config.resolves_from]
            model_sources.update(right_input.sources)
        return model_sources

    @post_run
    def _fingerprint(self) -> bytes:
        """Compute a content hash invariant to left/right ID order."""
        return hash_arrow_table(
            self._local_data.to_arrow(), as_sorted_list=["left_id", "right_id"]
        )

    @post_run
    def to_dto(self) -> Step:
        """Convert to Step DTO for API calls."""
        return Step(
            description=self.description,
            step_type=StepType.MODEL,
            config=self.config,
            fingerprint=self._fingerprint(),
        )

    @classmethod
    def from_dto(
        cls,
        step: Step,
        step_name: str,
        dag: DAG,
        **kwargs: Any,
    ) -> "Model":
        """Reconstruct from Step DTO."""
        if step.step_type != StepType.MODEL:
            raise ValueError("Step must be of type 'model'")

        return cls(
            dag=dag,
            name=ModelStepName(step_name),
            description=step.description,
            model_class=step.config.model_class,
            model_settings=step.config.model_settings,
            left_query=Query.from_config(step.config.left_query, dag=dag),
            right_query=Query.from_config(step.config.right_query, dag=dag)
            if step.config.right_query
            else None,
        )

    @property
    def path(self) -> ModelStepPath:
        """Return the model step path."""
        return ModelStepPath(
            collection=self.dag.name,
            run=self.dag.run,
            name=self.name,
        )

    @DAGStats.time("compute")
    def compute_scores(
        self, left_df: DataFrame, right_df: DataFrame | None = None
    ) -> DataFrame:
        """Run model instance against data."""
        if self.config.type == ModelType.LINKER:
            self.model_instance.prepare(left_df, right_df)
            scores = self.model_instance.link(left=left_df, right=right_df)
        else:
            self.model_instance.prepare(left_df)
            scores = self.model_instance.dedupe(data=left_df)
        return scores

    def run(
        self,
        left_data: DataFrame | None = None,
        right_data: DataFrame | None = None,
        low_memory: bool = False,
    ) -> pl.DataFrame:
        """Execute the model pipeline and return results.

        Args:
            left_data (optional): Pre-fetched query data to deduplicate if the model is
                a deduper, or link on the left if the model is a linker.
            right_data (optional): Pre-fetched query data to link on the right, if the
                model is a linker. If the model is a deduper, this argument is ignored.
            low_memory: If True, it will not download data from the server to support
                evaluation.
        """
        log_prefix = f"Run {self.name}"
        logger.info("Executing left query", prefix=log_prefix)

        with DAGStats.time("query", name=self.name, stats=self._stats):
            left_df = (
                left_data
                if left_data is not None
                else self.left_query.data(cache_leaf_ids=(not low_memory))
            )
            right_df = None

            if self.config.type == ModelType.LINKER:
                logger.info("Executing right query", prefix=log_prefix)
                right_df = (
                    right_data
                    if right_data is not None
                    else self.right_query.data(cache_leaf_ids=(not low_memory))
                )

        logger.info("Running model logic", prefix=log_prefix)
        scores = self.compute_scores(left_df, right_df)
        self._local_data = normalise_model_scores(scores)

        return self._local_data

    def resolver(
        self,
        *other_models: "Model",
        name: str,
        resolver_class: type["ResolverMethod"] | str,
        resolver_settings: "ResolverSettings | dict[str, Any] | None" = None,
        description: str | None = None,
    ) -> "Resolver":
        """Create a resolver rooted at this model and add it to the DAG."""
        return self.dag.resolver(
            name=name,
            inputs=[self, *other_models],
            resolver_class=resolver_class,
            resolver_settings=resolver_settings if resolver_settings else {},
            description=description,
        )

    def clear_data(self) -> None:
        """Suppress data clearing for models."""
        # TODO make DAG smart enough to know when to call this
        # without affecting downstream resolvers
