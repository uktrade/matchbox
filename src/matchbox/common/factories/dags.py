"""Simplified TestkitDAG that's just a registry of test data."""

from pydantic import BaseModel, ConfigDict, Field

from matchbox.client.dags import DAG
from matchbox.common.dtos import (
    ModelResolutionName,
    SourceResolutionName,
)
from matchbox.common.factories.models import ModelTestkit
from matchbox.common.factories.sources import LinkedSourcesTestkit, SourceTestkit


def _default_dag() -> DAG:
    """Create a default empty DAG."""
    dag = DAG(name="collection")
    dag.run = 1
    return dag


class TestkitDAG(BaseModel):
    """DAG test wrapper that's just a registry of test data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The real DAG that handles all logic
    dag: DAG = Field(default_factory=_default_dag)

    # Just registries of test data - no complex logic
    sources: dict[SourceResolutionName, SourceTestkit] = {}
    models: dict[ModelResolutionName, ModelTestkit] = {}
    linked: dict[str, LinkedSourcesTestkit] = {}
    source_to_linked: dict[str, LinkedSourcesTestkit] = {}

    def add_linked_sources(self, testkit: LinkedSourcesTestkit) -> None:
        """Add system of linked sources to the real DAG and register test data."""
        linked_key = f"linked_{'_'.join(sorted(testkit.sources.keys()))}"
        self.linked[linked_key] = testkit

        for source_testkit in testkit.sources.values():
            self.source_to_linked[source_testkit.name] = testkit
            self.add_source(source_testkit)

    def add_source(self, testkit: SourceTestkit) -> None:
        """Add source to the real DAG and register test data."""
        self.dag._add_step(testkit.source)
        self.sources[testkit.name] = testkit

    def add_model(self, testkit: ModelTestkit) -> None:
        """Add model to the real DAG and register test data."""
        self.dag._add_step(testkit.model)
        self.models[testkit.name] = testkit
