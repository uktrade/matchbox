"""DAG container for testkits."""

from pydantic import BaseModel, ConfigDict

from matchbox.client.dags import DAG
from matchbox.common.factories.models import ModelTestkit
from matchbox.common.factories.sources import LinkedSourcesTestkit, SourceTestkit
from matchbox.common.graph import (
    ModelResolutionName,
    ResolutionName,
    SourceResolutionName,
)


class TestkitDAG(BaseModel):
    """Simple DAG container for testkits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dag: DAG = DAG("collection")

    sources: dict[SourceResolutionName, SourceTestkit] = {}
    linked: dict[str, LinkedSourcesTestkit] = {}
    source_to_linked: dict[SourceResolutionName, str] = {}
    models: dict[ModelResolutionName, ModelTestkit] = {}

    # Dependency graph tracking
    adjacency: dict[ResolutionName, set[ResolutionName]] = {}  # -> direct dependencies
    root_source_names: dict[ResolutionName, set[SourceResolutionName]] = {}

    # Keep track of all used names to ensure uniqueness
    _all_names: set[str] = set()

    def _validate_unique_name(self, name: ResolutionName) -> str:
        """Ensure a name is unique across all testkits."""
        if name in self._all_names:
            raise ValueError(
                f"Name collision detected: '{name}' is already used in this DAG"
            )
        self._all_names.add(name)
        return name

    def _validate_dependencies(
        self, left_res: ResolutionName, right_res: ResolutionName | None
    ) -> bool:
        """Ensure dependencies are valid."""
        valid_deps = set(self.sources.keys()) | set(self.models.keys())
        dependencies = {dep for dep in [left_res, right_res] if dep is not None}

        missing_deps = dependencies - valid_deps
        if missing_deps:
            raise ValueError(
                f"Missing dependencies for model: {missing_deps}. "
                "Ensure all dependencies are added before models."
            )
        return True

    def _update_root_source_names(self, name: ModelResolutionName):
        """Update root source names for a model."""
        for dep_name in self.adjacency[name]:
            if dep_name in self.sources:
                # If dependency is a source, add it directly
                self.root_source_names[name].add(dep_name)
            elif dep_name in self.models:
                # If dependency is a model, add all its root sources
                self.root_source_names[name].update(self.root_source_names[dep_name])

    def add_source(self, testkit: SourceTestkit | LinkedSourcesTestkit):
        """Add a source testkit to the DAG."""
        if isinstance(testkit, LinkedSourcesTestkit):
            # Create linked key as "linked_" + concatenated source resolution names
            source_names = sorted(testkit.sources.keys())
            linked_key = f"linked_{'_'.join(source_names)}"

            # Store LinkedSourcesTestkit in the linked dict with the new key
            self.linked[linked_key] = testkit

            # Add each source and track its association with the linked key
            for source_name, source_testkit in testkit.sources.items():
                self._validate_unique_name(source_name)
                self.sources[source_name] = source_testkit
                self.source_to_linked[source_name] = linked_key
        else:
            source_name = self._validate_unique_name(testkit.name)
            self.sources[source_name] = testkit

    def add_model(self, testkit: ModelTestkit):
        """Add a model testkit to the DAG."""
        name = self._validate_unique_name(testkit.name)
        self.models[name] = testkit
        self.adjacency[name] = set()
        self.root_source_names[name] = set()

        # Validate dependencies
        left_res = testkit.model.config.left_query.point_of_truth
        right_res = (
            testkit.model.config.right_query.point_of_truth
            if testkit.model.config.right_query
            else None
        )
        self._validate_dependencies(left_res, right_res)

        # Track dependencies based on left/right resolution
        if left_res:
            self.adjacency[name].add(left_res)

        if right_res and right_res != left_res:
            self.adjacency[name].add(right_res)

        # Update root sources
        self._update_root_source_names(name)

    def get_sources_for_model(
        self, name: ModelResolutionName
    ) -> dict[str | None, set[SourceResolutionName]]:
        """Find the LinkedSourcesTestkit keys and specific source names for a model.

        Args:
            name: The name of the model to analyze

        Returns:
            A dictionary mapping:
            - LinkedSourcesTestkit keys (or None) to sets of model's source names
        """
        if name not in self.models:
            return {None: set()}

        root_sources = self.root_source_names.get(name, set())
        if not root_sources:
            return {None: set()}

        result = {}
        for name in root_sources:
            # linked_key could be None
            linked_key = self.source_to_linked.get(name)
            if linked_key not in result:
                result[linked_key] = set()
            result[linked_key].add(name)

        return result
