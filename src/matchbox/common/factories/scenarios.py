"""Scenario factories for creating TestkitDAG scenarios."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, Literal

import pyarrow as pa
from polars.testing import assert_frame_equal
from sqlalchemy import Engine

from matchbox.client.queries import Query
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.entities import FeatureConfig, SuffixRule
from matchbox.common.factories.models import query_to_model_factory
from matchbox.common.factories.sources import (
    SourceTestkitParameters,
    linked_sources_factory,
)
from matchbox.server.base import MatchboxDBAdapter, MatchboxSnapshot

# Type definitions
ScenarioBuilder = Callable[..., TestkitDAG]

SCENARIO_REGISTRY: dict[str, ScenarioBuilder] = {}

# Cache for database snapshots
_DATABASE_SNAPSHOTS_CACHE: dict[str, tuple[TestkitDAG, MatchboxSnapshot]] = {}


def register_scenario(name: str) -> Callable[[ScenarioBuilder], ScenarioBuilder]:
    """Decorator to register a new scenario builder function."""

    def decorator(func: ScenarioBuilder) -> ScenarioBuilder:
        SCENARIO_REGISTRY[name] = func
        return func

    return decorator


def _generate_cache_key(
    backend: MatchboxDBAdapter,
    scenario_type: str,
    warehouse: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> str:
    """Generate a unique hash based on input parameters."""
    if scenario_type not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return (
        f"{warehouse.url}_{backend.__class__.__name__}_"
        f"{scenario_type}_{n_entities}_{seed}"
    )


def _testkitdag_to_location(client: Engine, dag: TestkitDAG) -> None:
    """Upload a TestkitDAG to a location warehouse.

    * Writes all data to the location warehouse, replacing existing data
    * Updates the client of all sources in the DAG
    """
    for source_testkit in dag.sources.values():
        source_testkit.write_to_location(set_client=client)


# Scenario builders


@register_scenario("bare")
def create_bare_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> TestkitDAG:
    """Create a bare TestkitDAG scenario."""
    dag_testkit = TestkitDAG()

    # Create collection and run
    backend.create_collection(name=dag_testkit.dag.name)
    dag_testkit.dag.run = backend.create_run(collection=dag_testkit.dag.name).run_id

    # Create linked sources
    linked = linked_sources_factory(
        n_true_entities=n_entities,
        seed=seed,
        engine=warehouse_engine,
        dag=dag_testkit.dag,
    )
    dag_testkit.add_linked_sources(linked)

    # Write sources to warehouse
    _testkitdag_to_location(warehouse_engine, dag_testkit)

    return dag_testkit


@register_scenario("index")
def create_index_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> TestkitDAG:
    """Create an index TestkitDAG scenario."""
    # First create the bare scenario
    dag_testkit = create_bare_scenario(
        backend, warehouse_engine, n_entities, seed, **kwargs
    )

    # Index sources in backend
    for source_testkit in dag_testkit.sources.values():
        backend.create_resolution(
            resolution=source_testkit.source.to_resolution(),
            path=source_testkit.resolution_path,
        )
        backend.insert_source_data(
            path=source_testkit.resolution_path,
            data_hashes=source_testkit.data_hashes,
        )

    return dag_testkit


@register_scenario("dedupe")
def create_dedupe_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> TestkitDAG:
    """Create a dedupe TestkitDAG scenario."""
    # First create the index scenario
    dag_testkit = create_index_scenario(
        backend, warehouse_engine, n_entities, seed, **kwargs
    )

    # Get the LinkedSourcesTestkit using one of the sources
    linked = dag_testkit.source_to_linked["crn"]

    # Create and add deduplication models
    for testkit in dag_testkit.sources.values():
        name = f"naive_test_{testkit.name}"

        # Query the raw data
        source_data = backend.query(source=testkit.resolution_path)

        # Build model testkit using query data
        model_testkit = query_to_model_factory(
            left_query=Query(testkit.source, dag=dag_testkit.dag),
            left_data=source_data,
            left_keys={testkit.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=name,
            description=f"Deduplication of {testkit.name}",
            prob_range=(1.0, 1.0),
            seed=seed,
        )

        # Add to backend and DAG
        backend.create_resolution(
            resolution=model_testkit.model.to_resolution(),
            path=model_testkit.resolution_path,
        )
        backend.insert_model_data(
            path=model_testkit.resolution_path,
            results=model_testkit.probabilities.to_arrow(),
        )
        dag_testkit.add_model(model_testkit)

    return dag_testkit


@register_scenario("probabilistic_dedupe")
def create_probabilistic_dedupe_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> TestkitDAG:
    """Create a probabilistic dedupe TestkitDAG scenario."""
    # First create the index scenario
    dag_testkit = create_index_scenario(
        backend, warehouse_engine, n_entities, seed, **kwargs
    )

    # Get the LinkedSourcesTestkit using one of the sources
    linked = dag_testkit.source_to_linked["crn"]

    # Create and add deduplication models
    for testkit in dag_testkit.sources.values():
        name = f"probabilistic_test_{testkit.name}"

        # Query the raw data
        source_data = backend.query(source=testkit.resolution_path)

        # Build model testkit using query data
        model_testkit = query_to_model_factory(
            left_query=Query(testkit.source, dag=dag_testkit.dag),
            left_data=source_data,
            left_keys={testkit.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=name,
            description=f"Probabilistic deduplication of {testkit.name}",
            prob_range=(0.5, 0.99),
            seed=seed,
        )
        model_testkit.threshold = 50

        # Add to backend and DAG
        backend.create_resolution(
            resolution=model_testkit.model.to_resolution(),
            path=model_testkit.resolution_path,
        )
        backend.insert_model_data(
            path=model_testkit.resolution_path,
            results=model_testkit.probabilities.to_arrow(),
        )
        backend.set_model_truth(path=model_testkit.resolution_path, truth=50)
        dag_testkit.add_model(model_testkit)

    return dag_testkit


@register_scenario("link")
def create_link_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> TestkitDAG:
    """Create a link TestkitDAG scenario."""
    # First create the dedupe scenario
    dag_testkit = create_dedupe_scenario(
        backend, warehouse_engine, n_entities, seed, **kwargs
    )

    # Get the LinkedSourcesTestkit using one of the sources
    linked = dag_testkit.source_to_linked["crn"]

    # Extract models for linking
    crn_model = dag_testkit.models["naive_test_crn"]
    duns_model = dag_testkit.models["naive_test_duns"]
    cdms_model = dag_testkit.models["naive_test_cdms"]

    # Query data for each resolution
    crn_data = backend.query(
        source=dag_testkit.sources["crn"].resolution_path,
        point_of_truth=crn_model.resolution_path,
    )
    duns_data = backend.query(
        source=dag_testkit.sources["duns"].resolution_path,
        point_of_truth=duns_model.resolution_path,
    )
    cdms_data = backend.query(
        source=dag_testkit.sources["cdms"].resolution_path,
        point_of_truth=cdms_model.resolution_path,
    )

    # Create CRN-DUNS link
    crn_duns_name = "deterministic_naive_test_crn_naive_test_duns"
    crn_duns_model = query_to_model_factory(
        left_query=Query(
            dag_testkit.sources["crn"].source,
            model=crn_model.model,
            dag=dag_testkit.dag,
        ),
        left_data=crn_data,
        left_keys={"crn": "key"},
        right_query=Query(
            dag_testkit.sources["duns"], model=duns_model.model, dag=dag_testkit.dag
        ),
        right_data=duns_data,
        right_keys={"duns": "key"},
        true_entities=tuple(linked.true_entities),
        name=crn_duns_name,
        description="Link between CRN and DUNS",
        prob_range=(1.0, 1.0),
        seed=seed,
    )

    # Add to backend and DAG
    backend.create_resolution(
        resolution=crn_duns_model.model.to_resolution(),
        path=crn_duns_model.resolution_path,
    )
    backend.insert_model_data(
        path=crn_duns_model.resolution_path,
        results=crn_duns_model.probabilities.to_arrow(),
    )
    dag_testkit.add_model(crn_duns_model)

    # Create CRN-CDMS link
    crn_cdms_name = "probabilistic_naive_test_crn_naive_test_cdms"
    crn_cdms_model = query_to_model_factory(
        left_query=Query(
            dag_testkit.sources["crn"].source,
            model=crn_model.model,
            dag=dag_testkit.dag,
        ),
        left_data=crn_data,
        left_keys={"crn": "key"},
        right_query=Query(
            dag_testkit.sources["cdms"].source,
            model=cdms_model.model,
            dag=dag_testkit.dag,
        ),
        right_data=cdms_data,
        right_keys={"cdms": "key"},
        true_entities=tuple(linked.true_entities),
        name=crn_cdms_name,
        description="Link between CRN and CDMS",
        seed=seed,
    )

    # Add to backend and DAG
    backend.create_resolution(
        path=crn_cdms_model.resolution_path,
        resolution=crn_cdms_model.model.to_resolution(),
    )
    backend.insert_model_data(
        path=crn_cdms_model.resolution_path,
        results=crn_cdms_model.probabilities.to_arrow(),
    )
    backend.set_model_truth(path=crn_cdms_model.resolution_path, truth=75)
    dag_testkit.add_model(crn_cdms_model)

    # Create final join
    # Query the previous link's results
    crn_cdms_data_crn_only = backend.query(
        source=dag_testkit.sources["crn"].resolution_path,
        point_of_truth=crn_cdms_model.resolution_path,
    ).rename_columns(["id", "keys_crn"])
    crn_cdms_data_cdms_only = backend.query(
        source=dag_testkit.sources["cdms"].resolution_path,
        point_of_truth=crn_cdms_model.resolution_path,
    ).rename_columns(["id", "keys_cdms"])
    crn_cdms_data = pa.concat_tables(
        [crn_cdms_data_crn_only, crn_cdms_data_cdms_only],
        promote_options="default",
    ).combine_chunks()

    duns_data_linked = backend.query(
        source=dag_testkit.sources["duns"].resolution_path,
        point_of_truth=duns_model.resolution_path,
    )

    final_join_name = "final_join"
    final_join_model = query_to_model_factory(
        left_query=Query(
            dag_testkit.sources["crn"].source,
            dag_testkit.sources["cdms"].source,
            model=crn_cdms_model.model,
            dag=dag_testkit.dag,
        ),
        left_data=crn_cdms_data,
        left_keys={"crn": "keys_crn", "cdms": "keys_cdms"},
        right_query=Query(
            dag_testkit.sources["duns"].source,
            model=duns_model.model,
            dag=dag_testkit.dag,
        ),
        right_data=duns_data_linked,
        right_keys={"duns": "key"},
        true_entities=tuple(linked.true_entities),
        name=final_join_name,
        description="Final join of all entities",
        seed=seed,
    )

    # Add to backend and DAG
    backend.create_resolution(
        resolution=final_join_model.model.to_resolution(),
        path=final_join_model.resolution_path,
    )
    backend.insert_model_data(
        path=final_join_model.resolution_path,
        results=final_join_model.probabilities.to_arrow(),
    )
    dag_testkit.add_model(final_join_model)

    return dag_testkit


@register_scenario("alt_dedupe")
def create_alt_dedupe_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> TestkitDAG:
    """Create a TestkitDAG scenario with two alternative dedupers."""
    dag_testkit = TestkitDAG()

    # Create collection and run
    backend.create_collection(name=dag_testkit.dag.name)
    dag_testkit.dag.run = backend.create_run(collection=dag_testkit.dag.name).run_id

    # Create linked sources
    company_name_feature = FeatureConfig(
        name="company_name", base_generator="company"
    ).add_variations(SuffixRule(suffix=" UK"))

    foo_a_tkit_source = SourceTestkitParameters(
        name="foo_a",
        engine=warehouse_engine,
        features=(company_name_feature,),
        drop_base=False,
        n_true_entities=n_entities,
        repetition=1,
    )

    linked = linked_sources_factory(
        source_parameters=(foo_a_tkit_source,), dag=dag_testkit.dag
    )
    dag_testkit.add_linked_sources(linked)

    # Write sources to warehouse
    _testkitdag_to_location(warehouse_engine, dag_testkit)

    # Index sources in backend
    for source_testkit in dag_testkit.sources.values():
        backend.create_resolution(
            resolution=source_testkit.source.to_resolution(),
            path=source_testkit.resolution_path,
        )
        backend.insert_source_data(
            path=source_testkit.resolution_path,
            data_hashes=source_testkit.data_hashes,
        )

    # Create and add deduplication models
    for testkit in dag_testkit.sources.values():
        model_name1 = f"dedupe_{testkit.name}"
        model_name2 = f"dedupe2_{testkit.name}"

        # Query the raw data
        source_data = backend.query(source=testkit.resolution_path)

        # Build model testkit using query data
        model_testkit1 = query_to_model_factory(
            left_query=Query(testkit.source, dag=dag_testkit.dag),
            left_data=source_data,
            left_keys={testkit.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=model_name1,
            description=f"Deduplication of {testkit.name}",
            prob_range=(0.5, 1.0),
            seed=seed,
        )

        model_testkit2 = query_to_model_factory(
            left_query=Query(testkit.source, dag=dag_testkit.dag),
            left_data=source_data,
            left_keys={testkit.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=model_name2,
            description=f"Deduplication of {testkit.name}",
            prob_range=(0.5, 1.0),
            seed=seed,
        )

        assert len(model_testkit1.probabilities) > 0
        assert_frame_equal(model_testkit1.probabilities, model_testkit2.probabilities)

        for model, threshold in ((model_testkit1, 50), (model_testkit2, 75)):
            model.threshold = threshold

            # Add both models to backend and DAG
            backend.create_resolution(
                path=model.resolution_path, resolution=model.model.to_resolution()
            )
            backend.insert_model_data(
                path=model.resolution_path, results=model.probabilities.to_arrow()
            )
            backend.set_model_truth(path=model.resolution_path, truth=threshold)

            # Add to DAG
            dag_testkit.add_model(model)

    return dag_testkit


@register_scenario("convergent")
def create_convergent_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> TestkitDAG:
    """Create a convergent TestkitDAG scenario.

    This is where two SourceConfigs index almost identically. TestkitDAG contains two
    indexed sources with repetition, and two naive dedupe models that haven't yet
    had their results inserted.
    """
    dag_testkit = TestkitDAG()

    # Create collection and run
    backend.create_collection(name=dag_testkit.dag.name)
    dag_testkit.dag.run = backend.create_run(collection=dag_testkit.dag.name).run_id

    # Create linked sources
    company_name_feature = FeatureConfig(
        name="company_name", base_generator="company"
    ).add_variations(SuffixRule(suffix=" UK"))

    foo_a_tkit_source = SourceTestkitParameters(
        name="foo_a",
        engine=warehouse_engine,
        features=(company_name_feature,),
        drop_base=False,
        n_true_entities=n_entities,
        repetition=1,
    )

    linked = linked_sources_factory(
        source_parameters=(
            foo_a_tkit_source,
            foo_a_tkit_source.model_copy(update={"name": "foo_b"}),
        ),
        dag=dag_testkit.dag,
    )

    dag_testkit.add_linked_sources(linked)

    # Write sources to warehouse
    _testkitdag_to_location(warehouse_engine, dag_testkit)

    # Index sources in backend
    for source_testkit in dag_testkit.sources.values():
        backend.create_resolution(
            resolution=source_testkit.source.to_resolution(),
            path=source_testkit.resolution_path,
        )
        backend.insert_source_data(
            path=source_testkit.resolution_path,
            data_hashes=source_testkit.data_hashes,
        )

        # Query the raw data
        source_query = backend.query(source=source_testkit.resolution_path)

        # Build model testkit using query data
        model_testkit = query_to_model_factory(
            left_query=Query(source_testkit.source, dag=dag_testkit.dag),
            left_data=source_query,
            left_keys={source_testkit.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=f"naive_test_{source_testkit.name}",
            description=f"Deduplication of {source_testkit.name}",
            prob_range=(1.0, 1.0),
            seed=seed,
        )

        assert len(model_testkit.probabilities) > 0

        # Add to DAG
        dag_testkit.add_model(model_testkit)

    return dag_testkit


@contextmanager
def setup_scenario(
    backend: MatchboxDBAdapter,
    scenario_type: Literal[
        "bare",
        "index",
        "dedupe",
        "link",
        "probabilistic_dedupe",
        "alt_dedupe",
        "convergent",
    ],
    warehouse: Engine,
    n_entities: int = 10,
    seed: int = 42,
    **kwargs: dict[str, Any],
) -> Generator[TestkitDAG, None, None]:
    """Context manager for creating TestkitDAG scenarios."""
    if scenario_type not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    # Generate cache key for backend snapshot
    cache_key = _generate_cache_key(
        backend, scenario_type, warehouse, n_entities, seed, **kwargs
    )

    # Check if we have a backend snapshot cached
    if cache_key in _DATABASE_SNAPSHOTS_CACHE:
        # Load cached snapshot and DAG
        dag_testkit, snapshot = _DATABASE_SNAPSHOTS_CACHE[cache_key]
        # Because the location client can't be deep-copied, this will reuse an
        # old engine, with the same URL as `warehouse`, but a different object
        # Thus, in _testkitdag_to_location we need to overwrite all testkits with
        # our new warehouse object
        dag_testkit = dag_testkit.model_copy(deep=True)

        # Restore backend and write sources to warehouse
        backend.restore(snapshot=snapshot)
        _testkitdag_to_location(warehouse, dag_testkit)
    else:
        # Create new TestkitDAG with proper backend integration
        scenario_builder = SCENARIO_REGISTRY[scenario_type]
        dag_testkit = scenario_builder(backend, warehouse, n_entities, seed, **kwargs)

        # Cache the snapshot and DAG
        _DATABASE_SNAPSHOTS_CACHE[cache_key] = (dag_testkit, backend.dump())

    try:
        yield dag_testkit
    finally:
        backend.clear(certain=True)
