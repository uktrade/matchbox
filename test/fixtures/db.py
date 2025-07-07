import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal

import boto3
import pyarrow as pa
import pytest
import respx
from httpx import Client
from moto import mock_aws
from pydantic_settings import BaseSettings, SettingsConfigDict
from respx import MockRouter
from sqlalchemy import Engine, create_engine

from matchbox.client._handler import create_client
from matchbox.client._settings import ClientSettings, settings
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.entities import FeatureConfig, SuffixRule
from matchbox.common.factories.models import query_to_model_factory
from matchbox.common.factories.sources import (
    SourceTestkitParameters,
    linked_sources_factory,
)
from matchbox.server.base import (
    MatchboxDatastoreSettings,
    MatchboxDBAdapter,
    MatchboxSnapshot,
)
from matchbox.server.postgresql import MatchboxPostgres, MatchboxPostgresSettings

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = Any


# Database scenario fixtures and helper functions


ScenarioBuilder = Callable[[MatchboxDBAdapter, Engine, int, int], TestkitDAG]

SCENARIO_REGISTRY: dict[str, ScenarioBuilder] = {}


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
) -> str:
    """Generate a unique hash based on input parameters"""
    if scenario_type not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return (
        f"{warehouse.url}_{backend.__class__.__name__}_"
        f"{scenario_type}_{n_entities}_{seed}"
    )


def _testkitdag_to_location(credentials: Engine, dag: TestkitDAG) -> None:
    """Upload a TestkitDAG to a location warehouse.

    * Writes all data to the location warehouse, replacing existing data
    * Updates the credentials of all sources in the DAG
    """
    for source_testkit in dag.sources.values():
        source_testkit.write_to_location(credentials=credentials, set_credentials=True)


@register_scenario("bare")
def create_bare_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create a bare TestkitDAG scenario."""
    dag = TestkitDAG()

    # Create linked sources
    linked = linked_sources_factory(
        n_true_entities=n_entities, seed=seed, engine=warehouse_engine
    )
    dag.add_source(linked)

    # Write sources to warehouse
    _testkitdag_to_location(warehouse_engine, dag)

    return dag


@register_scenario("index")
def create_index_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create an index TestkitDAG scenario."""
    # First create the bare scenario
    dag = create_bare_scenario(backend, warehouse_engine, n_entities, seed)

    # Index sources in backend
    for source_testkit in dag.sources.values():
        backend.index(
            source_config=source_testkit.source_config,
            data_hashes=source_testkit.data_hashes,
        )

    return dag


@register_scenario("dedupe")
def create_dedupe_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create a dedupe TestkitDAG scenario."""
    # First create the index scenario
    dag = create_index_scenario(backend, warehouse_engine, n_entities, seed)

    # Get the linked sources
    linked_key = next(iter(dag.linked.keys()))
    linked = dag.linked[linked_key]

    # Create and add deduplication models
    for testkit in dag.sources.values():
        source = testkit.source_config
        name = f"naive_test.{source.name}"

        # Query the raw data
        source_query = backend.query(source=source.name)

        # Build model testkit using query data
        model_testkit = query_to_model_factory(
            left_resolution=source.name,
            left_query=source_query,
            left_keys={source.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=name,
            description=f"Deduplication of {source.name}",
            prob_range=(1.0, 1.0),
            seed=seed,
        )

        # Add to backend and DAG
        backend.insert_model(model_config=model_testkit.model.model_config)
        backend.set_model_results(name=name, results=model_testkit.probabilities)
        dag.add_model(model_testkit)

    return dag


@register_scenario("probabilistic_dedupe")
def create_probabilistic_dedupe_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create a dedupe TestkitDAG scenario."""
    # First create the index scenario
    dag = create_index_scenario(backend, warehouse_engine, n_entities, seed)

    # Get the linked sources
    linked_key = next(iter(dag.linked.keys()))
    linked = dag.linked[linked_key]

    # Create and add deduplication models
    for testkit in dag.sources.values():
        source = testkit.source_config
        name = f"probabilistic_test.{source.name}"

        # Query the raw data
        source_query = backend.query(source=source.name)

        # Build model testkit using query data
        model_testkit = query_to_model_factory(
            left_resolution=source.name,
            left_query=source_query,
            left_keys={source.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=name,
            description=f"Probabilistic deduplication of {source.name}",
            prob_range=(0.5, 0.99),
            seed=seed,
        )
        model_testkit.threshold = 50

        # Add to backend and DAG
        backend.insert_model(model_config=model_testkit.model.model_config)
        backend.set_model_results(name=name, results=model_testkit.probabilities)
        backend.set_model_truth(name=name, truth=0.5)
        dag.add_model(model_testkit)

    return dag


@register_scenario("link")
def create_link_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create a link TestkitDAG scenario."""
    # First create the dedupe scenario
    dag = create_dedupe_scenario(backend, warehouse_engine, n_entities, seed)

    # Get the linked sources
    linked_key = next(iter(dag.linked.keys()))
    linked = dag.linked[linked_key]

    # Extract models for linking
    crn_model = dag.models["naive_test.crn"]
    duns_model = dag.models["naive_test.duns"]
    cdms_model = dag.models["naive_test.cdms"]

    # Query data for each resolution
    crn_query = backend.query(source="crn", resolution=crn_model.name)
    duns_query = backend.query(source="duns", resolution=duns_model.name)
    cdms_query = backend.query(source="cdms", resolution=cdms_model.name)

    # Create CRN-DUNS link
    crn_duns_name = "deterministic_naive_test.crn_naive_test.duns"
    crn_duns_model = query_to_model_factory(
        left_resolution=crn_model.name,
        left_query=crn_query,
        left_keys={"crn": "key"},
        right_resolution=duns_model.name,
        right_query=duns_query,
        right_keys={"duns": "key"},
        true_entities=tuple(linked.true_entities),
        name=crn_duns_name,
        description="Link between CRN and DUNS",
        prob_range=(1.0, 1.0),
        seed=seed,
    )

    # Add to backend and DAG
    backend.insert_model(model_config=crn_duns_model.model.model_config)
    backend.set_model_results(name=crn_duns_name, results=crn_duns_model.probabilities)
    dag.add_model(crn_duns_model)

    # Create CRN-CDMS link
    crn_cdms_name = "probabilistic_naive_test.crn_naive_test.cdms"
    crn_cdms_model = query_to_model_factory(
        left_resolution=crn_model.name,
        left_query=crn_query,
        left_keys={"crn": "key"},
        right_resolution=cdms_model.name,
        right_query=cdms_query,
        right_keys={"cdms": "key"},
        true_entities=tuple(linked.true_entities),
        name=crn_cdms_name,
        description="Link between CRN and CDMS",
        seed=seed,
    )

    backend.insert_model(model_config=crn_cdms_model.model.model_config)
    backend.set_model_results(name=crn_cdms_name, results=crn_cdms_model.probabilities)
    backend.set_model_truth(name=crn_cdms_name, truth=75)
    dag.add_model(crn_cdms_model)

    # Create final join
    # Query the previous link's results
    crn_cdms_query_crn_only = backend.query(
        source="crn", resolution=crn_cdms_name
    ).rename_columns(["id", "keys_crn"])
    crn_cdms_query_cdms_only = backend.query(
        source="cdms", resolution=crn_cdms_name
    ).rename_columns(["id", "keys_cdms"])
    crn_cdms_query = pa.concat_tables(
        [crn_cdms_query_crn_only, crn_cdms_query_cdms_only],
        promote_options="default",
    ).combine_chunks()

    duns_query_linked = backend.query(source="duns", resolution=duns_model.name)

    final_join_name = "final_join"
    final_join_model = query_to_model_factory(
        left_resolution=crn_cdms_name,
        left_query=crn_cdms_query,
        left_keys={"crn": "keys_crn", "cdms": "keys_cdms"},
        right_resolution=duns_model.name,
        right_query=duns_query_linked,
        right_keys={"duns": "key"},
        true_entities=tuple(linked.true_entities),
        name=final_join_name,
        description="Final join of all entities",
        seed=seed,
    )

    backend.insert_model(model_config=final_join_model.model.model_config)
    backend.set_model_results(
        name=final_join_name, results=final_join_model.probabilities
    )
    dag.add_model(final_join_model)

    return dag


@register_scenario("convergent")
def create_convergent_scenario(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create a convergent TestkitDAG scenario.

    This is where two SourceConfigs index almost identically. TestkitDAG contains two
    indexed sources with repetition, and two naive dedupe models that haven't yet
    had their results inserted.
    """
    dag = TestkitDAG()

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
        )
    )

    dag.add_source(linked)

    # Write sources to warehouse
    _testkitdag_to_location(warehouse_engine, dag)

    # Index sources in backend
    for source_testkit in dag.sources.values():
        backend.index(
            source_config=source_testkit.source_config,
            data_hashes=source_testkit.data_hashes,
        )

    # Create and add deduplication models
    for testkit in dag.sources.values():
        source = testkit.source_config
        name = f"naive_test.{source.name}"

        # Query the raw data
        source_query = backend.query(source=source.name)

        # Build model testkit using query data
        model_testkit = query_to_model_factory(
            left_resolution=source.name,
            left_query=source_query,
            left_keys={source.name: "key"},
            true_entities=tuple(linked.true_entities),
            name=name,
            description=f"Deduplication of {source.name}",
            prob_range=(1.0, 1.0),
            seed=seed,
        )

        assert model_testkit.probabilities.num_rows > 0

        # Add to DAG
        dag.add_model(model_testkit)

    return dag


_DATABASE_SNAPSHOTS_CACHE: dict[str, tuple[TestkitDAG, MatchboxSnapshot]] = {}


@contextmanager
def setup_scenario(
    backend: MatchboxDBAdapter,
    scenario_type: Literal["bare", "index", "dedupe", "link", "convergent"],
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
        dag, snapshot = _DATABASE_SNAPSHOTS_CACHE[cache_key]
        dag = dag.model_copy(deep=True)

        # Restore backend and write sources to warehouse
        backend.restore(snapshot=snapshot)
        _testkitdag_to_location(warehouse, dag)
    else:
        # Create new TestkitDAG with proper backend integration
        scenario_builder = SCENARIO_REGISTRY[scenario_type]
        dag = scenario_builder(backend, warehouse, n_entities, seed, **kwargs)

        # Cache the snapshot and DAG
        _DATABASE_SNAPSHOTS_CACHE[cache_key] = (dag, backend.dump())

    try:
        yield dag
    finally:
        backend.clear(certain=True)


# Warehouse database fixtures


class DevelopmentSettings(BaseSettings):
    api_port: int = 8000
    datastore_console_port: int = 9003
    datastore_port: int = 9002
    warehouse_port: int = 7654
    postgres_backend_port: int = 9876

    model_config = SettingsConfigDict(
        extra="ignore",
        env_prefix="MB__DEV__",
        env_nested_delimiter="__",
        env_file=Path("environments/development.env"),
        env_file_encoding="utf-8",
    )


@pytest.fixture(scope="session")
def development_settings() -> Generator[DevelopmentSettings, None, None]:
    """Settings for the development environment."""
    settings = DevelopmentSettings()
    yield settings


@pytest.fixture(scope="function")
def postgres_warehouse(
    development_settings: DevelopmentSettings,
) -> Generator[Engine, None, None]:
    """Creates an engine for the test warehouse database"""
    user = "warehouse_user"
    password = "warehouse_password"
    host = "localhost"
    database = "warehouse"
    port = development_settings.warehouse_port

    engine = create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    )
    yield engine
    engine.dispose()


@contextmanager
def named_temp_file(filename: str) -> Generator[io.BufferedWriter, None, None]:
    """
    Create a temporary file with a specific name that auto-deletes.

    Args:
        filename: Just the filename (not path) you want to use
    """
    temp_dir = Path(tempfile.gettempdir())
    full_path = temp_dir / filename
    try:
        with full_path.open(mode="wb") as f:
            yield f
    finally:
        if full_path.exists():
            full_path.unlink()


@pytest.fixture(scope="function")
def sqlite_warehouse() -> Generator[Engine, None, None]:
    """Creates an engine for a function-scoped SQLite warehouse database.

    By using a temporary file, produces a URI that can be shared between processes.
    """
    with named_temp_file("db.sqlite") as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        yield engine
        engine.dispose()


# Matchbox database fixtures


@pytest.fixture(scope="session")
def matchbox_datastore(
    development_settings: DevelopmentSettings,
) -> MatchboxDatastoreSettings:
    """Settings for the Matchbox datastore."""
    return MatchboxDatastoreSettings(
        host="localhost",
        port=development_settings.datastore_port,
        access_key_id="access_key_id",
        secret_access_key="secret_access_key",
        default_region="eu-west-2",
        cache_bucket_name="cache",
    )


@pytest.fixture(scope="session")
def matchbox_postgres_settings(
    development_settings: DevelopmentSettings,
    matchbox_datastore: MatchboxDatastoreSettings,
) -> MatchboxPostgresSettings:
    """Settings for the Matchbox PostgreSQL database."""
    return MatchboxPostgresSettings(
        batch_size=250_000,
        postgres={
            "host": "localhost",
            "port": development_settings.postgres_backend_port,
            "user": "matchbox_user",
            "password": "matchbox_password",
            "database": "matchbox",
            "db_schema": "mb",
            "alembic_config": "src/matchbox/server/postgresql/alembic.ini",
        },
        datastore=matchbox_datastore,
    )


@pytest.fixture(scope="function")
def matchbox_postgres(
    matchbox_postgres_settings: MatchboxPostgresSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """The Matchbox PostgreSQL database, cleared."""

    adapter = MatchboxPostgres(settings=matchbox_postgres_settings)

    # Clean up the Matchbox database before each test
    adapter.clear(certain=True)

    yield adapter

    # Clean up the Matchbox database after each test
    adapter.clear(certain=True)


@pytest.fixture(scope="function")
def matchbox_postgres_dropped(
    matchbox_postgres_settings: MatchboxPostgresSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """The Matchbox PostgreSQL database, dropped and recreated."""

    adapter = MatchboxPostgres(settings=matchbox_postgres_settings)

    # Clean up the Matchbox database before each test
    adapter.drop(certain=True)

    yield adapter

    # Clean up the Matchbox database after each test
    adapter.drop(certain=True)


# Mock AWS fixtures


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "eu-west-2"


@pytest.fixture(scope="function")
def s3(aws_credentials: None) -> Generator[S3Client, None, None]:
    """Return a mocked S3 client."""
    with mock_aws():
        yield boto3.client("s3", region_name="eu-west-2")


# API, mocked and Docker


@pytest.fixture(scope="function")
def matchbox_api() -> Generator[MockRouter, None, None]:
    """Client for the mocked Matchbox API."""
    with respx.mock(base_url=settings.api_root, assert_all_called=True) as respx_mock:
        yield respx_mock


@pytest.fixture(scope="session")
def matchbox_client_settings() -> ClientSettings:
    """Client settings for the Matchbox API running in Docker."""
    return settings


@pytest.fixture(scope="session")
def matchbox_client(matchbox_client_settings: ClientSettings) -> Client:
    """Client for the Matchbox API running in Docker."""
    return create_client(settings=matchbox_client_settings)
