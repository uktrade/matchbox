import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Literal

import boto3
import pyarrow as pa
import pytest
import respx
from httpx import Client
from moto import mock_aws
from respx import MockRouter
from sqlalchemy import Engine, create_engine

from matchbox.client._handler import create_client
from matchbox.client._settings import ClientSettings, settings
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.models import query_to_model_factory
from matchbox.common.factories.sources import linked_sources_factory
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


def _generate_cache_key(
    backend: MatchboxDBAdapter,
    scenario_type: Literal["bare", "index", "dedupe", "link"],
    warehouse: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> str:
    """Generate a unique hash based on input parameters"""
    cache_key = (
        f"{warehouse.url}_{backend.__class__.__name__}_"
        f"{scenario_type}_{n_entities}_{seed}"
    )
    return cache_key


def _testkitdag_to_warehouse(warehouse_engine: Engine, dag: TestkitDAG) -> None:
    """Upload a TestkitDAG to a warehouse.

    * Writes all data to the warehouse, replacing existing data
    * Updates the engine of all sources in the DAG
    """
    for source_testkit in dag.sources.values():
        source_testkit.to_warehouse(warehouse_engine)
        source_testkit.source.set_engine(warehouse_engine)


def create_scenario_dag(
    backend: MatchboxDBAdapter,
    warehouse_engine: Engine,
    scenario_type: Literal["bare", "index", "dedupe", "link"],
    n_entities: int = 10,
    seed: int = 42,
) -> TestkitDAG:
    """Create a TestkitDAG representing a test scenario with backend integration.

    This approach first creates source data, writes it to backend and warehouse,
    then builds models by querying the backend to ensure ID alignment.
    """
    # Validate inputs
    if scenario_type not in ["bare", "index", "dedupe", "link"]:
        raise ValueError(f"Invalid scenario: {scenario_type}")

    dag = TestkitDAG()

    # 1. Create linked sources
    linked = linked_sources_factory(
        n_true_entities=n_entities, seed=seed, engine=warehouse_engine
    )
    dag.add_source(linked)

    # 2. Write sources to warehouse
    _testkitdag_to_warehouse(warehouse_engine, dag)

    # End here for bare database scenarios
    if scenario_type == "bare":
        return dag

    # 3. Index sources in backend
    for source_testkit in dag.sources.values():
        backend.index(
            source=source_testkit.source, data_hashes=source_testkit.data_hashes
        )

    # End here for index-only scenarios
    if scenario_type == "index":
        return dag

    # 4. Create and add deduplication models
    for testkit in dag.sources.values():
        source = testkit.source
        model_name = f"naive_test.{source.address.full_name}"

        # Query the raw data
        source_query = backend.query(
            source_address=linked.sources[source.address.full_name].source.address,
        )

        # Build model testkit using query data
        model_testkit = query_to_model_factory(
            left_resolution=source.resolution_name,
            left_query=source_query,
            left_source_pks={source.address.full_name: "source_pk"},
            true_entities=tuple(linked.true_entities),
            name=model_name,
            description=f"Deduplication of {source.address.full_name}",
            prob_range=(1.0, 1.0),
            seed=seed,
        )

        # Add to backend and DAG
        backend.insert_model(model=model_testkit.model.metadata)
        backend.set_model_results(model=model_name, results=model_testkit.probabilities)
        dag.add_model(model_testkit)

    # End here for dedupe-only scenarios
    if scenario_type == "dedupe":
        return dag

    # 5. Create linking models
    # First create CRN-DUNS link
    crn_model = dag.models["naive_test.crn"]
    duns_model = dag.models["naive_test.duns"]
    cdms_model = dag.models["naive_test.cdms"]

    # Query data for each resolution
    crn_query = backend.query(
        source_address=linked.sources["crn"].source.address,
        resolution_name=crn_model.name,
    )

    duns_query = backend.query(
        source_address=linked.sources["duns"].source.address,
        resolution_name=duns_model.name,
    )

    cdms_query = backend.query(
        source_address=linked.sources["cdms"].source.address,
        resolution_name=cdms_model.name,
    )

    # Create CRN-DUNS link
    crn_duns_name = "deterministic_naive_test.crn_naive_test.duns"
    crn_duns_model = query_to_model_factory(
        left_resolution=crn_model.name,
        left_query=crn_query,
        left_source_pks={"crn": "source_pk"},
        right_resolution=duns_model.name,
        right_query=duns_query,
        right_source_pks={"duns": "source_pk"},
        true_entities=tuple(linked.true_entities),
        name=crn_duns_name,
        description="Link between CRN and DUNS",
        prob_range=(1.0, 1.0),
        seed=seed,
    )

    # Add to backend and DAG
    backend.insert_model(model=crn_duns_model.model.metadata)
    backend.set_model_results(model=crn_duns_name, results=crn_duns_model.probabilities)
    dag.add_model(crn_duns_model)

    # Create CRN-CDMS link
    crn_cdms_name = "probabilistic_naive_test.crn_naive_test.cdms"
    crn_cdms_model = query_to_model_factory(
        left_resolution=crn_model.name,
        left_query=crn_query,
        left_source_pks={"crn": "source_pk"},
        right_resolution=cdms_model.name,
        right_query=cdms_query,
        right_source_pks={"cdms": "source_pk"},
        true_entities=tuple(linked.true_entities),
        name=crn_cdms_name,
        description="Link between CRN and CDMS",
        seed=seed,
    )

    backend.insert_model(model=crn_cdms_model.model.metadata)
    backend.set_model_results(model=crn_cdms_name, results=crn_cdms_model.probabilities)
    backend.set_model_truth(model=crn_cdms_name, truth=75)
    dag.add_model(crn_cdms_model)

    # Create final join
    # Query the previous link's results
    crn_cdms_query_crn_only = backend.query(
        source_address=linked.sources["crn"].source.address,
        resolution_name=crn_cdms_name,
    ).rename_columns(["id", "source_pk_crn"])
    crn_cdms_query_cdms_only = backend.query(
        source_address=linked.sources["cdms"].source.address,
        resolution_name=crn_cdms_name,
    ).rename_columns(["id", "source_pk_cdms"])
    crn_cdms_query = pa.concat_tables(
        [crn_cdms_query_crn_only, crn_cdms_query_cdms_only],
        promote_options="default",
    ).combine_chunks()

    duns_query_linked = backend.query(
        source_address=linked.sources["duns"].source.address,
        resolution_name=crn_duns_name,
    )

    final_join_name = "final_join"
    final_join_model = query_to_model_factory(
        left_resolution=crn_cdms_name,
        left_query=crn_cdms_query,
        left_source_pks={"crn": "source_pk_crn", "cdms": "source_pk_cdms"},
        right_resolution=duns_model.name,
        right_query=duns_query_linked,
        right_source_pks={"duns": "source_pk"},
        true_entities=tuple(linked.true_entities),
        name=final_join_name,
        description="Final join of all entities",
        seed=seed,
    )

    backend.insert_model(model=final_join_model.model.metadata)
    backend.set_model_results(
        model=final_join_name, results=final_join_model.probabilities
    )
    dag.add_model(final_join_model)

    return dag


_DATABASE_SNAPSHOTS_CACHE: dict[str, tuple[TestkitDAG, MatchboxSnapshot]] = {}


@contextmanager
def setup_scenario(
    backend: MatchboxDBAdapter,
    scenario_type: Literal["bare", "index", "dedupe", "link"],
    warehouse: Engine,
    n_entities: int = 10,
    seed: int = 42,
) -> Generator[TestkitDAG, None, None]:
    """Context manager for creating TestkitDAG scenarios."""
    # Generate cache key for backend snapshot
    cache_key = _generate_cache_key(backend, scenario_type, warehouse, n_entities, seed)

    # Check if we have a backend snapshot cached
    if cache_key in _DATABASE_SNAPSHOTS_CACHE:
        # Load cached snapshot and DAG
        dag, snapshot = _DATABASE_SNAPSHOTS_CACHE[cache_key]
        dag = dag.model_copy(deep=True)

        # Restore backend and write sources to warehouse
        backend.restore(clear=True, snapshot=snapshot)
        _testkitdag_to_warehouse(warehouse, dag)
    else:
        # Create new TestkitDAG with proper backend integration
        dag = create_scenario_dag(backend, warehouse, scenario_type, n_entities, seed)

        # Cache the snapshot and DAG
        _DATABASE_SNAPSHOTS_CACHE[cache_key] = (dag, backend.dump())

    yield dag

    backend.clear(certain=True)


# Warehouse database fixtures


@pytest.fixture(scope="function")
def postgres_warehouse() -> Generator[Engine, None, None]:
    """Creates an engine for the test warehouse database"""
    user = "warehouse_user"
    password = "warehouse_password"
    host = "localhost"
    database = "warehouse"
    port = 7654

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
def matchbox_datastore() -> MatchboxDatastoreSettings:
    """Settings for the Matchbox datastore."""
    return MatchboxDatastoreSettings(
        host="localhost",
        port=9000,
        access_key_id="access_key_id",
        secret_access_key="secret_access_key",
        default_region="eu-west-2",
        cache_bucket_name="cache",
    )


@pytest.fixture(scope="session")
def matchbox_settings(
    matchbox_datastore: MatchboxDatastoreSettings,
) -> MatchboxPostgresSettings:
    """Settings for the Matchbox database."""
    return MatchboxPostgresSettings(
        batch_size=250_000,
        postgres={
            "host": "localhost",
            "port": 5432,
            "user": "matchbox_user",
            "password": "matchbox_password",
            "database": "matchbox",
            "db_schema": "mb",
        },
        datastore=matchbox_datastore,
    )


@pytest.fixture(scope="function")
def matchbox_postgres(
    matchbox_settings: MatchboxPostgresSettings,
) -> Generator[MatchboxPostgres, None, None]:
    """The Matchbox PostgreSQL database."""

    adapter = MatchboxPostgres(settings=matchbox_settings)

    # Clean up the Matchbox database before each test, just in case
    adapter.clear(certain=True)

    yield adapter

    # Clean up the Matchbox database after each test
    adapter.clear(certain=True)


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
