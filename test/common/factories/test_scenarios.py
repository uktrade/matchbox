from unittest.mock import patch

import pytest
from sqlalchemy import Engine

from matchbox.common.factories.scenarios import (
    SCENARIO_REGISTRY,
    TestkitDAG,
    create_bare_scenario,
    register_scenario,
    setup_scenario,
)
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.docker
def test_setup_scenario_bare(
    matchbox_postgres: MatchboxDBAdapter, sqlite_warehouse: Engine
):
    """Test that the bare scenario can be set up."""
    with setup_scenario(matchbox_postgres, "bare", warehouse=sqlite_warehouse) as dag:
        assert isinstance(dag, TestkitDAG)
        assert len(dag.sources) > 0


def test_scenario_registry():
    """Test that the scenario registry contains the built-in scenarios."""
    assert "bare" in SCENARIO_REGISTRY
    assert "index" in SCENARIO_REGISTRY
    assert "dedupe" in SCENARIO_REGISTRY
    assert "link" in SCENARIO_REGISTRY
    assert "probabilistic_dedupe" in SCENARIO_REGISTRY
    assert "alt_dedupe" in SCENARIO_REGISTRY
    assert "convergent" in SCENARIO_REGISTRY


@pytest.mark.docker
def test_register_custom_scenario(
    matchbox_postgres: MatchboxDBAdapter, sqlite_warehouse: Engine
):
    """Test that a custom scenario can be registered and used."""

    @register_scenario("custom")
    def create_custom_scenario(backend, warehouse_engine, n_entities, seed, **kwargs):
        return create_bare_scenario(
            backend, warehouse_engine, n_entities=n_entities, seed=seed, **kwargs
        )

    assert "custom" in SCENARIO_REGISTRY
    with setup_scenario(matchbox_postgres, "custom", warehouse=sqlite_warehouse) as dag:
        assert isinstance(dag, TestkitDAG)

    # Clean up the registry
    del SCENARIO_REGISTRY["custom"]


@pytest.mark.docker
def test_setup_unknown_scenario(
    matchbox_postgres: MatchboxDBAdapter, sqlite_warehouse: Engine
):
    """Test that asking for an unknown scenario raises a ValueError."""
    with (
        pytest.raises(ValueError, match="Unknown scenario type: nonexistent"),
        setup_scenario(matchbox_postgres, "nonexistent", warehouse=sqlite_warehouse),
    ):
        pass


@pytest.mark.docker
@patch("matchbox.common.factories.scenarios._DATABASE_SNAPSHOTS_CACHE", {})
@patch("matchbox.common.factories.scenarios.SCENARIO_REGISTRY", {})
def test_caching_scenario(
    matchbox_postgres: MatchboxDBAdapter, sqlite_warehouse: Engine
):
    """Test that scenario caching works."""

    call_count = 0

    @register_scenario("cacheable")
    def create_cacheable_scenario(
        backend, warehouse_engine, n_entities, seed, **kwargs
    ):
        nonlocal call_count
        call_count += 1
        return create_bare_scenario(
            backend, warehouse_engine, n_entities=n_entities, seed=seed, **kwargs
        )

    with setup_scenario(matchbox_postgres, "cacheable", warehouse=sqlite_warehouse):
        pass
    assert call_count == 1

    # Running it again should use the cache
    with setup_scenario(matchbox_postgres, "cacheable", warehouse=sqlite_warehouse):
        pass
    assert call_count == 1

    # Running with a different seed should not use the cache
    with setup_scenario(
        matchbox_postgres, "cacheable", warehouse=sqlite_warehouse, seed=43
    ):
        pass
    assert call_count == 2
