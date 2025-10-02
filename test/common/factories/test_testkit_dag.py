"""Unit tests for TestkitDAG."""

import pytest

from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.entities import FeatureConfig
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import (
    SourceTestkitParameters,
    linked_sources_factory,
    source_factory,
)


def test_add_single_source():
    """Test adding a standalone source testkit."""
    dag_testkit = TestkitDAG()

    standalone = source_factory(
        name="test_source",
        features=[
            {"name": "company", "base_generator": "company"},
        ],
        dag=dag_testkit.dag,
    )

    dag_testkit.add_source(standalone)

    # Should store in sources registry
    assert "test_source" in dag_testkit.sources
    assert dag_testkit.sources["test_source"] == standalone

    # Should add to real DAG
    assert "test_source" in dag_testkit.dag.nodes
    assert dag_testkit.dag.nodes["test_source"] == standalone.source

    # Should not create linked entry
    assert len(dag_testkit.linked) == 0


def test_add_linked_sources():
    """Test adding a LinkedSourcesTestkit."""
    dag_testkit = TestkitDAG()

    features = [FeatureConfig(name="company", base_generator="company")]
    configs = (
        SourceTestkitParameters(name="source1", features=tuple(features)),
        SourceTestkitParameters(name="source2", features=tuple(features)),
    )

    linked = linked_sources_factory(
        source_parameters=configs, n_true_entities=5, dag=dag_testkit.dag
    )
    dag_testkit.add_linked_sources(linked)

    # Should store in linked registry with correct key
    expected_key = "linked_source1_source2"
    assert expected_key in dag_testkit.linked
    assert dag_testkit.linked[expected_key] == linked

    # Should store each source in sources registry
    assert "source1" in dag_testkit.sources
    assert "source2" in dag_testkit.sources
    assert dag_testkit.sources["source1"] == linked.sources["source1"]
    assert dag_testkit.sources["source2"] == linked.sources["source2"]

    # Should add both sources to real DAG
    assert "source1" in dag_testkit.dag.nodes
    assert "source2" in dag_testkit.dag.nodes


def test_add_model():
    """Test adding a model testkit."""
    dag_testkit = TestkitDAG()

    # Create source first
    linked = linked_sources_factory(n_true_entities=5, dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)

    # Create and add model
    model = model_factory(
        name="test_model",
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )
    dag_testkit.add_model(model)

    # Should store in models registry
    assert "test_model" in dag_testkit.models
    assert dag_testkit.models["test_model"] == model

    # Should add to real DAG
    assert "test_model" in dag_testkit.dag.nodes
    assert dag_testkit.dag.nodes["test_model"] == model.model


def test_get_linked_testkit_for_source():
    """Test getting linked testkit for a source."""
    dag_testkit = TestkitDAG()

    # Add linked sources
    linked = linked_sources_factory(n_true_entities=5, dag=dag_testkit.dag)
    dag_testkit.add_linked_sources(linked)

    # Add standalone source
    standalone = source_factory(name="standalone", dag=dag_testkit.dag)
    dag_testkit.add_source(standalone)

    # Should find linked testkit for sources that belong to one
    linked_testkit = dag_testkit.source_to_linked["crn"]
    assert linked_testkit == linked

    linked_testkit = dag_testkit.source_to_linked["duns"]
    assert linked_testkit == linked

    # Standalone and nonexistent sources should generate a KeyError
    assert "standalone" not in dag_testkit.source_to_linked
    assert "nonexistent" not in dag_testkit.source_to_linked


def test_multiple_linked_testkits():
    """Test handling multiple LinkedSourcesTestkit objects."""
    dag_testkit = TestkitDAG()

    # Create two different linked testkits
    features = [
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="email", base_generator="email"),
    ]

    configs1 = (
        SourceTestkitParameters(name="foo1", features=tuple(features[:1])),
        SourceTestkitParameters(name="foo2", features=tuple(features[:1])),
    )
    configs2 = (
        SourceTestkitParameters(name="bar1", features=tuple(features[1:])),
        SourceTestkitParameters(name="bar2", features=tuple(features[1:])),
    )

    linked1 = linked_sources_factory(
        source_parameters=configs1, n_true_entities=5, dag=dag_testkit.dag
    )
    linked2 = linked_sources_factory(
        source_parameters=configs2, n_true_entities=5, dag=dag_testkit.dag
    )

    dag_testkit.add_linked_sources(linked1)
    dag_testkit.add_linked_sources(linked2)

    # Expected linked keys
    assert "linked_foo1_foo2" in dag_testkit.linked
    assert "linked_bar1_bar2" in dag_testkit.linked
    assert dag_testkit.linked["linked_foo1_foo2"] == linked1
    assert dag_testkit.linked["linked_bar1_bar2"] == linked2

    # Sources should be registered separately
    assert len(dag_testkit.sources) == 4
    assert all(name in dag_testkit.sources for name in ["foo1", "foo2", "bar1", "bar2"])

    # Test linked testkit lookup works for each
    assert dag_testkit.source_to_linked["foo1"] == linked1
    assert dag_testkit.source_to_linked["bar1"] == linked2


def test_name_collision_caught_by_dag():
    """Test that name collisions are still caught by the real dag_testkit."""
    dag_testkit = TestkitDAG()

    # Add first source
    source1 = source_factory(name="duplicate_name", dag=dag_testkit.dag)
    dag_testkit.add_source(source1)

    # Try to add second source with same name
    source2 = source_factory(name="duplicate_name", dag=dag_testkit.dag)

    with pytest.raises(ValueError, match="already taken"):
        dag_testkit.add_source(source2)


def test_empty_dag_properties():
    """Test properties of an empty TestkitDAG."""
    dag_testkit = TestkitDAG()

    # All registries should be empty
    assert len(dag_testkit.sources) == 0
    assert len(dag_testkit.models) == 0
    assert len(dag_testkit.linked) == 0

    # get_linked_testkit should return None for anything
    assert "anything" not in dag_testkit.source_to_linked

    # DAG should be empty but valid
    assert len(dag_testkit.dag.nodes) == 0

    # final_step should raise on empty DAG
    with pytest.raises(ValueError):
        _ = dag_testkit.dag.final_step
