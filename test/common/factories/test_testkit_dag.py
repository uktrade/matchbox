import pytest

from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.entities import FeatureConfig
from matchbox.common.factories.models import model_factory
from matchbox.common.factories.sources import (
    SourceTestkitParameters,
    linked_sources_factory,
    source_factory,
)


@pytest.mark.parametrize(
    ("chain_config", "expected_sources_by_model"),
    [
        pytest.param(
            [
                {
                    "name": "deduper_crn",
                    "sources": ["crn"],
                    "previous_models": [],
                    "standalone": False,
                }
            ],
            {
                "deduper_crn": ["crn"],
            },
            id="simple_deduper_finds_linked",
        ),
        pytest.param(
            [
                {
                    "name": "deduper_cdms",
                    "sources": ["cdms"],
                    "previous_models": [],
                    "standalone": False,
                }
            ],
            {
                "deduper_cdms": ["cdms"],
            },
            id="another_deduper_finds_linked",
        ),
        pytest.param(
            [
                {
                    "name": "standalone_deduper",
                    "sources": ["standalone_source"],
                    "previous_models": [],
                    "standalone": True,
                }
            ],
            {
                "standalone_deduper": [],
            },
            id="standalone_source_no_linked",
        ),
        pytest.param(
            [
                {
                    "name": "deduper_crn",
                    "sources": ["crn"],
                    "previous_models": [],
                    "standalone": False,
                },
                {
                    "name": "deduper_cdms",
                    "sources": ["cdms"],
                    "previous_models": [],
                    "standalone": False,
                },
                {
                    "name": "linker_models",
                    "sources": [],
                    "previous_models": ["deduper_crn", "deduper_cdms"],
                    "standalone": False,
                },
            ],
            {
                "deduper_crn": ["crn"],
                "deduper_cdms": ["cdms"],
                "linker_models": ["cdms", "crn"],
            },
            id="deduper_deduper_linker_chain",
        ),
        pytest.param(
            [
                {
                    "name": "deduper_crn",
                    "sources": ["crn"],
                    "previous_models": [],
                    "standalone": False,
                },
                {
                    "name": "linker_model_cdms",
                    "sources": ["cdms"],
                    "previous_models": ["deduper_crn"],
                    "standalone": False,
                },
            ],
            {
                "deduper_crn": ["crn"],
                "linker_model_cdms": ["cdms", "crn"],
            },
            id="deduper_linker_chain",
        ),
        pytest.param(
            [
                {
                    "name": "linker_crn_cdms",
                    "sources": ["crn", "cdms"],
                    "previous_models": [],
                    "standalone": False,
                }
            ],
            {
                "linker_crn_cdms": ["cdms", "crn"],
            },
            id="linker_finds_linked",
        ),
    ],
)
def test_testkit_dag_model_chain(
    chain_config: list[dict],
    expected_sources_by_model: dict[str, list[str]],
) -> None:
    """Test TestkitDAG with different model configurations including standalone sources.

    This test verifies that:
    1. We can create different types of models: single source dedupers, linkers, chains
    2. The DAG correctly tracks model dependencies
    3. We can find the original LinkedSourcesTestkit key from any model in the chain
    4. We can identify the correct sources for each model
    5. Standalone sources are handled correctly
    """
    # Setup: Create sources and DAG
    linked = linked_sources_factory(seed=hash(str(chain_config)))
    standalone = source_factory(
        name="standalone_source",
        features=[
            {"name": "name", "base_generator": "name"},
            {"name": "age", "base_generator": "random_int"},
        ],
    )

    dag = TestkitDAG()
    dag.add_source(linked)
    dag.add_source(standalone)

    linked_key = f"linked_{'_'.join(sorted(linked.sources.keys()))}"
    all_true_sources = tuple(linked.true_entities)
    models = {}

    # Build model chain
    for config in chain_config:
        # Extract config values
        name = config["name"]
        source_names = config["sources"]
        prev_models = config["previous_models"]
        is_standalone = config.get("standalone", False)

        # Determine testkits
        left_testkit = None
        right_testkit = None

        # Validate total inputs don't exceed 2
        total_inputs = len(prev_models) + len(source_names)
        if total_inputs > 2:
            raise ValueError(
                "Model can only have a maximum of two inputs, got "
                f"{len(prev_models)} previous models and {len(source_names)} sources"
            )

        # Set left testkit (always required)
        if prev_models:
            left_testkit = models[prev_models[0]]
        elif is_standalone:
            left_testkit = standalone
        elif source_names:
            left_testkit = linked.sources[source_names[0]]

        # Set right testkit (if there's a second input)
        if len(prev_models) == 2:
            right_testkit = models[prev_models[1]]
        elif len(prev_models) == 1 and source_names:
            right_testkit = linked.sources[source_names[0]]
        elif len(source_names) == 2:
            right_testkit = linked.sources[source_names[1]]

        # Create and add model
        model = model_factory(
            name=name,
            left_testkit=left_testkit,
            right_testkit=right_testkit,
            true_entities=all_true_sources,
        )
        dag.add_model(model)
        models[name] = model

    # Test: Verify sources for each model
    for name, expected_sources in expected_sources_by_model.items():
        sources_dict = dag.get_sources_for_model(name)

        if not expected_sources:  # Standalone case
            assert None in sources_dict, f"{name} should have standalone sources"
            assert linked_key not in sources_dict, (
                f"{name} should not use linked sources"
            )
        else:  # Linked sources case
            assert linked_key in sources_dict, f"{name} should use linked sources"
            actual_sources = {s.split("@")[0] for s in sources_dict[linked_key]}
            assert actual_sources == set(expected_sources), (
                f"{name} expected sources {expected_sources}, got {actual_sources}"
            )


def test_testkit_dag_multiple_linked_sources():
    """Test handling of multiple LinkedSourcesTestkit objects."""
    # Create features for source configs
    features = [
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="email", base_generator="email"),
        FeatureConfig(name="address", base_generator="address"),
    ]

    # Create two linked source testkits
    configs1 = (
        SourceTestkitParameters(name="foo1", features=tuple(features[:1])),
        SourceTestkitParameters(name="foo2", features=tuple(features[:1])),
    )
    configs2 = (
        SourceTestkitParameters(name="bar1", features=tuple(features[1:])),
        SourceTestkitParameters(name="bar2", features=tuple(features[1:])),
    )

    linked1 = linked_sources_factory(source_parameters=configs1, n_true_entities=10)
    linked2 = linked_sources_factory(source_parameters=configs2, n_true_entities=10)

    # Expected linked keys
    linked1_key = "linked_foo1_foo2"
    linked2_key = "linked_bar1_bar2"

    # Create DAG and add linked testkits
    dag = TestkitDAG()
    dag.add_source(linked1)
    dag.add_source(linked2)

    # Create and add models using sources from each linked testkit
    model1 = model_factory(
        left_testkit=linked1.sources["foo1"],
        true_entities=tuple(linked1.true_entities),
        seed=42,
    )
    model2 = model_factory(
        left_testkit=linked2.sources["bar1"],
        true_entities=tuple(linked2.true_entities),
        seed=43,
    )
    dag.add_model(model1)
    dag.add_model(model2)

    # Test single linked testkit models
    sources_model1 = dag.get_sources_for_model(model1.name)
    sources_model2 = dag.get_sources_for_model(model2.name)

    # Check that each model uses the correct linked testkit
    assert linked1_key in sources_model1 and linked1_key not in sources_model2
    assert linked2_key in sources_model2 and linked2_key not in sources_model1
    assert sources_model1[linked1_key] == {"foo1"}
    assert sources_model2[linked2_key] == {"bar1"}

    # Test cross-linked model
    model3 = model_factory(
        left_testkit=linked1.sources["foo1"],
        right_testkit=linked2.sources["bar1"],
        true_entities=tuple(list(linked1.true_entities) + list(linked2.true_entities)),
        seed=44,
    )
    dag.add_model(model3)
    sources_model3 = dag.get_sources_for_model(model3.name)

    # Check that model3 uses both linked testkits
    assert linked1_key in sources_model3 and linked2_key in sources_model3
    assert sources_model3[linked1_key] == {"foo1"}
    assert sources_model3[linked2_key] == {"bar1"}

    # Test model using multiple sources from same linked testkit
    model4 = model_factory(
        left_testkit=linked1.sources["foo1"],
        right_testkit=linked1.sources["foo2"],
        true_entities=tuple(linked1.true_entities),
        seed=45,
    )
    dag.add_model(model4)
    sources_model4 = dag.get_sources_for_model(model4.name)

    # Check that model4 uses both sources from linked1
    assert sources_model4[linked1_key] == {"foo1", "foo2"}


def test_testkit_dag_out_of_order_models():
    """Test that adding models in non-dependency order works correctly."""
    linked = linked_sources_factory(seed=42)
    dag = TestkitDAG()
    dag.add_source(linked)

    # Create models with dependencies
    model1 = model_factory(
        name="base_model",
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )

    model2 = model_factory(
        name="dependent_model",
        left_testkit=model1,  # This creates a dependency on model1
        true_entities=tuple(linked.true_entities),
    )

    # Add model2 out of order
    with pytest.raises(ValueError):
        dag.add_model(model2)
