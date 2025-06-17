import logging

import pytest
from httpx import Client
from splink import SettingsCreator
from splink import comparison_library as cl
from sqlalchemy import Engine, text

from matchbox import query
from matchbox.client.clean import remove_prefix, steps
from matchbox.client.clean.utils import cleaning_function, select_cleaners
from matchbox.client.dags import DAG, DedupeStep, IndexStep, LinkStep, StepInput
from matchbox.client.helpers import cleaner, select
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker, SplinkLinker
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)
from matchbox.common.sources import RelationalDBLocation, SourceConfig


@pytest.mark.docker
class TestE2EPipelineBuilder:
    """End to end tests for DAG pipeline functionality."""

    client: Client | None = None
    warehouse_engine: Engine | None = None
    linked_testkit: LinkedSourcesTestkit | None = None
    n_true_entities: int | None = None

    @pytest.fixture(autouse=True, scope="function")
    def setup_environment(
        self,
        matchbox_client: Client,
        postgres_warehouse: Engine,
    ):
        """Set up warehouse and database using fixtures."""
        # Store fixtures as class attributes
        n_true_entities = 10  # Keep it small for simplicity

        self.__class__.client = matchbox_client
        self.__class__.warehouse_engine = postgres_warehouse
        self.__class__.n_true_entities = n_true_entities

        # Create simple feature configurations - just two sources
        features = {
            "company_name": FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            "registration_id": FeatureConfig(
                name="registration_id",
                base_generator="bothify",
                parameters=(("text", "REG-###-???"),),
            ),
        }

        # Create two simple sources that can be linked
        source_parameters = (
            SourceTestkitParameters(
                name="source_a",
                engine=postgres_warehouse,
                features=(
                    features["company_name"].add_variations(
                        SuffixRule(suffix=" Ltd"),
                        SuffixRule(suffix=" Limited"),
                    ),
                    features["registration_id"],
                ),
                n_true_entities=n_true_entities,
                repetition=0,  # No duplicates
            ),
            SourceTestkitParameters(
                name="source_b",
                engine=postgres_warehouse,
                features=(
                    features["company_name"],
                    features["registration_id"],
                ),
                n_true_entities=n_true_entities // 2,  # Half overlap
                repetition=1,  # Duplicate all rows for deduplication testing
            ),
        )

        # Create linked sources testkit
        self.__class__.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters,
            seed=42,
        )

        # Setup - Create tables in warehouse
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location(
                credentials=postgres_warehouse, set_credentials=True
            )

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

        yield

        # Teardown
        with postgres_warehouse.connect() as conn:
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))
            conn.commit()

        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    def test_dag_pipeline_creation_and_rerun(self):
        """Test DAG API with simple two-source pipeline

        Rerun to test overwriting.
        """

        # === SETUP PHASE ===
        dw_loc = RelationalDBLocation.from_engine(self.warehouse_engine)
        batch_size = 1000

        # Create source configs
        source_a_config = SourceConfig.new(
            location=dw_loc,
            name="source_a",
            extract_transform="""
                select
                    key::text as id, 
                    company_name, 
                    registration_id 
                from
                    source_a;
            """,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        source_b_config = SourceConfig.new(
            location=dw_loc,
            name="source_b",
            extract_transform="""
                select
                    key::text as id,
                    company_name,
                    registration_id
                from
                    source_b;
            """,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        # Create simple cleaners
        clean_company_name = cleaning_function(
            steps.tokenise,
            lambda tokens: [t for t in tokens if t not in ["Ltd", "Limited"]],
            steps.list_join_to_string,
            steps.trim,
        )

        source_a_cleaners = {
            "company_name": cleaner(
                clean_company_name,
                {"column": source_a_config.qualify_field("company_name")},
            ),
            "make_conformant": cleaner(
                remove_prefix,
                {"column": "", "prefix": source_a_config.qualify_field("")},
            ),
        }

        source_b_cleaners = {
            "company_name": cleaner(
                clean_company_name,
                {"column": source_b_config.qualify_field("company_name")},
            ),
            "make_conformant": cleaner(
                remove_prefix,
                {"column": "", "prefix": source_b_config.qualify_field("")},
            ),
        }

        # === DAG DEFINITION ===
        # Index steps
        i_source_a = IndexStep(source_config=source_a_config, batch_size=batch_size)
        i_source_b = IndexStep(source_config=source_b_config, batch_size=batch_size)

        # Dedupe steps
        dedupe_a = DedupeStep(
            left=StepInput(
                prev_node=i_source_a,
                select={source_a_config: ["company_name", "registration_id"]},
                cleaners=select_cleaners(
                    (source_a_cleaners, ["company_name"]),
                ),
                batch_size=batch_size,
            ),
            name="dedupe_source_a",
            description="Deduplicate source A",
            model_class=NaiveDeduper,
            settings={
                "id": "id",
                "unique_fields": [source_a_config.qualify_field("registration_id")],
            },
            truth=1.0,
        )

        dedupe_b = DedupeStep(
            left=StepInput(
                prev_node=i_source_b,
                select={source_b_config: ["company_name", "registration_id"]},
                cleaners=select_cleaners(
                    (source_b_cleaners, ["company_name"]),
                ),
                batch_size=batch_size,
            ),
            name="dedupe_source_b",
            description="Deduplicate source B",
            model_class=NaiveDeduper,
            settings={
                "id": "id",
                "unique_fields": [source_b_config.qualify_field("registration_id")],
            },
            truth=1.0,
        )

        # Link step
        link_ab = LinkStep(
            left=StepInput(
                prev_node=dedupe_a,
                select={source_a_config: ["company_name", "registration_id"]},
                cleaners=select_cleaners(
                    (source_a_cleaners, ["company_name"]),
                ),
                batch_size=batch_size,
            ),
            right=StepInput(
                prev_node=dedupe_b,
                select={source_b_config: ["company_name", "registration_id"]},
                cleaners=select_cleaners(
                    (source_b_cleaners, ["company_name"]),
                ),
                batch_size=batch_size,
            ),
            name="__DEFAULT__",
            description="Link sources A and B on registration_id",
            model_class=DeterministicLinker,
            settings={
                "left_id": "id",
                "right_id": "id",
                "comparisons": (
                    f"l.{source_a_config.qualify_field('registration_id')}"
                    f"= r.{source_b_config.qualify_field('registration_id')}",
                ),
            },
            truth=1.0,
        )

        # Create and run DAG
        dag = DAG()
        dag.add_steps(i_source_a, i_source_b)
        dag.add_steps(dedupe_a, dedupe_b)
        dag.add_steps(link_ab)

        # === FIRST RUN ===
        logging.info("Running DAG for the first time")
        dag.run()

        # Basic verification - check that we have some linked results

        final_df = query(
            select(
                {
                    source_a_config.name: ["company_name", "registration_id"],
                    source_b_config.name: ["company_name", "registration_id"],
                },
                credentials=self.warehouse_engine,
            ),
            resolution="__DEFAULT__",
            return_type="pandas",
        )

        # Should have linked results
        assert len(final_df) > 0, "Expected some results from first run"
        assert final_df["id"].nunique() == len(self.linked_testkit.true_entities)

        first_run_entities = final_df["id"].nunique()
        logging.info(f"First run produced {first_run_entities} unique entities")

        # === SECOND RUN (OVERWRITE) ===
        logging.info("Running DAG again to test overwriting")
        dag.run()

        # Verify second run produces same results
        final_df_second = query(
            select(
                {
                    source_a_config.name: ["company_name", "registration_id"],
                    source_b_config.name: ["company_name", "registration_id"],
                },
                credentials=self.warehouse_engine,
            ),
            resolution="__DEFAULT__",
            return_type="pandas",
        )

        second_run_entities = final_df_second["id"].nunique()
        logging.info(f"Second run produced {second_run_entities} unique entities")

        # Should have same number of entities after rerun
        assert first_run_entities == second_run_entities, (
            "Expected same results after rerun: "
            f"{first_run_entities} vs {second_run_entities}"
        )

        # Basic sanity checks
        assert len(final_df_second) > 0, "Expected some results from second run"

        # === SECOND PARALLEL DAG RUN ===
        logging.info("Running second parallel link step with Splink linker")

        # Define comparisons using Splink's built-in comparison library
        comparisons = [
            cl.JaroWinklerAtThresholds("company_name", [0.9, 0.8]),
            cl.ExactMatch("registration_id"),
        ]

        # Training functions
        linker_training_functions = [
            {
                "function": "estimate_probability_two_random_records_match",
                "arguments": {
                    "deterministic_matching_rules": [
                        "l.registration_id = r.registration_id",
                        "l.company_name = r.company_name",
                    ],
                    "recall": 0.8,
                },
            },
            {
                "function": "estimate_u_using_random_sampling",
                "arguments": {"max_pairs": 1e3},
            },
            {
                "function": "estimate_parameters_using_expectation_maximisation",
                "arguments": {
                    "blocking_rule": ("l.registration_id = r.registration_id"),
                },
            },
        ]

        # Blocking rules
        blocking_rules = [
            "l.registration_id = r.registration_id",
            "l.company_name = r.company_name",
        ]

        linker_ab_2 = LinkStep(
            left=StepInput(
                prev_node=dedupe_a,
                select={source_a_config: ["company_name", "registration_id"]},
                cleaners=select_cleaners(
                    (source_a_cleaners, ["company_name", "make_conformant"]),
                ),
                batch_size=batch_size,
            ),
            right=StepInput(
                prev_node=dedupe_b,
                select={source_b_config: ["company_name", "registration_id"]},
                cleaners=select_cleaners(
                    (source_b_cleaners, ["company_name", "make_conformant"]),
                ),
                batch_size=batch_size,
            ),
            name="splink_linker",
            description="Link sources A and B on registration_id (second run)",
            model_class=SplinkLinker,
            settings={
                "left_id": "id",
                "right_id": "id",
                "linker_training_functions": linker_training_functions,
                "linker_settings": SettingsCreator(
                    link_type="link_only",
                    retain_matching_columns=False,
                    retain_intermediate_calculation_columns=False,
                    blocking_rules_to_generate_predictions=blocking_rules,
                    comparisons=comparisons,
                ),
                "threshold": 0.01,
            },
            truth=0.01,
        )

        # Create and run second DAG
        second_dag = DAG()
        second_dag.add_steps(i_source_a, i_source_b)
        second_dag.add_steps(dedupe_a, dedupe_b)
        second_dag.add_steps(linker_ab_2)

        second_dag.run()

        # Basic verification - check that we have some linked results

        second_final_df = query(
            select(
                {
                    source_a_config.name: ["company_name", "registration_id"],
                    source_b_config.name: ["company_name", "registration_id"],
                },
                credentials=self.warehouse_engine,
            ),
            resolution="splink_linker",
            return_type="pandas",
        )

        # Should have linked results
        assert len(second_final_df) > 0, "Expected some results from second DAG"
        assert second_final_df["id"].nunique() == len(self.linked_testkit.true_entities)

        logging.info("DAG pipeline test completed successfully!")
