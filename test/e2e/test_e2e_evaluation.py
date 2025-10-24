import pytest
from httpx import Client
from sqlalchemy import Engine, text

from matchbox.client import _handler
from matchbox.client.cli.eval import EntityResolutionApp
from matchbox.client.dags import DAG
from matchbox.client.eval import compare_models
from matchbox.client.locations import RelationalDBLocation
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.common.factories.sources import (
    FeatureConfig,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EModelEvaluation:
    """End to end tests for model evaluation functionality."""

    def _clean_company_name(self, column: str) -> str:
        """Generate cleaning SQL for a company name column."""
        return f"""
            trim(
                regexp_replace(
                    {column},
                    ' (Ltd|Limited)$',
                    '',
                    'g'
                )
            )
        """

    @pytest.fixture(autouse=True, scope="function")
    def setup_environment(
        self,
        matchbox_client: Client,
        postgres_warehouse: Engine,
    ):
        """Set up warehouse and database using fixtures."""
        # Persist shared setup for use in the test body
        n_true_entities = 10
        final_resolution_1_name = "final_1"
        final_resolution_2_name = "final_2"
        self.warehouse_engine = postgres_warehouse

        # Create a SINGLE source with duplicates
        source_parameters = (
            SourceTestkitParameters(
                name="source_a",
                engine=postgres_warehouse,
                features=(
                    FeatureConfig(
                        name="company_name",
                        base_generator="company",
                    ).add_variations(
                        SuffixRule(suffix=" Ltd"),
                        SuffixRule(suffix=" Limited"),
                    ),
                    FeatureConfig(
                        name="registration_id",
                        base_generator="bothify",
                        parameters=(("text", "REG-###-???"),),
                    ),
                ),
                n_true_entities=n_true_entities,
                repetition=1,  # Duplicate all rows for deduplication
            ),
        )
        self.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters, seed=42
        )
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location()

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

        # Create DAG
        dw_loc = RelationalDBLocation(name="postgres").set_client(postgres_warehouse)

        # === DAG 1: Created by User 1 (Strict Deduplication) ===
        dag1 = DAG("companies1").new_run()

        source_a_dag1 = dag1.source(
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
            infer_types=True,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        source_a_dag1.query().deduper(
            name=final_resolution_1_name,
            description="Deduplicate by registration ID",
            model_class=NaiveDeduper,
            model_settings={
                "id": "id",
                "unique_fields": [source_a_dag1.f("registration_id")],
            },
        )

        dag1.run_and_sync()

        # Retain DAG for use in tests
        self.dag1 = dag1

        # === DAG 2: Created by User 2 (Loose Deduplication) ===
        dag2 = DAG("companies2").new_run()

        source_a_dag2 = dag2.source(
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
            infer_types=True,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        source_a_dag2.query(
            cleaning={
                "company_name": self._clean_company_name(
                    source_a_dag2.f("company_name")
                )
            }
        ).deduper(
            name=final_resolution_2_name,
            description="Deduplicate by company name",
            model_class=NaiveDeduper,
            model_settings={"id": "id", "unique_fields": ["company_name"]},
        )

        dag2.run_and_sync()

        self.dag2 = dag2

        yield

        # Teardown
        with postgres_warehouse.connect() as conn:
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))
            conn.commit()
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    @pytest.mark.asyncio
    async def test_evaluation_workflow(self):
        """Test end-to-end data pipeline: DAG → samples → judgement → model comparison.

        This test focuses on the full data flow through the system with real warehouse
        data, multiple DAGs, and model comparison. UI interaction details are tested
        separately in unit/integration tests.
        """
        # Load DAG from server with warehouse location
        dag: DAG = (
            DAG(str(self.dag1.name)).load_pending().set_client(self.warehouse_engine)
        )

        # Create app and verify it can load samples from real data
        app = EntityResolutionApp(
            resolution=dag.final_step.resolution_path.name,
            num_samples=2,
            user="alice",
            dag=dag,
        )

        async with app.run_test() as pilot:
            await pilot.pause()

            # Verify app authenticated and loaded samples from real warehouse data
            await app.authenticate()
            assert app.user_id is not None

            if not app.queue.items:
                await app.load_samples()

            assert len(app.queue.items) > 0, "Should load samples from warehouse"

            # Submit one judgement to verify data flow
            item = app.queue.items[0]
            for i in range(len(item.display_columns)):
                item.assignments[i] = "a"  # Assign all to same cluster

            initial_judgements, _ = _handler.download_eval_data()
            initial_count = len(initial_judgements)

            await app.action_submit()

            final_judgements, _ = _handler.download_eval_data()
            assert len(final_judgements) == initial_count + 1, (
                "Judgement should flow through to backend"
            )

        # Test model comparison functionality with both DAGs
        comparison = compare_models(
            [
                dag.final_step.resolution_path,
                self.dag2.final_step.resolution_path,
            ]
        )
        expected_keys = {
            str(dag.final_step.resolution_path),
            str(self.dag2.final_step.resolution_path),
        }
        assert expected_keys.issubset(comparison.keys()), (
            "Comparison should include both models"
        )
        for key in expected_keys:
            assert len(comparison[key]) == 2, "Each model should have precision/recall"
