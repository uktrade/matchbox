import pytest
from httpx import Client
from matplotlib.figure import Figure
from sqlalchemy import Engine, text

from matchbox.client import _handler
from matchbox.client.cli.eval import EvalData
from matchbox.client.cli.eval.ui import EntityResolutionApp
from matchbox.client.dags import DAG, DedupeStep, IndexStep, StepInput
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.common.arrow import SCHEMA_CLUSTER_EXPANSION, SCHEMA_JUDGEMENTS
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)
from matchbox.common.sources import RelationalDBLocation, SourceConfig


@pytest.mark.docker
class TestE2EModelEvaluation:
    """End to end tests for model evaluation functionality."""

    client: Client | None = None
    warehouse_engine: Engine | None = None
    linked_testkit: LinkedSourcesTestkit | None = None
    n_true_entities: int | None = None
    final_resolution: str | None = None

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
        # Store fixtures as class attributes
        n_true_entities = 10
        self.__class__.client = matchbox_client
        self.__class__.warehouse_engine = postgres_warehouse
        self.__class__.n_true_entities = n_true_entities
        self.__class__.final_resolution_1 = "final_1"
        self.__class__.final_resolution_2 = "final_2"
        self.__class__.final_resolution_1_results = None
        self.__class__.final_resolution_2_results = None

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
        self.__class__.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters, seed=42
        )
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location(client=postgres_warehouse, set_client=True)

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

        # === SHARED CONFIGURATION ===
        dw_loc = RelationalDBLocation(name="postgres", client=postgres_warehouse)
        batch_size = 1000
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

        # The IndexStep is a common prerequisite for both DAGs
        i_source_a = IndexStep(source_config=source_a_config, batch_size=batch_size)

        # === DAG 1: Created by User 1 (Strict Deduplication) ===
        dedupe_by_id = DedupeStep(
            left=StepInput(
                prev_node=i_source_a,
                select={source_a_config: ["registration_id"]},
                batch_size=batch_size,
            ),
            name=self.final_resolution_1,
            description="Deduplicate by registration ID",
            model_class=NaiveDeduper,
            settings={
                "id": "id",
                "unique_fields": [source_a_config.f("registration_id")],
            },
            truth=1.0,
        )
        dag1 = DAG()
        dag1.add_steps(i_source_a, dedupe_by_id)
        dag1.run()

        # Stash results
        self.final_resolution_1_results = dedupe_by_id.run(for_eval=True)

        # === DAG 2: Created by User 2 (Loose Deduplication) ===
        # This DAG also indexes the same source and applies a different rule.
        # Its final step name is what we will use for the evaluation.
        dedupe_by_name = DedupeStep(
            left=StepInput(
                prev_node=i_source_a,
                select={source_a_config: ["company_name"]},
                cleaning_dict={
                    "company_name": self._clean_company_name(
                        source_a_config.f("company_name")
                    )
                },
                batch_size=batch_size,
            ),
            name=self.final_resolution_2,
            description="Deduplicate by company name",
            model_class=NaiveDeduper,
            settings={"id": "id", "unique_fields": ["company_name"]},
            truth=0.8,
        )
        dag2 = DAG()
        dag2.add_steps(i_source_a, dedupe_by_name)
        dag2.run()

        # Stash results
        self.final_resolution_2_results = dedupe_by_name.run(for_eval=True)

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
        """Test complete end-to-end evaluation workflow.

        app startup → user painting → submission → model evaluation.
        """
        app = EntityResolutionApp(
            resolution=self.final_resolution_1,
            num_samples=5,
            user="alice",
            warehouse=self.warehouse_engine.url,
        )

        # Test the complete user workflow with Textual UI
        async with app.run_test() as pilot:
            await pilot.pause()

            # Authenticate the app
            await app.authenticate()

            # Phase 1: App should be authenticated and have samples
            assert app.state.user_name == "alice"
            assert app.state.user_id is not None

            # Let the app fetch samples as a user would experience
            if not app.state.queue.items:
                await app._fetch_additional_samples()

            # Should now have samples to work with
            assert len(app.state.queue.items) > 0, "App should have loaded samples"

            # Phase 2: Simulate user painting clusters (as user would do)
            initial_items = list(app.state.queue.items)
            painted_count = 0

            for item in initial_items[:2]:  # Paint first 2 items like a user would
                # Paint each display column to different groups
                for i, _ in enumerate(item.display_columns):
                    group = "a" if i % 2 == 0 else "b"  # Alternate assignments
                    item.assignments[i] = group
                painted_count += 1

            # Verify we have painted items
            painted_items = [item for item in app.state.queue.items if item.is_painted]
            assert (
                len(painted_items) >= 1
            ), "Should have painted items ready for submission"

            # Phase 3: Submit judgements to backend
            initial_judgements, _ = _handler.download_eval_data()
            initial_count = len(initial_judgements)

            # Submit painted items using the app's method
            await app.action_submit_and_fetch()

            # Verify judgements were submitted
            final_judgements, _ = _handler.download_eval_data()
            final_count = len(final_judgements)

            assert (
                final_count > initial_count
            ), "Should have more judgements after submission"

        # Phase 4: Test evaluation infrastructure with submitted judgements
        final_judgements, expansion = _handler.download_eval_data()
        assert SCHEMA_JUDGEMENTS.equals(final_judgements.schema)
        assert SCHEMA_CLUSTER_EXPANSION.equals(expansion.schema)
        assert len(final_judgements) > 0, "Should have judgements to evaluate with"

        # Phase 5: Test EvalData with real submitted judgements and DAG results
        eval_data = EvalData.from_results(self.final_resolution_1_results)
        assert SCHEMA_JUDGEMENTS.equals(eval_data.judgements.schema)
        assert SCHEMA_CLUSTER_EXPANSION.equals(eval_data.expansion.schema)
        assert (
            len(eval_data.judgements) > 0
        ), "EvalData should contain submitted judgements"

        # Get PR at 0.5 threshold
        pr_results = eval_data.precision_recall([0.5])
        assert len(pr_results) == 1
        _, p, r, _, _ = pr_results[0]
        pr = (p, r)
        assert isinstance(pr, tuple)
        assert len(pr) == 2

        # Test PR curve generation with real judgements
        pr_curve = eval_data.pr_curve_mpl()
        assert isinstance(pr_curve, Figure)
