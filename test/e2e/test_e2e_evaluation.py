import pytest
from httpx import Client
from matplotlib.figure import Figure
from sqlalchemy import Engine, text

from matchbox import make_model, query, select
from matchbox.client import _handler
from matchbox.client.dags import DAG, DedupeStep, IndexStep, StepInput
from matchbox.client.eval import EvalData, compare_models, get_samples
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.sources import RelationalDBLocation, Source
from matchbox.common.arrow import SCHEMA_CLUSTER_EXPANSION, SCHEMA_JUDGEMENTS
from matchbox.common.eval import Judgement
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EModelEvaluation:
    """End to end tests for model evaluation functionality."""

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
        # Will ne beeded later
        self.__class__.engine = postgres_warehouse
        # Set up testkits
        n_true_entities = 10  # Keep it small for simplicity

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
                repetition=1,  # No duplicates
            ),
        )

        linked_testkit = linked_sources_factory(
            source_parameters=source_parameters,
            seed=42,
        )

        # Create tables in warehouse
        for source_testkit in linked_testkit.sources.values():
            source_testkit.write_to_location()

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

        # Create DAG
        dw_loc = RelationalDBLocation(name="postgres", client=postgres_warehouse)
        batch_size = 1000

        source_a = Source(
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
        self.__class__.source_a = source_a

        i_source_a = IndexStep(source=source_a, batch_size=batch_size)

        self.__class__.final_resolution = "final"
        dedupe_a = DedupeStep(
            left=StepInput(
                prev_node=i_source_a,
                select={source_a: ["company_name", "registration_id"]},
                batch_size=batch_size,
            ),
            name=self.final_resolution,
            description="Deduplicate source A",
            model_class=NaiveDeduper,
            settings={
                "id": "id",
                "unique_fields": [source_a.f("registration_id")],
            },
            truth=1.0,
        )

        dag = DAG()
        dag.add_steps(i_source_a, dedupe_a)
        dag.run()

        yield

        # Teardown
        with postgres_warehouse.connect() as conn:
            for source_name in linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))
            conn.commit()

        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    def test_evaluation_workflow(self):
        """Test sampling, judging and model scoring."""

        # "Login"
        user_id = _handler.login(user_name="alice")

        # Get some samples
        samples = get_samples(
            n=5,
            resolution=self.final_resolution,
            user_id=user_id,
            clients={"postgres": self.engine},
        )

        # Make some judgements
        judged_cluster = next(iter(samples.keys()))
        judged_leaves = samples[judged_cluster]["leaf"].unique().to_list()

        judgement = Judgement(
            user_id=user_id,
            shown=judged_cluster,
            endorsed=[judged_leaves[:1], judged_leaves[1:]],
        )

        _handler.send_eval_judgement(judgement=judgement)

        # Compare models based on judgements
        comparison = compare_models([self.final_resolution])
        assert "final" in comparison
        assert len(comparison["final"]) == 2
        assert isinstance(comparison["final"], tuple)

        # Create and run a deduper model locally
        queried_source = query(
            select(self.source_a.config.name, client=self.engine),
            return_type="polars",
        )
        deduper = make_model(
            name="deduper_alt",
            description="Deduplication",
            model_class=NaiveDeduper,
            model_settings={
                "id": "id",
                "unique_fields": [self.source_a.f("registration_id")],
            },
            left_data=queried_source,
            left_resolution=self.source_a.config.name,
        )

        results = deduper.run()

        # We can download judgements locally
        eval_data = EvalData()
        assert SCHEMA_JUDGEMENTS.equals(eval_data.judgements.schema)
        assert SCHEMA_CLUSTER_EXPANSION.equals(eval_data.expansion.schema)

        # We can evaluate local model with cached judgements
        assert isinstance(eval_data.pr_curve(results), Figure)
        pr = eval_data.precision_recall(results, threshold=0.5)
        assert isinstance(pr, tuple)
        assert len(pr) == 2
