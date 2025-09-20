import logging

import polars as pl
import pytest
from httpx import Client
from sqlalchemy import Engine, text

from matchbox.client.dags import DAG
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.sources import RelationalDBLocation
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EPipelineBuilder:
    """End to end tests for DAG pipeline functionality."""

    client: Client | None = None
    warehouse_engine: Engine | None = None
    linked_testkit: LinkedSourcesTestkit | None = None
    n_true_entities: int | None = None

    def _clean_company_name(self, column: str) -> str:
        """Generate cleaning SQL for a company name column.

        Removes company suffixes (Ltd, Limited) and normalises whitespace
        from the company_name field.
        """

        return f"""
            trim(
                regexp_replace(
                    regexp_replace(
                        {column},
                        ' (Ltd|Limited)$',
                        '',
                        'g'
                    ),
                    '\\s+',
                    ' ',
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
            source_testkit.write_to_location()

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
        dw_loc = RelationalDBLocation(name="dbname", client=self.warehouse_engine)

        dag = DAG("companies", new=True)

        # Create source configs
        source_a = dag.source(
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

        source_b = dag.source(
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
            infer_types=True,
            key_field="id",
            index_fields=["company_name", "registration_id"],
        )

        # Dedupe steps
        dedupe_a = source_a.query(
            cleaning={
                "company_name": self._clean_company_name(source_a.f("company_name"))
            },
        ).deduper(
            name="dedupe_source_a",
            description="Deduplicate source A",
            model_class=NaiveDeduper,
            model_settings={"unique_fields": [source_a.f("registration_id")]},
        )

        dedupe_b = source_b.query(
            cleaning={
                "company_name": self._clean_company_name(source_b.f("company_name")),
            }
        ).deduper(
            name="dedupe_source_b",
            description="Deduplicate source B",
            model_class=NaiveDeduper,
            model_settings={"unique_fields": [source_b.f("registration_id")]},
        )

        # Link deduplicated sources A and B
        link_a_b = dedupe_a.query(
            source_a,
            cleaning={
                "company_name": self._clean_company_name(source_a.f("company_name")),
                "registration_id": source_a.f("registration_id"),
            },
        ).linker(
            dedupe_b.query(
                source_b,
                cleaning={
                    "company_name": self._clean_company_name(
                        source_b.f("company_name")
                    ),
                    "registration_id": source_b.f("registration_id"),
                },
            ),
            name="final",
            description="Link sources A and B on registration_id",
            model_class=DeterministicLinker,
            model_settings={"comparisons": ["l.registration_id = r.registration_id"]},
        )

        # === FIRST RUN ===
        logging.info("Running DAG for the first time")
        dag.run_and_sync()

        # Basic verification - we have some linked results and can retrieve them
        final_df = link_a_b.query(source_a, source_b).run()

        # # Should have linked results
        assert len(final_df) > 0, "Expected some results from first run"
        assert final_df["id"].n_unique() == len(self.linked_testkit.true_entities)

        first_run_entities = final_df["id"].n_unique()
        logging.info(f"First run produced {first_run_entities} unique entities")

        # Lookup works too
        matches = dag.lookup_key(
            from_source=source_b.name,
            to_sources=[source_a.name],
            key=final_df.filter(
                pl.col(source_b.f(source_b.config.key_field.name)).is_not_null()
            )[source_b.f(source_b.config.key_field.name)][0],
        )
        assert len(matches[source_a.name]) >= 1

        # === SECOND RUN (OVERWRITE) ===

        logging.info("Running DAG again to test overwriting")
        dag.run_and_sync()

        # Verify second run produces same results
        final_df_second = link_a_b.query(source_a, source_b).run()

        second_run_entities = final_df_second["id"].n_unique()
        logging.info(f"Second run produced {second_run_entities} unique entities")

        # Should have same number of entities after rerun
        assert first_run_entities == second_run_entities, (
            "Expected same results after rerun: "
            f"{first_run_entities} vs {second_run_entities}"
        )

        # # Basic sanity checks
        assert len(final_df_second) > 0, "Expected some results from second run"

        logging.info("DAG pipeline test completed successfully!")
