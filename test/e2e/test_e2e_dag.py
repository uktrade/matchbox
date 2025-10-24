import logging

import polars as pl
import pytest
from httpx import Client
from polars.testing import assert_frame_equal
from sqlalchemy import Engine, text

from matchbox.client.dags import DAG
from matchbox.client.locations import RelationalDBLocation
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.sources import (
    FeatureConfig,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EPipelineBuilder:
    """End to end tests for DAG pipeline functionality."""

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
        # Persist shared setup for use in the test body
        n_true_entities = 10  # Keep it small for simplicity
        self.warehouse_engine = postgres_warehouse

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
        self.linked_testkit = linked_sources_factory(
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
        dw_loc = RelationalDBLocation(name="dbname").set_client(self.warehouse_engine)

        dag = DAG("companies").new_run()

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
                "company_name": self._clean_company_name(source_a.f("company_name")),
                "registration_id": source_a.f("registration_id"),
            },
        ).deduper(
            name="dedupe_source_a",
            description="Deduplicate source A",
            model_class=NaiveDeduper,
            model_settings={"unique_fields": ["registration_id"]},
        )

        dedupe_b = source_b.query(
            cleaning={
                "company_name": self._clean_company_name(source_b.f("company_name")),
                "registration_id": source_b.f("registration_id"),
            }
        ).deduper(
            name="dedupe_source_b",
            description="Deduplicate source B",
            model_class=NaiveDeduper,
            model_settings={"unique_fields": ["registration_id"]},
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
        test_key = final_df.filter(
            pl.col(source_b.f(source_b.config.key_field.name)).is_not_null()
        )[source_b.f(source_b.config.key_field.name)][0]

        matches = dag.lookup_key(
            from_source=source_b.name,
            to_sources=[source_a.name],
            key=test_key,
        )
        assert len(matches[source_a.name]) >= 1

        # Can retrieve whole lookup
        dag1_lookup = dag.extract_lookup()

        # Set as new default
        dag.set_default()

        # === SECOND RUN ===

        logging.info("Running DAG again to test downloading and using the default")

        # Load default
        reconstructed_dag = DAG("companies").load_default()
        assert reconstructed_dag.run == dag.run

        # Can directly read data from default
        assert matches == reconstructed_dag.lookup_key(
            from_source=source_b.name,
            to_sources=[source_a.name],
            key=test_key,
        )

        # Start a new run
        rerun_dag = reconstructed_dag.set_client(self.warehouse_engine).new_run()
        assert rerun_dag.run != dag.run
        rerun_dag.run_and_sync()

        # The lookup is identical
        assert_frame_equal(
            pl.from_arrow(rerun_dag.extract_lookup()),
            pl.from_arrow(dag1_lookup),
            check_column_order=False,
            check_row_order=False,
        )

        # Load pending to check we can
        pending_dag: DAG = (
            DAG("companies").load_pending().set_client(self.warehouse_engine)
        )
        assert pending_dag.run == rerun_dag.run

        logging.info("DAG pipeline test completed successfully!")
