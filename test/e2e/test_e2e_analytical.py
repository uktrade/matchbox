import logging
from functools import partial

import pytest
from httpx import Client
from pandas import DataFrame
from sqlalchemy import Engine, text

from matchbox import index, make_model, process, query
from matchbox.client.clean import steps
from matchbox.client.clean.utils import cleaning_function
from matchbox.client.helpers import cleaner, cleaners, select
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.factories.entities import query_to_cluster_entities
from matchbox.common.factories.sources import (
    FeatureConfig,
    LinkedSourcesTestkit,
    SourceTestkitParameters,
    SuffixRule,
    linked_sources_factory,
)


@pytest.mark.docker
class TestE2EAnalyticalUser:
    """End to end tests for analytical user functionality."""

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
        # Store fixtures as class attributes for use in tests with self.*
        n_true_entities = 100

        self.__class__.client = matchbox_client
        self.__class__.warehouse_engine = postgres_warehouse
        self.__class__.n_true_entities = n_true_entities

        # Create feature configurations
        features = {
            "company_name": FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            "crn": FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
            "duns": FeatureConfig(
                name="duns",
                base_generator="numerify",
                parameters=(("text", "########"),),
            ),
            "cdms": FeatureConfig(
                name="cdms",
                base_generator="numerify",
                parameters=(("text", "ORG-########"),),
            ),
        }

        # Create source configurations that match our test fixtures
        source_parameters = (
            SourceTestkitParameters(
                name="crn",
                engine=postgres_warehouse,
                features=(
                    features["company_name"].add_variations(
                        SuffixRule(suffix=" Limited"),
                        SuffixRule(suffix=" UK"),
                        SuffixRule(suffix=" Company"),
                    ),
                    features["crn"],
                ),
                n_true_entities=n_true_entities,
                repetition=0,  # No duplicates within the variations
            ),
            SourceTestkitParameters(
                name="duns",
                engine=postgres_warehouse,
                features=(
                    features["company_name"],
                    features["duns"],
                ),
                n_true_entities=n_true_entities // 2,  # Half the companies
                repetition=0,
            ),
            SourceTestkitParameters(
                name="cdms",
                engine=postgres_warehouse,
                features=(
                    features["crn"],
                    features["cdms"],
                ),
                n_true_entities=n_true_entities,
                repetition=1,  # Duplicate all rows
            ),
        )

        # Create linked sources testkit with our configurations
        self.__class__.linked_testkit = linked_sources_factory(
            source_parameters=source_parameters,
            seed=42,  # For reproducibility
        )

        # Setup code - Create tables in warehouse
        for source_testkit in self.linked_testkit.sources.values():
            source_testkit.write_to_location(
                credentials=postgres_warehouse, set_credentials=True
            )

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})

        assert response.status_code == 200, "Failed to clear matchbox database"

        yield

        # Teardown code
        # Clean up database tables
        with postgres_warehouse.connect() as conn:
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))
            conn.commit()

        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    def test_e2e_deduplication_and_linking_pipeline(self):
        """Runs an end to end test of the entire entity resolution pipeline.

        1. Create synthetic test data with linked_sources_factory
        2. Index the data in PostgreSQL
        3. Deduplicate each source, verify results
        4. Link pairs of deduplicated sources, verify results
        5. Link the linked pairs to create a full graph, verify results
        6. Verify the final linked data matches the original entity structure
        """
        # === SETUP PHASE ===
        # Get all sources and true entities for later verification
        all_true_entities = list(self.linked_testkit.true_entities)

        # Basic validation of our test data
        assert len(self.linked_testkit.sources) == 3, "Expected 3 sources"
        assert len(all_true_entities) == self.n_true_entities

        # Index all sources in the PostgreSQL database
        for source_testkit in self.linked_testkit.sources.values():
            source = source_testkit.source_config
            index(source_config=source)
            logging.debug(f"Indexed source: {source.name}")

        # Helper functions
        # Define custom company name cleaner, mirroring the FeatureConfig
        remove_stopwords_redux = partial(
            steps.remove_stopwords, stopwords=["Limited", "UK", "Company"]
        )
        clean_company_name = cleaning_function(
            steps.tokenise,  # returns array
            remove_stopwords_redux,
            steps.list_join_to_string,  # returns column
            steps.trim,
        )

        def _clean_company_name(df: DataFrame, prefix: str) -> DataFrame:
            """Clean company_name feature in the source or model."""
            if any("company_name" in col for col in df.columns):
                company_cleaner = cleaner(
                    function=clean_company_name,
                    arguments={"column": f"{prefix}company_name"},
                )
                clean_pipeline = cleaners(company_cleaner)
                return process(data=df, pipeline=clean_pipeline)

            return df

        # === DEDUPLICATION PHASE ===
        deduper_names = {}

        for source_testkit in self.linked_testkit.sources.values():
            # Get source config
            source_config = source_testkit.source_config

            # Query data from the source
            # keys included then dropped to create ClusterEntity objects for later diff
            source_select = select(
                {
                    source_config.name: ["key"]
                    + [field.name for field in source_config.index_fields]
                },
                credentials=self.warehouse_engine,
            )
            raw_df = query(source_select, return_type="pandas")
            clusters = query_to_cluster_entities(
                query=raw_df,
                keys={source_config.name: source_config.qualified_key},
            )
            df = raw_df.drop(columns=[source_config.qualified_key])

            # Apply cleaning based on features in the source
            cleaned = _clean_company_name(df, source_config.prefix)

            # Get feature names with prefix for deduplication
            feature_names = [
                f"{source_config.prefix}{feature.name}"
                for feature in source_testkit.features
            ]

            # Create and run a deduper model
            deduper_name = f"deduper_{source_config.name}"
            deduper = make_model(
                name=deduper_name,
                description=f"Deduplication of {source_config.name}",
                model_class=NaiveDeduper,
                model_settings={
                    "id": "id",
                    "unique_fields": feature_names,
                },
                left_data=cleaned,
                left_resolution=source_testkit.source_config.name,
            )

            # Run the deduper and store results
            results = deduper.run()

            # Verify deduplication results match expected true entities
            identical, report = self.linked_testkit.diff_results(
                probabilities=results.probabilities,
                left_clusters=clusters,
                right_clusters=None,
                sources=[source_config.name],
                threshold=0,
            )

            assert identical, f"Deduplication of {source_config.name} failed: {report}"

            results.to_matchbox()
            deduper.truth = 1.0

            logging.debug(f"Successfully deduplicated {source_config.name}")

            # Store the deduper resolution name for later use
            deduper_names[source_config.name] = deduper_name

        # === PAIRWISE LINKING PHASE ===
        linker_names = {}

        # Define linking pairs based on common fields
        linking_pairs = [
            (
                self.linked_testkit.sources["crn"],
                self.linked_testkit.sources["duns"],
                "company_name",
            ),  # CRN-DUNS via company_name
            (
                self.linked_testkit.sources["crn"],
                self.linked_testkit.sources["cdms"],
                "crn",
            ),  # CRN-CDMS via crn
        ]

        for left_testkit, right_testkit, common_field in linking_pairs:
            # Get sources
            left_source = left_testkit.source_config
            right_source = right_testkit.source_config

            # Query deduplicated data
            # keys included then dropped to create ClusterEntity objects for later diff
            left_raw_df = query(
                select(
                    {left_source.name: ["key", common_field]},
                    credentials=self.warehouse_engine,
                ),
                resolution=deduper_names[left_source.name],
                return_type="pandas",
            )
            left_clusters = query_to_cluster_entities(
                query=left_raw_df,
                keys={left_source.name: left_source.qualified_key},
            )
            left_df = left_raw_df.drop(columns=[left_source.qualified_key])

            right_raw_df = query(
                select(
                    {right_source.name: ["key", common_field]},
                    credentials=self.warehouse_engine,
                ),
                resolution=deduper_names[right_source.name],
                return_type="pandas",
            )
            right_clusters = query_to_cluster_entities(
                query=right_raw_df,
                keys={right_source.name: right_source.qualified_key},
            )
            right_df = right_raw_df.drop(columns=[right_source.qualified_key])

            # Apply cleaning based on features in the sources
            left_cleaned = _clean_company_name(left_df, left_source.prefix)
            right_cleaned = _clean_company_name(right_df, right_source.prefix)

            # Build comparison clause
            comparison_clause = (
                f"l.{left_source.prefix}{common_field} "
                f"= r.{right_source.prefix}{common_field}"
            )

            # Create and run linker model
            linker_name = f"linker_{left_source.name}_{right_source.name}"
            linker = make_model(
                name=linker_name,
                description=f"Linking {left_testkit.name} and {right_testkit.name}",
                model_class=DeterministicLinker,
                model_settings={
                    "left_id": "id",
                    "right_id": "id",
                    "comparisons": comparison_clause,
                },
                left_data=left_cleaned,
                left_resolution=deduper_names[left_source.name],
                right_data=right_cleaned,
                right_resolution=deduper_names[right_source.name],
            )

            # Run the linker and store results
            results = linker.run()

            # Verify linking results
            identical, report = self.linked_testkit.diff_results(
                probabilities=results.probabilities,
                left_clusters=left_clusters,
                right_clusters=right_clusters,
                sources=[left_source.name, right_source.name],
                threshold=0,
            )

            report_counts = {
                k: len(v) if isinstance(v, list) else v for k, v in report.items()
            }

            assert identical, (
                f"Linking of {left_testkit.name} to {right_testkit.name} failed: "
                f"{report_counts}"
            )

            results.to_matchbox()
            linker.truth = 1.0

            logging.debug(
                f"Successfully linked {left_testkit.name} and {right_testkit.name}"
            )

            # Store the linker resolution name for later use
            linker_names[(left_source.name, right_source.name)] = linker_name

        # === FINAL LINKING PHASE ===
        # Now link the first linked pair (crn-duns) with the third source (cdms)
        crn_source = self.linked_testkit.sources["crn"].source_config
        duns_source = self.linked_testkit.sources["duns"].source_config
        cdms_source = self.linked_testkit.sources["cdms"].source_config
        first_pair = (crn_source.name, duns_source.name)

        # Query data from the first linked pair and the third source
        # keys included then dropped to create ClusterEntity objects for later diff
        left_raw_df = query(
            select(
                {crn_source.name: ["key", "crn"]}, credentials=self.warehouse_engine
            ),
            select({duns_source.name: ["key"]}, credentials=self.warehouse_engine),
            resolution=linker_names[first_pair],
            return_type="pandas",
        )
        left_clusters = query_to_cluster_entities(
            query=left_raw_df,
            keys={
                crn_source.name: crn_source.qualified_key,
                duns_source.name: duns_source.qualified_key,
            },
        )
        left_df = left_raw_df.drop(
            columns=[crn_source.qualified_key, duns_source.qualified_key]
        )

        right_raw_df = query(
            select(
                {cdms_source.name: ["key", "crn"]}, credentials=self.warehouse_engine
            ),
            resolution=deduper_names[cdms_source.name],
            return_type="pandas",
        )
        right_clusters = query_to_cluster_entities(
            query=right_raw_df, keys={cdms_source.name: cdms_source.qualified_key}
        )
        right_df = right_raw_df.drop(columns=[cdms_source.qualified_key])

        # Apply cleaning if needed
        left_cleaned = _clean_company_name(left_df, crn_source.prefix)
        right_cleaned = _clean_company_name(right_df, cdms_source.prefix)

        # Create and run final linker with the common "crn" field
        final_linker_name = "__DEFAULT__"
        final_linker = make_model(
            name=final_linker_name,
            description="Final linking of all sources",
            model_class=DeterministicLinker,
            model_settings={
                "left_id": "id",
                "right_id": "id",
                "comparisons": f"l.{crn_source.prefix}crn = r.{cdms_source.prefix}crn",
            },
            left_data=left_cleaned,
            left_resolution=linker_names[first_pair],
            right_data=right_cleaned,
            right_resolution=deduper_names[cdms_source.name],
        )

        # Run the final linker and store results
        results = final_linker.run()

        # Verify final linking results
        identical, report = self.linked_testkit.diff_results(
            probabilities=results.probabilities,
            left_clusters=left_clusters,
            right_clusters=right_clusters,
            sources=[crn_source.name, duns_source.name, cdms_source.name],
            threshold=0,
        )

        report_counts = {
            k: len(v) if isinstance(v, list) else v for k, v in report.items()
        }

        assert identical, f"Final linking failed: {report_counts}"

        results.to_matchbox()
        final_linker.truth = 1.0

        logging.debug("Successfully linked all sources")

        # === FINAL VERIFICATION PHASE ===
        # Query the final linked data with specific fields
        crn_source = self.linked_testkit.sources["crn"].source_config
        duns_source = self.linked_testkit.sources["duns"].source_config
        cdms_source = self.linked_testkit.sources["cdms"].source_config

        # Get necessary field from each source
        final_df = query(
            select(
                {
                    crn_source.name: ["key", "company_name", "crn"],
                    duns_source.name: ["key", "company_name", "duns"],
                    cdms_source.name: ["key", "crn", "cdms"],
                },
                credentials=self.warehouse_engine,
            ),
            resolution=final_linker_name,
            return_type="pandas",
        )

        final_clusters = query_to_cluster_entities(
            query=final_df,
            keys={
                crn_source.name: crn_source.qualified_key,
                duns_source.name: duns_source.qualified_key,
                cdms_source.name: cdms_source.qualified_key,
            },
        )

        # Verify the final data structure - number of unique entities
        assert (
            final_df["id"].nunique()
            == len(self.linked_testkit.true_entities)
            == self.n_true_entities
        ), (
            f"Expected {len(self.linked_testkit.true_entities)} unique entities, "
            f"got {final_df['id'].nunique()}"
        )

        # Verify the final cluster membership -- the golden check

        true_entities = {
            entity.to_cluster_entity(
                crn_source.name, duns_source.name, cdms_source.name
            )
            for entity in self.linked_testkit.true_entities
            if entity.to_cluster_entity(
                crn_source.name, duns_source.name, cdms_source.name
            )
            is not None
        }

        assert true_entities == set(final_clusters), "Final clusters do not match"

        logging.debug("E2E test completed successfully!")
