import logging

import pytest
from httpx import Client
from sqlalchemy import Engine, text

from matchbox import clean
from matchbox.client import _handler
from matchbox.client.helpers import delete_resolution
from matchbox.client.models import Model
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.queries import Query
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
            source_testkit.write_to_location()

        # Clear matchbox database before test
        response = matchbox_client.delete("/database", params={"certain": "true"})

        assert response.status_code == 200, "Failed to clear matchbox database"

        yield

        # Teardown code

        # Clean up database tables
        with postgres_warehouse.connect() as conn:
            # Drop all tables created by the test
            for source_name in self.linked_testkit.sources:
                conn.execute(text(f"DROP TABLE IF EXISTS {source_name};"))

            conn.commit()

        response = matchbox_client.delete("/database", params={"certain": "true"})
        assert response.status_code == 200, "Failed to clear matchbox database"

    def _get_cleaning_dict(
        self, source_prefix: str, columns: list[str]
    ) -> dict[str, str] | None:
        """Get cleaning dictionary for a source based on its columns.

        Mirrors the perturbations made to the company_name field in the
        linked_sources_factory testkit.
        """
        company_name_col = f"{source_prefix}company_name"

        if company_name_col in columns:
            return {
                company_name_col: rf"""
                    trim(
                        regexp_replace(
                            {company_name_col}, 
                            '\b(Limited|UK|Company)\b', 
                            '', 
                            'gi'
                        )
                    )
                """
            }

        return None

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
            source = source_testkit.source
            source.run()
            source.sync()
            logging.debug(f"Indexed source: {source.name}")

        # === DEDUPLICATION PHASE ===
        dedupers = {}
        deduper_names = {}

        for source_testkit in self.linked_testkit.sources.values():
            # Get source
            source = source_testkit.source

            # Query data from the source
            raw_df = Query(source).run()
            clusters = query_to_cluster_entities(
                query=raw_df,
                keys={source.name: source.qualified_key},
            )
            df = raw_df.drop(source.qualified_key)

            # Apply cleaning
            cleaning_dict = self._get_cleaning_dict(source.prefix, df.columns)

            # Get feature names with prefix for deduplication
            feature_names = [
                f"{source.prefix}{feature.name}" for feature in source_testkit.features
            ]

            # Create and run a deduper model
            deduper_name = f"deduper_{source.name}"
            deduper = Model(
                name=deduper_name,
                description=f"Deduplication of {source.name}",
                model_class=NaiveDeduper,
                model_settings={
                    "id": "id",
                    "unique_fields": feature_names,
                },
                left_query=Query(source_testkit.source, cleaning=cleaning_dict),
            )

            # Run the deduper and store results
            results = deduper.run()

            # Verify deduplication results match expected true entities
            identical, report = self.linked_testkit.diff_results(
                probabilities=results.probabilities,
                left_clusters=clusters,
                right_clusters=None,
                sources=[source.name],
                threshold=0,
            )

            assert identical, f"Deduplication of {source.name} failed: {report}"

            deduper.truth = 1.0
            deduper.sync()

            logging.debug(f"Successfully deduplicated {source.name}")

            # Store the deduper resolution name for later use
            dedupers[source.name] = deduper
            deduper_names[source.name] = deduper_name

        # === PAIRWISE LINKING PHASE ===
        linker_names = {}
        linkers = {}

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
            left_source = left_testkit.source
            right_source = right_testkit.source

            # Query deduplicated data
            left_query = Query(left_source, model=dedupers[left_source.name])
            left_raw_df = left_query.run()

            left_clusters = query_to_cluster_entities(
                query=left_raw_df,
                keys={left_source.name: left_source.qualified_key},
            )
            left_df = left_raw_df.drop(left_source.qualified_key)

            right_query = Query(right_source, model=dedupers[right_source.name])
            right_raw_df = right_query.run()
            right_clusters = query_to_cluster_entities(
                query=right_raw_df,
                keys={right_source.name: right_source.qualified_key},
            )
            right_df = right_raw_df.drop(right_source.qualified_key)

            # Apply cleaning
            left_cleaning_dict = self._get_cleaning_dict(
                left_source.prefix, left_df.columns
            )
            clean(left_df, left_cleaning_dict)

            right_cleaning_dict = self._get_cleaning_dict(
                right_source.prefix, right_df.columns
            )
            clean(right_df, right_cleaning_dict)

            # Build comparison clause
            comparison_clause = (
                f"l.{left_source.prefix}{common_field} "
                f"= r.{right_source.prefix}{common_field}",
            )

            # Create and run linker model
            linker_name = f"linker_{left_source.name}_{right_source.name}"
            linker = Model(
                name=linker_name,
                description=f"Linking {left_testkit.name} and {right_testkit.name}",
                model_class=DeterministicLinker,
                model_settings={
                    "left_id": "id",
                    "right_id": "id",
                    "comparisons": comparison_clause,
                },
                left_query=left_query,
                right_query=right_query,
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

            linker.truth = 1.0
            linker.sync()

            logging.debug(
                f"Successfully linked {left_testkit.name} and {right_testkit.name}"
            )

            # Store the linker resolution name for later use
            linker_names[(left_source.name, right_source.name)] = linker_name
            linkers[(left_source.name, right_source.name)] = linker

        # === FINAL LINKING PHASE ===
        # Now link the first linked pair (crn-duns) with the third source (cdms)
        crn_source = self.linked_testkit.sources["crn"].source
        duns_source = self.linked_testkit.sources["duns"].source
        cdms_source = self.linked_testkit.sources["cdms"].source
        first_pair = (crn_source.name, duns_source.name)

        # Query data from the first linked pair and the third source
        left_query = Query(crn_source, duns_source, model=linkers[first_pair])
        left_raw_df = left_query.run()
        left_clusters = query_to_cluster_entities(
            query=left_raw_df,
            keys={
                crn_source.name: crn_source.qualified_key,
                duns_source.name: duns_source.qualified_key,
            },
        )
        left_df = left_raw_df.drop(crn_source.qualified_key, duns_source.qualified_key)

        right_query = Query(cdms_source, model=dedupers[cdms_source.name])
        right_raw_df = right_query.run()
        right_clusters = query_to_cluster_entities(
            query=right_raw_df,
            keys={cdms_source.name: cdms_source.qualified_key},
        )
        right_df = right_raw_df.drop(cdms_source.qualified_key)

        # Apply cleaning
        left_cleaning_dict = self._get_cleaning_dict(crn_source.prefix, left_df.columns)

        right_cleaning_dict = self._get_cleaning_dict(
            cdms_source.prefix, right_df.columns
        )

        # Create and run final linker with the common "crn" field
        final_linker_name = "__DEFAULT__"
        final_linker = Model(
            name=final_linker_name,
            description="Final linking of all sources",
            model_class=DeterministicLinker,
            model_settings={
                "left_id": "id",
                "right_id": "id",
                "comparisons": [
                    f"l.{crn_source.prefix}crn = r.{cdms_source.prefix}crn"
                ],
            },
            left_query=left_query,
            right_query=right_query,
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

        final_linker.truth = 1.0
        final_linker.sync()

        logging.debug("Successfully linked all sources")

        # === FINAL VERIFICATION PHASE ===
        # Query the final linked data with specific fields
        crn_source = self.linked_testkit.sources["crn"].source
        duns_source = self.linked_testkit.sources["duns"].source
        cdms_source = self.linked_testkit.sources["cdms"].source

        # Get necessary field from each source
        final_df = Query(crn_source, duns_source, cdms_source, model=final_linker).run()

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
            final_df["id"].n_unique()
            == len(self.linked_testkit.true_entities)
            == self.n_true_entities
        ), (
            f"Expected {len(self.linked_testkit.true_entities)} unique entities, "
            f"got {final_df['id'].n_unique()}"
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

        # Delete some resolutions as if my experimental model wasn't good enough

        final_linker_name = "__DEFAULT__"
        crn_source_name = self.linked_testkit.sources["crn"].source.name

        counts = _handler.count_backend_items()
        source_config_count = counts["entities"]["sources"]
        model_count = counts["entities"]["models"]

        # Delete the final linker resolution
        delete_resolution(name=final_linker_name, certain=True)

        counts = _handler.count_backend_items()
        assert counts["entities"]["sources"] == source_config_count
        assert counts["entities"]["models"] == model_count - 1, (
            "Expected one less model after deleting the final linker"
        )

        # Delete a source resolution
        delete_resolution(name=crn_source_name, certain=True)

        counts = _handler.count_backend_items()
        assert counts["entities"]["sources"] == source_config_count - 1, (
            "Expected one less source after deleting crn source"
        )
        assert counts["entities"]["models"] == model_count - 4, (
            "Expected all CRN descendant models to be deleted"
        )

        logging.debug("E2E test completed successfully!")
