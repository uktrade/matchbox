"""PostgreSQL query utility tests for resolver-era semantics.

These tests are intentionally thin. The adapter query tests cover most behaviour across
backends; this module focuses only on PostgreSQL utility invariants that are easy to
regress while changing SQL composition.
"""

from collections.abc import Generator

import polars as pl
import pytest
from sqlalchemy import Engine

from matchbox.common.dtos import UploadStage
from matchbox.common.exceptions import MatchboxResolutionNotQueriable
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import Resolutions
from matchbox.server.postgresql.utils.query import match, query


@pytest.fixture(scope="function")
def link_scenario(
    matchbox_postgres: MatchboxPostgres,
    sqla_sqlite_warehouse: Engine,
) -> Generator[tuple[MatchboxPostgres, object], None, None]:
    """Create a resolver-rich scenario for query utility testing."""
    with setup_scenario(
        matchbox_postgres, "link", warehouse=sqla_sqlite_warehouse
    ) as dag:
        yield matchbox_postgres, dag


@pytest.mark.docker
class TestPostgresQueryUtilities:
    """PG-specific query utility checks."""

    def test_query_returns_stably_ordered_rows(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Rows should be ordered deterministically before applying limits."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        linker_name = "deterministic_naive_test_crn_naive_test_dh"
        resolver = dag.resolvers[f"resolver_{linker_name}"]

        source_rows = pl.from_arrow(
            query(source=crn.resolution_path, return_leaf_id=True)
        )
        assert source_rows.equals(source_rows.sort(["id", "leaf_id", "key"]))

        limited_rows = pl.from_arrow(
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
                return_leaf_id=True,
                limit=5,
            )
        )
        assert len(limited_rows) == 5
        assert limited_rows.equals(limited_rows.sort(["id", "leaf_id", "key"]))

    def test_query_rejects_model_as_point_of_truth(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Only resolver resolutions are queryable points of truth."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        model = dag.models["deterministic_naive_test_crn_naive_test_dh"]

        with pytest.raises(MatchboxResolutionNotQueriable):
            query(
                source=crn.resolution_path,
                point_of_truth=model.resolution_path,
            )

    def test_query_rejects_incomplete_resolver(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Resolvers with non-complete upload stage should not be queryable."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        linker_name = "deterministic_naive_test_crn_naive_test_dh"
        resolver = dag.resolvers[f"resolver_{linker_name}"]

        with MBDB.get_session() as session:
            resolver_orm = Resolutions.from_path(
                path=resolver.resolution_path,
                session=session,
                for_update=True,
            )
            resolver_orm.upload_stage = UploadStage.PROCESSING
            session.commit()

        with pytest.raises(MatchboxResolutionNotQueriable):
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
            )

    def test_query_accepts_direct_model_threshold_overrides(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Direct model overrides should be accepted on resolver queries."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        model_name = "probabilistic_naive_test_crn_naive_test_cdms"
        resolver = dag.resolvers[f"resolver_{model_name}"]

        rows = pl.from_arrow(
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
                threshold_overrides={model_name: 0},
                return_leaf_id=True,
            )
        )

        assert len(rows) == crn.data.num_rows
        assert rows.equals(rows.sort(["id", "leaf_id", "key"]))

    def test_query_rejects_resolver_name_override_keys(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Resolver input names are not valid threshold override keys."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        model_name = "probabilistic_naive_test_crn_naive_test_cdms"
        resolver = dag.resolvers[f"resolver_{model_name}"]
        resolver_input_name = "resolver_naive_test_crn"

        with pytest.raises(
            MatchboxResolutionNotQueriable,
            match="resolver inputs cannot be overridden",
        ):
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
                threshold_overrides={resolver_input_name: 0},
            )

    def test_query_rejects_non_direct_model_override_keys(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Override keys must be direct model inputs of the queried resolver."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        model_name = "probabilistic_naive_test_crn_naive_test_cdms"
        resolver = dag.resolvers[f"resolver_{model_name}"]

        with pytest.raises(
            MatchboxResolutionNotQueriable,
            match="unknown or non-direct model inputs",
        ):
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
                threshold_overrides={"final_join": 0},
            )

    def test_query_rejects_invalid_override_threshold_values(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Override thresholds must be backend ints in [0, 100]."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        model_name = "probabilistic_naive_test_crn_naive_test_cdms"
        resolver = dag.resolvers[f"resolver_{model_name}"]

        with pytest.raises(
            MatchboxResolutionNotQueriable, match="ints in \\[0, 100\\]"
        ):
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
                threshold_overrides={model_name: 101},
            )

    def test_query_rejects_overrides_without_resolver_point_of_truth(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Threshold overrides cannot be used on source-only queries."""
        _, dag = link_scenario
        crn = dag.sources["crn"]

        with pytest.raises(
            MatchboxResolutionNotQueriable,
            match="require a resolver point_of_truth",
        ):
            query(
                source=crn.resolution_path,
                threshold_overrides={"probabilistic_naive_test_crn_naive_test_cdms": 0},
            )

    def test_match_cluster_matches_query_projection(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Cluster returned by match should align with query projection IDs."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        dh = dag.sources["dh"]
        linker_name = "deterministic_naive_test_crn_naive_test_dh"
        resolver = dag.resolvers[f"resolver_{linker_name}"]
        linked = dag.source_to_linked["crn"]

        entity = linked.find_entities(
            min_appearances={"crn": 1, "dh": 1},
        )[0]
        source_key = next(iter(entity.keys["crn"]))

        matched = match(
            key=source_key,
            source=crn.resolution_path,
            targets=[dh.resolution_path],
            point_of_truth=resolver.resolution_path,
        )
        assert len(matched) == 1
        assert matched[0].cluster is not None

        crn_rows = pl.from_arrow(
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
            )
        )
        expected_cluster = int(
            crn_rows.filter(pl.col("key") == source_key).select("id").item()
        )
        assert matched[0].cluster == expected_cluster

        if matched[0].target_id:
            dh_rows = pl.from_arrow(
                query(
                    source=dh.resolution_path,
                    point_of_truth=resolver.resolution_path,
                )
            )
            target_clusters = (
                dh_rows.filter(pl.col("key").is_in(list(matched[0].target_id)))
                .select("id")
                .unique()
                .to_series()
                .to_list()
            )
            assert target_clusters == [expected_cluster]

    def test_match_cluster_matches_query_projection_with_overrides(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Override matching should stay consistent with override query projection."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        cdms = dag.sources["cdms"]
        model_name = "probabilistic_naive_test_crn_naive_test_cdms"
        resolver = dag.resolvers[f"resolver_{model_name}"]
        linked = dag.source_to_linked["crn"]
        overrides = {model_name: 0}

        entity = linked.find_entities(
            min_appearances={"crn": 1, "cdms": 1},
        )[0]
        source_key = next(iter(entity.keys["crn"]))

        matched = match(
            key=source_key,
            source=crn.resolution_path,
            targets=[cdms.resolution_path],
            point_of_truth=resolver.resolution_path,
            threshold_overrides=overrides,
        )
        assert len(matched) == 1
        assert matched[0].cluster is not None

        crn_rows = pl.from_arrow(
            query(
                source=crn.resolution_path,
                point_of_truth=resolver.resolution_path,
                threshold_overrides=overrides,
            )
        )
        expected_cluster = int(
            crn_rows.filter(pl.col("key") == source_key).select("id").item()
        )
        assert matched[0].cluster == expected_cluster

        if matched[0].target_id:
            cdms_rows = pl.from_arrow(
                query(
                    source=cdms.resolution_path,
                    point_of_truth=resolver.resolution_path,
                    threshold_overrides=overrides,
                )
            )
            target_clusters = (
                cdms_rows.filter(pl.col("key").is_in(list(matched[0].target_id)))
                .select("id")
                .unique()
                .to_series()
                .to_list()
            )
            assert target_clusters == [expected_cluster]
