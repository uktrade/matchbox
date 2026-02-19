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
        resolver = dag.resolvers[f"resolver_{linker_name}"].resolver

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
        resolver = dag.resolvers[f"resolver_{linker_name}"].resolver

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

    def test_match_cluster_matches_query_projection(
        self,
        link_scenario: tuple[MatchboxPostgres, object],
    ) -> None:
        """Cluster returned by match should align with query projection IDs."""
        _, dag = link_scenario
        crn = dag.sources["crn"]
        dh = dag.sources["dh"]
        linker_name = "deterministic_naive_test_crn_naive_test_dh"
        resolver = dag.resolvers[f"resolver_{linker_name}"].resolver
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
