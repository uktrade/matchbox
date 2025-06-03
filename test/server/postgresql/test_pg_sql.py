"""Test SQL functions for the Matchbox PostgreSQL backend.

The backend adapter tests provide the key coverage we want for any backend. However,
the PostgreSQL SQL functions are complex, and we've found having lower-level tests is
useful for debugging the complex logic required to query a hierarchical system.

Nevertheless, these tests are completely ephemeral, and if the hierarchical
representation changes, they should be rewritten in whatever form best-aids the
development of the new query functions. Don't be precious.
"""

from typing import Generator

import pytest
from sqlalchemy import and_

from matchbox.common.db import sql_to_df
from matchbox.common.exceptions import (
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
    SourceFields,
)
from matchbox.server.postgresql.utils.db import compile_sql
from matchbox.server.postgresql.utils.query import (
    _build_model_query,
    _build_source_query,
    _build_unified_query,
    _empty_result,
    _get_resolution_priority,
    _resolve_cluster_hierarchy,
    _resolve_hierarchy_assignments,
    _resolve_thresholds,
    get_clusters_with_leaves,
    get_source_config,
    match,
    query,
)


@pytest.fixture(scope="function")
def populated_postgres_db(
    matchbox_postgres: MatchboxPostgres,
) -> Generator[MatchboxPostgres, None, None]:
    """PostgreSQL database with a rich yet simple test dataset.

    - Source A: 6 keys → 5 clusters (keys 1&2 share cluster 101)
    - Source B: 5 keys → 5 clusters (one key per cluster)
    - Dedupe A: Creates
        * 80% cluster (301=101+102)
        * 70% 3-way cluster (302=101+102+103) with pairwise (303=101+103)
    - Dedupe B: Creates simple 70% cluster (401=201+202)
    - Linker: Caches truth DA=80, DB=70. Creates:
        * Cluster 503, which shows connected dedupe outputs and dedupe+raw mix:
            * At 90%, pairwise component 501=301+401=101+102+201+202
            * At 80%, pairwise 502=301+205=101+102+205, which forms component
                503=301+205+401=101+102+201+202+205
        * Cluster 504, which mixes raw leaves:
            * At 90%, pairwise component 504=103+203 (103 undeduped at DA=80)

    This diagram shows the structure of the test dataset. Each node shows the pairwise
    clusters that would be formed by a model's results, but can also be traced back to
    the sources to show the root-leaf relationships that are created. For example,
    cluster 502 is stored only so we can recover pairs, because 503 is the true cluster.

    The diagram also shows query results: if we query Dedupe A at 70%, clusters 101,
    102, and 103 would all map to component 302 (the highest cluster containing them
    at ≥70%), while 104 and 105 would return as themselves since they're not
    processed by Dedupe A.

    Tests: threshold filtering, role flags, complex hierarchy, cross-source linking,
    truth inheritance.

    ```mermaid
    graph TD
        %% Legend showing resolution hierarchy
        subgraph Legend["🔄 Resolution Subgraph"]
            SA_key["🔵 Source A"]
            SB_key["🟠 Source B"]
            DA_key["🟢 Dedupe A"]
            DB_key["🟣 Dedupe B"]
            L_key["🔴 Linker (caches: DA=80%, DB=70%)"]

            DA_key --> SA_key
            DB_key --> SB_key
            L_key --> DA_key
            L_key --> DB_key
        end

        %% Main cluster formation tree
        subgraph ClusterTree["📊 Data Subgraph"]
            %% Final Linker Components
            C503["503 @80%"]
            C504["504 @90%"]

            %% Linker Pairwise Formation
            C503 --> C501["501 @90%"]
            C503 --> C502["502 @80%"]

            %% Dedupe Formation - Used by Linker
            C501 --> C301["301 @80%"]
            C501 --> C401["401 @70%"]

            C502 --> C301
            C502 --> C205["205"]

            %% Raw leaves for C504
            C504 --> C103["103"]
            C504 --> C203["203"]

            %% Dedupe Sub-formation - Used clusters
            C301 --> C101["101, two keys"]
            C301 --> C102["102"]

            C401 --> C201["201"]
            C401 --> C202["202"]

            %% Additional Dedupe A clusters
            %% (below cache threshold - not connected to linker)
            C302["302 @70%"]
            C303["303 @70%"]

            %% Hierarchical formation: 302 comprises 301 + 303
            C302 --> C301
            C302 --> C303

            %% 303 pairwise formation
            C303 --> C101
            C303 --> C103

            %% Disconnected source clusters
            %% (not processed by any dedupe/linker)
            C104["104"]
            C105["105"]
            C204["204"]
        end

        %% Styling by source/model
        classDef source_a fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
        classDef source_b fill:#fff3e0,stroke:#f57c00,stroke-width:2px
        classDef dedupe_a fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
        classDef dedupe_b fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
        classDef linker fill:#ffebee,stroke:#d32f2f,stroke-width:4px
        classDef legend fill:#f9f9f9,stroke:#666666,stroke-width:1px

        class C101,C102,C103,C104,C105 source_a
        class C201,C202,C203,C204,C205 source_b
        class C301,C302,C303 dedupe_a
        class C401 dedupe_b
        class C501,C502,C503,C504 linker
        class SA_key source_a
        class SB_key source_b
        class DA_key dedupe_a
        class DB_key dedupe_b
        class L_key linker
    ```
    """

    with MBDB.get_session() as session:
        # === RESOLUTIONS ===
        resolutions = [
            Resolutions(resolution_id=1, name="source_a", type="source", truth=None),
            Resolutions(resolution_id=2, name="source_b", type="source", truth=None),
            Resolutions(resolution_id=3, name="dedupe_a", type="model", truth=80),
            Resolutions(resolution_id=4, name="dedupe_b", type="model", truth=70),
            Resolutions(resolution_id=5, name="linker_ab", type="model", truth=90),
        ]

        # === RESOLUTION LINEAGE ===
        lineage = [
            # Source A -> Dedupe A
            ResolutionFrom(parent=1, child=3, level=1, truth_cache=None),
            # Source B -> Dedupe B
            ResolutionFrom(parent=2, child=4, level=1, truth_cache=None),
            # Dedupe A -> Linker
            ResolutionFrom(parent=3, child=5, level=1, truth_cache=80),
            # Dedupe B -> Linker
            ResolutionFrom(parent=4, child=5, level=1, truth_cache=70),
            # Source A -> Linker (indirect)
            ResolutionFrom(parent=1, child=5, level=2, truth_cache=None),
            # Source B -> Linker (indirect)
            ResolutionFrom(parent=2, child=5, level=2, truth_cache=None),
        ]

        # === SOURCE CONFIGS ===
        source_configs = [
            SourceConfigs(
                source_config_id=11,
                resolution_id=1,
                location_type="test",
                location_uri="test://source_a",
                extract_transform="identity",
            ),
            SourceConfigs(
                source_config_id=22,
                resolution_id=2,
                location_type="test",
                location_uri="test://source_b",
                extract_transform="identity",
            ),
        ]

        # === SOURCE FIELDS ===
        source_fields = [
            # Source A fields
            SourceFields(
                field_id=1,
                source_config_id=11,
                index=0,
                name="key",
                type="TEXT",
                is_key=True,
            ),
            SourceFields(
                field_id=2,
                source_config_id=11,
                index=1,
                name="value",
                type="TEXT",
                is_key=False,
            ),
            # Source B fields
            SourceFields(
                field_id=3,
                source_config_id=22,
                index=0,
                name="key",
                type="TEXT",
                is_key=True,
            ),
            SourceFields(
                field_id=4,
                source_config_id=22,
                index=1,
                name="value",
                type="TEXT",
                is_key=False,
            ),
        ]

        # === CLUSTERS ===
        clusters = [
            # Source A clusters (100-series)
            Clusters(cluster_id=101, cluster_hash=b"hash_101"),
            Clusters(cluster_id=102, cluster_hash=b"hash_102"),
            Clusters(cluster_id=103, cluster_hash=b"hash_103"),
            Clusters(cluster_id=104, cluster_hash=b"hash_104"),
            Clusters(cluster_id=105, cluster_hash=b"hash_105"),
            # Source B clusters (200-series)
            Clusters(cluster_id=201, cluster_hash=b"hash_201"),
            Clusters(cluster_id=202, cluster_hash=b"hash_202"),
            Clusters(cluster_id=203, cluster_hash=b"hash_203"),
            Clusters(cluster_id=204, cluster_hash=b"hash_204"),
            Clusters(cluster_id=205, cluster_hash=b"hash_205"),
            # Dedupe A clusters (300-series)
            Clusters(cluster_id=301, cluster_hash=b"hash_301"),
            Clusters(cluster_id=302, cluster_hash=b"hash_302"),
            Clusters(cluster_id=303, cluster_hash=b"hash_303"),
            # Dedupe B clusters (400-series)
            Clusters(cluster_id=401, cluster_hash=b"hash_401"),
            # Linker clusters (500-series)
            Clusters(cluster_id=501, cluster_hash=b"hash_501"),
            Clusters(cluster_id=502, cluster_hash=b"hash_502"),
            Clusters(cluster_id=503, cluster_hash=b"hash_503"),
            Clusters(cluster_id=504, cluster_hash=b"hash_504"),
        ]

        # === CLUSTER SOURCE KEYS ===
        cluster_keys = [
            # Source A: 6 keys → 5 clusters (key1,key2 share cluster 101)
            ClusterSourceKey(
                key_id=1, cluster_id=101, source_config_id=11, key="src_a_key1"
            ),
            ClusterSourceKey(
                key_id=2, cluster_id=101, source_config_id=11, key="src_a_key2"
            ),
            ClusterSourceKey(
                key_id=3, cluster_id=102, source_config_id=11, key="src_a_key3"
            ),
            ClusterSourceKey(
                key_id=4, cluster_id=103, source_config_id=11, key="src_a_key4"
            ),
            ClusterSourceKey(
                key_id=5, cluster_id=104, source_config_id=11, key="src_a_key5"
            ),
            ClusterSourceKey(
                key_id=6, cluster_id=105, source_config_id=11, key="src_a_key6"
            ),
            # Source B: 5 keys → 5 clusters (one key per cluster)
            ClusterSourceKey(
                key_id=7, cluster_id=201, source_config_id=22, key="src_b_key1"
            ),
            ClusterSourceKey(
                key_id=8, cluster_id=202, source_config_id=22, key="src_b_key2"
            ),
            ClusterSourceKey(
                key_id=9, cluster_id=203, source_config_id=22, key="src_b_key3"
            ),
            ClusterSourceKey(
                key_id=10, cluster_id=204, source_config_id=22, key="src_b_key4"
            ),
            ClusterSourceKey(
                key_id=11, cluster_id=205, source_config_id=22, key="src_b_key5"
            ),
        ]

        # === CONTAINS RELATIONSHIPS ===
        contains = [
            # Dedupe A: C301 contains C101+C102 (80% cluster)
            Contains(root=301, leaf=101),
            Contains(root=301, leaf=102),
            # Dedupe A: C302 contains C101+C102+C103 (70% component)
            Contains(root=302, leaf=101),
            Contains(root=302, leaf=102),
            Contains(root=302, leaf=103),
            # Dedupe A: C303 contains C101+C103 (70% pairwise)
            Contains(root=303, leaf=101),
            Contains(root=303, leaf=103),
            # Dedupe B: C401 contains C201+C202 (70% cluster)
            Contains(root=401, leaf=201),
            Contains(root=401, leaf=202),
            # Linker: C501 links C301+C401 (90% pairwise)
            Contains(root=501, leaf=101),  # from C301
            Contains(root=501, leaf=102),  # from C301
            Contains(root=501, leaf=201),  # from C401
            Contains(root=501, leaf=202),  # from C401
            # Linker: C502 contains C301+C205 (80% pairwise)
            Contains(root=502, leaf=101),  # from C301
            Contains(root=502, leaf=102),  # from C301
            Contains(root=502, leaf=205),  # direct source leaf
            # Linker: C503 contains C501+C502 (80% component)
            Contains(root=503, leaf=101),  # from C301 (via C501 and C502)
            Contains(root=503, leaf=102),  # from C301 (via C501 and C502)
            Contains(root=503, leaf=201),  # from C401 (via C501)
            Contains(root=503, leaf=202),  # from C401 (via C501)
            Contains(root=503, leaf=205),  # direct (via C502)
            # Linker: C504 contains C103+C203 (90% raw leaves)
            Contains(root=504, leaf=103),  # direct source leaf
            Contains(root=504, leaf=203),  # direct source leaf
        ]

        # === PROBABILITIES ===
        probabilities = [
            # Dedupe A probabilities
            Probabilities(
                resolution=3, cluster=301, probability=80, role_flag=1
            ),  # both
            Probabilities(
                resolution=3, cluster=302, probability=70, role_flag=2
            ),  # component
            Probabilities(
                resolution=3, cluster=303, probability=70, role_flag=0
            ),  # pairwise
            # Dedupe B probabilities
            Probabilities(
                resolution=4, cluster=401, probability=70, role_flag=1
            ),  # both
            # Linker probabilities
            Probabilities(
                resolution=5, cluster=501, probability=90, role_flag=0
            ),  # pairwise
            Probabilities(
                resolution=5, cluster=502, probability=80, role_flag=0
            ),  # pairwise
            Probabilities(
                resolution=5, cluster=503, probability=80, role_flag=2
            ),  # component
            Probabilities(
                resolution=5, cluster=504, probability=90, role_flag=1
            ),  # both
        ]

        # Insert all objects
        for obj in (
            resolutions,
            lineage,
            source_configs,
            source_fields,
            clusters,
            cluster_keys,
            contains,
            probabilities,
        ):
            if isinstance(obj, list):
                session.add_all(obj)
                session.flush()

        session.commit()

    yield matchbox_postgres


@pytest.mark.docker
class TestGetSourceConfig:
    """Test source config retrieval."""

    def test_get_existing_source(self, populated_postgres_db: MatchboxPostgres):
        """Should return source config for existing source."""
        with MBDB.get_session() as session:
            source_config = get_source_config("source_a", session)
            assert source_config.source_config_id == 11
            assert source_config.resolution_id == 1

    def test_get_nonexistent_source(self, populated_postgres_db: MatchboxPostgres):
        """Should raise exception for nonexistent source."""
        with MBDB.get_session() as session:
            with pytest.raises(MatchboxSourceNotFoundError):
                get_source_config("nonexistent", session)


@pytest.mark.docker
class TestResolveThresholds:
    """Test threshold resolution logic."""

    def test_resolve_with_no_threshold(self, populated_postgres_db: MatchboxPostgres):
        """Should use default truth values when no threshold provided."""
        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 5)  # linker_ab
            lineage_truths = {1: None, 3: 80, 4: 70, 5: 90}  # mix of sources and models

            result = _resolve_thresholds(lineage_truths, resolution, threshold=None)

            assert result[1] is None  # source keeps None
            assert result[3] == 80  # model keeps default
            assert result[4] == 70  # model keeps default
            assert result[5] == 90  # target resolution keeps default

    def test_resolve_with_threshold_override(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should override target resolution threshold only."""
        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 5)  # linker_ab
            lineage_truths = {3: 80, 4: 70, 5: 90}

            result = _resolve_thresholds(lineage_truths, resolution, threshold=85)

            assert result[3] == 80  # parent keeps default
            assert result[4] == 70  # parent keeps default
            assert result[5] == 85  # target gets override


@pytest.mark.docker
class TestGetResolutionPriority:
    """Test resolution priority calculation."""

    def test_direct_parent_priority(self, populated_postgres_db: MatchboxPostgres):
        """Should return level 1 for direct parent."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # dedupe_a is level 1 parent of linker
            priority = _get_resolution_priority(3, linker_res)  # dedupe_a -> linker
            assert priority == 1

    def test_indirect_parent_priority(self, populated_postgres_db: MatchboxPostgres):
        """Should return level 2 for indirect parent."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # source_a is level 2 parent of linker (through dedupe_a)
            priority = _get_resolution_priority(1, linker_res)  # source_a -> linker
            assert priority == 2

    def test_nonexistent_relationship(self, populated_postgres_db: MatchboxPostgres):
        """Should return 0 for non-parent resolution."""
        with MBDB.get_session() as session:
            source_res = session.get(Resolutions, 1)  # source_a

            # dedupe_b has no relationship to source_a
            priority = _get_resolution_priority(4, source_res)  # dedupe_b -> source_a
            assert priority == 0


@pytest.mark.docker
class TestEmptyResult:
    """Test empty result generation."""

    def test_empty_result_structure(self, populated_postgres_db: MatchboxPostgres):
        """Should return empty result with correct columns."""
        query = _empty_result()

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        assert len(result) == 0
        expected_columns = {
            "root_id",
            "root_hash",
            "leaf_id",
            "leaf_hash",
            "leaf_key",
            "source_config_id",
        }
        assert set(result.columns) == expected_columns


class TestBuildSourceQuery:
    """Test source query building."""

    def test_build_simple_source_query(self, populated_postgres_db: MatchboxPostgres):
        """Should build query for source resolutions."""
        # Simple condition: source_config belongs to resolution 1 (source_a)
        source_conditions = [
            (SourceConfigs.resolution_id == 1, 999, 1)  # condition, priority, res_id
        ]
        source_config_filter = True  # no filtering

        query = _build_source_query(source_conditions, source_config_filter)

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return 6 keys from source_a, root_id == leaf_id
        assert len(result) == 6
        assert all(result["root_id"] == result["leaf_id"])
        expected_keys = {
            "src_a_key1",
            "src_a_key2",
            "src_a_key3",
            "src_a_key4",
            "src_a_key5",
            "src_a_key6",
        }
        assert set(result["leaf_key"]) == expected_keys


@pytest.mark.docker
class TestBuildModelQuery:
    """Test model query building."""

    def test_build_simple_model_query(self, populated_postgres_db: MatchboxPostgres):
        """Should build query for model resolutions."""
        # Condition: probabilities from resolution 3 (dedupe_a) with prob >= 80
        model_conditions = [
            (
                and_(
                    Probabilities.resolution == 3,
                    Probabilities.role_flag >= 1,
                    Probabilities.probability >= 80,
                ),
                1,
                3,
            )  # condition, priority, res_id
        ]
        source_config_filter = SourceConfigs.resolution_id == 1  # only source_a

        query = _build_model_query(model_conditions, source_config_filter)

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return keys that belong to clusters with prob >= 80
        # Only C301 qualifies (prob=80), which contains C101+C102 (3 keys)
        assert len(result) > 0
        # Check that we have hierarchy - some root_ids should be different from leaf_ids
        hierarchical_rows = result.filter(result["root_id"] != result["leaf_id"])
        assert len(hierarchical_rows) > 0


@pytest.mark.docker
class TestBuildUnifiedQuery:
    """Test unified query building."""

    def test_build_unified_source_only(self, populated_postgres_db: MatchboxPostgres):
        """Should build unified query for source-only scenario."""
        resolved_thresholds = {1: None}  # source_a only
        source_config_filter = True

        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 1)
            query = _build_unified_query(
                resolved_thresholds, source_config_filter, resolution
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return all 6 keys from source_a
        assert len(result) == 6
        assert all(result["root_id"] == result["leaf_id"])

    def test_build_unified_mixed_scenario(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should build unified query mixing sources and models."""
        resolved_thresholds = {
            1: None,  # source_a
            3: 80,  # dedupe_a with threshold 80
        }
        source_config_filter = True

        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 3)  # dedupe_a context
            query = _build_unified_query(
                resolved_thresholds, source_config_filter, resolution
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return keys, with some potentially mapped to dedupe clusters
        assert len(result) > 0
        assert "root_id" in result.columns
        assert "leaf_key" in result.columns

    def test_build_unified_linker_scenario(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should build unified query for complex linker scenario."""
        resolved_thresholds = {
            1: None,  # source_a
            2: None,  # source_b
            3: 80,  # dedupe_a cached
            4: 70,  # dedupe_b cached
            5: 85,  # linker with threshold 85
        }
        source_config_filter = True

        with MBDB.get_session() as session:
            resolution = session.get(Resolutions, 5)  # linker context
            query = _build_unified_query(
                resolved_thresholds, source_config_filter, resolution
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn, "polars")

        # Should return all 11 keys from both sources
        assert len(result) == 11

        # Should have linker clusters as roots for some keys
        root_ids = set(result["root_id"])

        # At threshold 85, only C504 (prob=90, role_flag=1) should qualify from linker
        assert 504 in root_ids  # C504 qualifies: 90% >= 85% and role_flag=1
        assert 503 not in root_ids  # C503 excluded: 80% < 85%

        # Should still see dedupe clusters from cached thresholds
        assert 301 in root_ids  # from dedupe_a cached=80
        assert 401 in root_ids  # from dedupe_b cached=70


@pytest.mark.docker
class TestResolveHierarchyAssignments:
    """Test hierarchy assignment resolution."""

    def test_source_only_resolution(self, populated_postgres_db: MatchboxPostgres):
        """Should return direct cluster assignments for source resolution."""
        with MBDB.get_session() as session:
            source_res = session.get(Resolutions, 1)  # source_a

            query = _resolve_hierarchy_assignments(source_res, None, None)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Source A has 6 keys, root_id should equal leaf_id for sources
            assert len(result) == 6
            assert all(result["root_id"] == result["leaf_id"])
            expected_keys = {
                "src_a_key1",
                "src_a_key2",
                "src_a_key3",
                "src_a_key4",
                "src_a_key5",
                "src_a_key6",
            }
            assert set(result["leaf_key"]) == expected_keys

    def test_linker_with_high_threshold(self, populated_postgres_db: MatchboxPostgres):
        """Should filter linker clusters only, using cached truth for parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # Test with threshold=95 - should exclude all linker clusters (max is 90)
            # But parents should still use cached values: dedupe_a=80, dedupe_b=70
            query = _resolve_hierarchy_assignments(linker_res, None, threshold=95)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 11 keys (6 from source_a + 5 from source_b)
            assert len(result) == 11

            # No linker clusters qualify, so keys should fall back to parent assignments
            # From dedupe_a (cached truth=80): C301 should still be used
            # From dedupe_b (cached truth=70): C401 should still be used
            root_ids = set(result["root_id"])
            assert 301 in root_ids  # dedupe_a's cluster (uses cached truth=80)
            assert 401 in root_ids  # dedupe_b's cluster (uses cached truth=70)

            # Linker clusters should NOT appear as roots
            linker_cluster_ids = {501, 502, 503, 504}
            assert linker_cluster_ids.isdisjoint(root_ids)

    def test_linker_with_medium_threshold(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should include some linker clusters, parents still use cached truth."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # Test with threshold=85 - should include C504 (prob=90) but exclude others
            # Parents still use cached: dedupe_a=80, dedupe_b=70
            query = _resolve_hierarchy_assignments(linker_res, None, threshold=85)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 11 keys
            assert len(result) == 11

            root_ids = set(result["root_id"])

            # From linker (threshold=85): only C504 qualifies (prob=90, flag=1)
            assert 504 in root_ids

            # Other linker clusters should not appear
            excluded_linker_clusters = {
                501,
                502,
                503,
            }  # C501: flag=0, C502: 80%<85%, C503: 80%<85%
            assert excluded_linker_clusters.isdisjoint(root_ids)

            # Parents should still contribute their cached-truth clusters
            assert 301 in root_ids  # from dedupe_a (cached=80)
            assert 401 in root_ids  # from dedupe_b (cached=70)

    def test_linker_with_low_threshold(self, populated_postgres_db: MatchboxPostgres):
        """Should include most linker clusters at low threshold."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            # Test with threshold=80
            # Should include C503 (80, flag=2) and C504 (90, flag=1)
            # C501 (90, flag=0) and C502 (80, flag=0) excluded by role_flag >= 1 filter
            query = _resolve_hierarchy_assignments(linker_res, None, threshold=80)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 11 keys
            assert len(result) == 11

            root_ids = set(result["root_id"])

            # From linker: C503 (prob=80, flag=2) and C504 (prob=90, flag=1)
            assert 503 in root_ids
            assert 504 in root_ids

            # C501 and C502 have flag=0, so excluded by role_flag >= 1 filter
            pairwise_only_clusters = {501, 502}
            assert pairwise_only_clusters.isdisjoint(root_ids)

            # C503 captures ALL dedupe cluster keys (101,102,201,202) plus 205
            # So NEITHER C301 nor C401 should appear as roots
            assert 301 not in root_ids  # C301's keys captured by C503
            assert 401 not in root_ids  # C401's keys captured by C503

            # Only uncaptured source clusters should remain
            remaining_source_clusters = {104, 105, 204}
            assert remaining_source_clusters.issubset(root_ids)


@pytest.mark.docker
class TestResolveClusterHierarchy:
    """Test cluster hierarchy resolution for specific source."""

    def test_same_resolution_passthrough(self, populated_postgres_db: MatchboxPostgres):
        """Should return direct mapping when truth resolution equals source."""
        with MBDB.get_session() as session:
            source_config = session.get(SourceConfigs, 11)  # source_a
            source_res = session.get(Resolutions, 1)  # source_a resolution

            query = _resolve_cluster_hierarchy(source_config, source_res, None)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            # Should return all 6 keys with their original cluster IDs
            assert len(result) == 6
            # For same resolution, id should equal original cluster assignments
            expected_pairs = {
                ("src_a_key1", 101),
                ("src_a_key2", 101),
                ("src_a_key3", 102),
                ("src_a_key4", 103),
                ("src_a_key5", 104),
                ("src_a_key6", 105),
            }
            actual_pairs = {
                (row["key"], row["id"]) for row in result.iter_rows(named=True)
            }
            assert actual_pairs == expected_pairs

    def test_hierarchy_through_model(self, populated_postgres_db: MatchboxPostgres):
        """Should resolve hierarchy through model resolution."""
        with MBDB.get_session() as session:
            source_config = session.get(SourceConfigs, 11)  # source_a
            dedupe_res = session.get(Resolutions, 3)  # dedupe_a

            query = _resolve_cluster_hierarchy(source_config, dedupe_res, threshold=80)

            with MBDB.get_adbc_connection() as conn:
                result = sql_to_df(compile_sql(query), conn, "polars")

            print(f"Result: {len(result)} rows")
            print("Key assignments:")
            for row in result.iter_rows(named=True):
                print(f"  {row['key']} -> cluster {row['id']}")

            # At threshold 80, only C301 qualifies (prob=80), which contains
            # clusters 101+102
            # Keys from clusters 101+102: src_a_key1, src_a_key2, src_a_key3
            # Keys 1,2,3 should map to C301, others keep original clusters
            c301_keys = result.filter(result["id"] == 301)
            original_keys = result.filter(result["id"] != 301)

            assert len(c301_keys) == 3  # Keys that qualify for dedupe
            expected_301_keys = {"src_a_key1", "src_a_key2", "src_a_key3"}
            assert set(c301_keys["key"]) == expected_301_keys

            assert len(original_keys) == 3  # Keys that don't qualify
            expected_original_keys = {"src_a_key4", "src_a_key5", "src_a_key6"}
            assert set(original_keys["key"]) == expected_original_keys


@pytest.mark.docker
class TestGetClustersWithLeaves:
    """Test cluster-leaf relationship extraction."""

    def test_get_clusters_for_model(self, populated_postgres_db: MatchboxPostgres):
        """Should return cluster hierarchy for model's parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Linker has parents dedupe_a and dedupe_b,
            # should get their cluster assignments
            assert len(result) > 0

            # Should contain cluster info with leaves
            for _, cluster_info in result.items():
                assert "root_hash" in cluster_info
                assert "leaves" in cluster_info
                assert isinstance(cluster_info["leaves"], list)

                # Each leaf should have required fields
                for leaf in cluster_info["leaves"]:
                    assert "leaf_id" in leaf
                    assert "leaf_hash" in leaf

    def test_get_clusters_for_deduper(self, populated_postgres_db: MatchboxPostgres):
        """Should return cluster assignments from deduper's parent (source)."""
        with MBDB.get_session() as session:
            dedupe_res = session.get(Resolutions, 3)  # dedupe_a

            result = get_clusters_with_leaves(dedupe_res)

            # Dedupe_a has parent source_a, so should get source_a's cluster assignments
            # Source assignments are 1:1 (each key maps to its own cluster)
            assert len(result) == 5  # source_a has 5 clusters (101-105)

            # Verify we have the expected source clusters
            expected_clusters = {101, 102, 103, 104, 105}
            actual_clusters = set(result.keys())
            assert actual_clusters == expected_clusters

            # Each source cluster should have exactly one leaf (itself)
            for cluster_id, cluster_info in result.items():
                assert len(cluster_info["leaves"]) == 1
                leaf = cluster_info["leaves"][0]
                assert leaf["leaf_id"] == cluster_id  # Source: leaf_id == root_id
                assert leaf["leaf_hash"] == cluster_info["root_hash"]

    def test_get_clusters_for_linker_specific_clusters(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should return specific cluster assignments from linker's parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Should include clusters from both dedupe_a and dedupe_b parents
            # Based on their default thresholds (cached truth values)

            # From dedupe_a (cached truth=80): should include C301 (prob=80)
            assert 301 in result, (
                f"C301 missing from linker's parent clusters: {result.keys()}"
            )

            # From dedupe_b (cached truth=70): should include C401 (prob=70)
            assert 401 in result, (
                f"C401 missing from linker's parent clusters: {result.keys()}"
            )

            # Verify C301's leaves (should contain clusters 101, 102)
            c301_leaves = result[301]["leaves"]
            c301_leaf_ids = {leaf["leaf_id"] for leaf in c301_leaves}
            assert c301_leaf_ids == {101, 102}, (
                f"C301 leaves incorrect: {c301_leaf_ids}"
            )

            # Verify C401's leaves (should contain clusters 201, 202)
            c401_leaves = result[401]["leaves"]
            c401_leaf_ids = {leaf["leaf_id"] for leaf in c401_leaves}
            assert c401_leaf_ids == {201, 202}, (
                f"C401 leaves incorrect: {c401_leaf_ids}"
            )

    def test_get_clusters_excludes_low_probability(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should exclude clusters that don't meet parent's cached threshold."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Should NOT include C302 (prob=70, flag=2) or C303 (prob=70, flag=0)
            # from dedupe_a because they're below/equal to cached truth=80
            assert 302 not in result, (
                f"C302 should be excluded (prob=70 < cached=80): {result.keys()}"
            )
            assert 303 not in result, (
                f"C303 should be excluded (prob=70 < cached=80): {result.keys()}"
            )

            # Should include C301 (prob=80, meets cached=80)
            assert 301 in result, (
                f"C301 should be included (prob=80 >= cached=80): {result.keys()}"
            )

    def test_get_clusters_for_source_returns_empty(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should return empty dict for source resolution (no parents)."""
        with MBDB.get_session() as session:
            source_res = session.get(Resolutions, 1)  # source_a

            result = get_clusters_with_leaves(source_res)

            # Sources have no parents, so should return empty
            assert result == {}, f"Source should have no parent clusters: {result}"

    def test_get_clusters_includes_source_assignments(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should include source cluster assignments from indirect parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            # Should include source clusters from both source_a and source_b
            # (indirect parents through dedupe_a and dedupe_b)

            result_clusters = set(result.keys())

            # Some source clusters should be present (those not captured by dedupers)
            # At minimum, uncaptured clusters should appear
            uncaptured_a = {103, 104, 105}  # Not in C301 (which captures 101, 102)
            uncaptured_b = {203, 204, 205}  # Not in C401 (which captures 201, 202)

            for cluster_id in uncaptured_a.union(uncaptured_b):
                if cluster_id in result_clusters:
                    # If present, should have itself as only leaf
                    leaves = result[cluster_id]["leaves"]
                    assert len(leaves) == 1
                    assert leaves[0]["leaf_id"] == cluster_id

    def test_get_clusters_leaf_structure(self, populated_postgres_db: MatchboxPostgres):
        """Should return leaves with correct structure and unique entries."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_clusters_with_leaves(linker_res)

            for cluster_id, cluster_info in result.items():
                leaves = cluster_info["leaves"]

                # Each leaf should have correct structure
                for leaf in leaves:
                    assert "leaf_id" in leaf
                    assert "leaf_hash" in leaf
                    assert isinstance(leaf["leaf_id"], int)
                    assert isinstance(leaf["leaf_hash"], bytes)

                # Should not have duplicate leaves
                leaf_ids = [leaf["leaf_id"] for leaf in leaves]
                assert len(leaf_ids) == len(set(leaf_ids)), (
                    f"Duplicate leaves in cluster {cluster_id}: {leaf_ids}"
                )

                # Leaves should be valid cluster IDs
                for leaf in leaves:
                    leaf_id = leaf["leaf_id"]
                    assert leaf_id in range(101, 206), (
                        f"Invalid leaf cluster ID: {leaf_id}"
                    )


@pytest.mark.docker
class TestQueryFunction:
    """Test main query function with various scenarios."""

    def test_query_source_only(self, populated_postgres_db: MatchboxPostgres):
        """Should query source data without resolution."""
        result = query("source_a", resolution=None, threshold=None, limit=None)

        # Should return all keys from source_a with their cluster assignments
        assert result.shape[0] == 6
        assert "id" in result.column_names
        assert "key" in result.column_names

        # For source-only queries, id should be the original cluster ID
        assert set(result["id"].to_pylist()) == {
            101,
            102,
            103,
            104,
            105,
        }  # Two keys in 101
        assert set(result["key"].to_pylist()) == {
            "src_a_key1",
            "src_a_key2",
            "src_a_key3",
            "src_a_key4",
            "src_a_key5",
            "src_a_key6",
        }

    def test_query_source_b_only(self, populated_postgres_db: MatchboxPostgres):
        """Should query source_b which has one key per cluster."""
        result = query("source_b", resolution=None, threshold=None, limit=None)

        # Should return all keys from source_b
        assert result.shape[0] == 5
        assert "id" in result.column_names
        assert "key" in result.column_names

        # Source B has one key per cluster
        assert set(result["id"].to_pylist()) == {201, 202, 203, 204, 205}
        assert set(result["key"].to_pylist()) == {
            "src_b_key1",
            "src_b_key2",
            "src_b_key3",
            "src_b_key4",
            "src_b_key5",
        }

    def test_query_through_deduper(self, populated_postgres_db: MatchboxPostgres):
        """Should query source through its deduper resolution."""
        result = query("source_a", resolution="dedupe_a", threshold=None, limit=None)

        # Should return all 6 keys, but some mapped to dedupe clusters
        assert result.shape[0] == 6
        assert "id" in result.column_names
        assert "key" in result.column_names

        # At dedupe_a's default threshold (80), should see C301 for some keys
        cluster_ids = set(result["id"].to_pylist())
        assert 301 in cluster_ids  # C301 should appear

        # Keys 1,2,3 should map to C301 (dedupe of clusters 101+102)
        keys_in_301 = [row["key"] for row in result.to_pylist() if row["id"] == 301]
        expected_301_keys = {"src_a_key1", "src_a_key2", "src_a_key3"}
        assert set(keys_in_301) == expected_301_keys

    def test_query_through_deduper_with_threshold(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should query source through deduper with threshold override."""
        # Test with threshold=90 (higher than dedupe_a's clusters)
        result = query("source_a", resolution="dedupe_a", threshold=90, limit=None)

        # Should return all 6 keys, but no dedupe clusters qualify
        assert result.shape[0] == 6

        # No dedupe clusters should qualify at threshold=90
        cluster_ids = set(result["id"].to_pylist())
        assert 301 not in cluster_ids  # C301 excluded (prob=80 < 90)

        # Should fall back to original source clusters
        expected_source_clusters = {101, 102, 103, 104, 105}
        assert cluster_ids.issubset(expected_source_clusters)

    def test_query_through_linker(self, populated_postgres_db: MatchboxPostgres):
        """Should query source through complex linker resolution."""
        result = query("source_a", resolution="linker_ab", threshold=None, limit=None)

        # Should return all 6 keys with linker cluster assignments
        assert result.shape[0] == 6

        cluster_ids = set(result["id"].to_pylist())

        # At linker's default threshold (90), only C504 qualifies (prob=90, role_flag=1)
        # C503 is excluded because prob=80 < threshold=90
        assert 504 in cluster_ids  # C504 should appear (90% >= 90%)
        assert 503 not in cluster_ids  # C503 should be excluded (80% < 90%)

        # Verify specific key mappings
        key_cluster_map = {row["key"]: row["id"] for row in result.to_pylist()}

        # src_a_key4 (cluster 103) should map to C504 (contains 103+203)
        assert key_cluster_map["src_a_key4"] == 504

        # src_a_key6 (cluster 105) is not contained in any linker cluster
        assert key_cluster_map["src_a_key6"] == 105

    def test_query_both_sources_through_linker(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should query both sources through linker with consistent results."""
        result_a = query("source_a", resolution="linker_ab", threshold=80, limit=None)
        result_b = query("source_b", resolution="linker_ab", threshold=80, limit=None)

        # Both should return their respective key counts
        assert result_a.shape[0] == 6  # source_a keys
        assert result_b.shape[0] == 5  # source_b keys

        # Should see overlapping cluster assignments
        clusters_a = set(result_a["id"].to_pylist())
        clusters_b = set(result_b["id"].to_pylist())

        # At threshold=80, both C503 (prob=80, role_flag=2) and
        # C504 (prob=90, role_flag=1) qualify
        linker_clusters = {503, 504}
        assert linker_clusters.intersection(clusters_a)  # Some linker clusters in A
        assert linker_clusters.intersection(clusters_b)  # Some linker clusters in B

        # Get key cluster mappings
        key_cluster_map_a = {row["key"]: row["id"] for row in result_a.to_pylist()}
        key_cluster_map_b = {row["key"]: row["id"] for row in result_b.to_pylist()}

        # Cross-source linking via C503:
        # C503 contains: 101, 102, 201, 202, 205

        # Keys that should map to C503
        assert key_cluster_map_a["src_a_key1"] == 503  # cluster 101 → C503
        assert key_cluster_map_a["src_a_key2"] == 503  # cluster 101 → C503
        assert key_cluster_map_a["src_a_key3"] == 503  # cluster 102 → C503

        assert key_cluster_map_b["src_b_key1"] == 503  # cluster 201 → C503
        assert key_cluster_map_b["src_b_key2"] == 503  # cluster 202 → C503
        assert key_cluster_map_b["src_b_key5"] == 503  # cluster 205 → C503

        # Cross-source linking via C504:
        # C504 contains: 103, 203
        assert key_cluster_map_a["src_a_key4"] == 504  # cluster 103 → C504
        assert key_cluster_map_b["src_b_key3"] == 504  # cluster 203 → C504

    def test_query_with_limit(self, populated_postgres_db: MatchboxPostgres):
        """Should respect limit parameter."""
        result = query("source_a", resolution=None, threshold=None, limit=3)

        # Should return only 3 rows
        assert result.shape[0] == 3
        assert "id" in result.column_names
        assert "key" in result.column_names

    def test_query_multiple_keys_per_cluster_scenario(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should handle case where multiple keys belong to same cluster."""
        # This tests the scenario causing your test failure
        result = query("source_a", resolution="dedupe_a", threshold=80, limit=None)

        # source_a has 6 keys but some share clusters:
        # - keys 1,2 both in cluster 101 → both map to C301
        # - key 3 in cluster 102 → also maps to C301
        # - keys 4,5,6 in separate clusters → keep original assignments

        assert result.shape[0] == 6  # Still 6 keys total

        # Count unique clusters vs total keys
        unique_clusters = len(set(result["id"].to_pylist()))
        total_keys = result.shape[0]

        # Should have fewer unique clusters than total keys (due to C301 grouping)
        assert unique_clusters < total_keys

        # Verify the specific multiple-keys-to-one-cluster mapping
        key_cluster_map = {row["key"]: row["id"] for row in result.to_pylist()}

        # Keys 1,2,3 should all map to the same dedupe cluster C301
        dedupe_cluster = key_cluster_map["src_a_key1"]
        assert key_cluster_map["src_a_key2"] == dedupe_cluster
        assert key_cluster_map["src_a_key3"] == dedupe_cluster
        assert dedupe_cluster == 301

    def test_query_nonexistent_source(self, populated_postgres_db: MatchboxPostgres):
        """Should raise exception for nonexistent source."""
        with pytest.raises(MatchboxSourceNotFoundError):
            query("nonexistent", resolution=None)

    def test_query_nonexistent_resolution(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should raise exception for nonexistent resolution."""
        with pytest.raises(MatchboxResolutionNotFoundError):
            query("source_a", resolution="nonexistent")


@pytest.mark.docker
class TestMatchFunction:
    """Test matching function."""

    def test_match_within_same_cluster(self, populated_postgres_db: MatchboxPostgres):
        """Should find matches within the same cluster."""
        matches = match(
            key="src_a_key1",
            source="source_a",
            targets=["source_a", "source_b"],
            resolution="dedupe_a",
            threshold=80,
        )

        assert len(matches) == 2  # One for each target

        # Should find the key in source_a target
        source_a_match = next(m for m in matches if m.target == "source_a")
        assert "src_a_key1" in source_a_match.source_id
        assert "src_a_key1" in source_a_match.target_id  # Should match itself

    def test_match_no_cross_source_matches(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should return empty matches when no cross-source links exist."""
        # src_a_key6 is in C105, not linked to source_b at dedupe level
        matches = match(
            key="src_a_key6",
            source="source_a",
            targets=["source_b"],
            resolution="dedupe_a",
            threshold=80,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert len(source_b_match.target_id) == 0  # No matches in source_b

    def test_match_nonexistent_key(self, populated_postgres_db: MatchboxPostgres):
        """Should handle nonexistent key gracefully."""
        matches = match(
            key="nonexistent_key",
            source="source_a",
            targets=["source_b"],
            resolution="dedupe_a",
            threshold=80,
        )

        assert len(matches) == 1
        assert len(matches[0].target_id) == 0  # No matches for nonexistent key
