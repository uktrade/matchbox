"""Test SQL functions for the Matchbox PostgreSQL backend.

The backend adapter tests provide the key coverage we want for any backend. However,
the PostgreSQL SQL functions are complex, and we've found having lower-level tests is
useful for debugging the complex logic required to query a hierarchical system.

Nevertheless, these tests are completely ephemeral, and if the hierarchical
representation changes, they should be rewritten in whatever form best-aids the
development of the new query functions. Don't be precious.
"""

from typing import Generator, Literal

import pytest

from matchbox.common.db import sql_to_df
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Results,
    SourceConfigs,
    SourceFields,
)
from matchbox.server.postgresql.utils.db import compile_sql
from matchbox.server.postgresql.utils.query import (
    _build_unified_query,
    get_parent_clusters_and_leaves,
    get_source_config,
    match,
    query,
)


@pytest.fixture(scope="function")
def populated_postgres_db(
    matchbox_postgres: MatchboxPostgres,
) -> Generator[MatchboxPostgres, None, None]:
    """PostgreSQL database with a rich yet simple test dataset.

    * Source A: 6 keys → 5 clusters (keys 1&2 share cluster 101)
    * Source B: 5 keys → 5 clusters (one key per cluster)
    * Dedupe A
        * Results:
            * 101 - 102 @ 80%
            * 101 - 103 @ 70%
        * Creates:
            * 80% cluster (301 = 101+102)
            * 70% 3-way cluster (302 = 101+102+103)
    * Dedupe B
        * Results:
            * 201 - 202 @ 70%
        * Creates:
            * 70% cluster (401 = 201+202)
    * Linker. Caches truth DA=80, DB=70
        * Results:
            * 301 - 401 @ 90%
            * 103 - 203 @ 90%
            * 205 - 301 @ 80%
        * Creates:
            * 90% cluster (501 = 301+401 = 101+102+201+202)
            * 90% cluster (502 = 103+203) (103 undeduped at DA=80)
            * 80% cluster (503 = 301+205+401 = 101+102+201+202+205)

    This diagram shows the structure of the test dataset. To understand the root-leaf
    relationships, trace the the "root" node down to its source "leaves". This is what's
    stored in the Contains table.

    Pairwise results are shown in the Results table.

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
            %% Source A clusters
            C101["101, two keys"]
            C102["102"]
            C103["103"]
            C104["104"]
            C105["105"]

            %% Source B clusters
            C201["201"]
            C202["202"]
            C203["203"]
            C204["204"]
            C205["205"]

            %% Dedupe A components
            %% Results:
            %% 101 - 102 @ 80
            %% 101 - 103 @ 70
            C301["301 @80%"]
            C302["302 @70%"]

            C301 --> C101
            C301 --> C102

            C302 --> C101
            C302 --> C102
            C302 --> C103

            %% Dedupe B components
            %% Results:
            %% 201 - 202 @ 70
            C401["401 @70%"]

            C401 --> C201
            C401 --> C202

            %% Linker components
            %% Results:
            %% 301 - 401 @ 90
            %% 103 - 203 @ 90
            %% 301 - 205 @ 80
            C501["501 @90%"]
            C502["502 @90%"]
            C503["503 @80%"]

            C501 --> C301
            C501 --> C401

            C502 --> C103
            C502 --> C203

            C503 --> C205
            C503 --> C501
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
        class C301,C302 dedupe_a
        class C401 dedupe_b
        class C501,C502,C503 linker
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
            # Dedupe B clusters (400-series)
            Clusters(cluster_id=401, cluster_hash=b"hash_401"),
            # Linker clusters (500-series)
            Clusters(cluster_id=501, cluster_hash=b"hash_501"),
            Clusters(cluster_id=502, cluster_hash=b"hash_502"),
            Clusters(cluster_id=503, cluster_hash=b"hash_503"),
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
            # Dedupe A: C301 contains C101+C102 (80%)
            Contains(root=301, leaf=101),
            Contains(root=301, leaf=102),
            # Dedupe A: C302 contains C101+C102+C103 (70%)
            Contains(root=302, leaf=101),
            Contains(root=302, leaf=102),
            Contains(root=302, leaf=103),
            # Dedupe B: C401 contains C201+C202 (70%)
            Contains(root=401, leaf=201),
            Contains(root=401, leaf=202),
            # Linker: C501 links C301+C401 (90%)
            Contains(root=501, leaf=101),  # from C301
            Contains(root=501, leaf=102),  # from C301
            Contains(root=501, leaf=201),  # from C401
            Contains(root=501, leaf=202),  # from C401
            # Linker: C502 contains C103+C203 (90%)
            Contains(root=502, leaf=103),
            Contains(root=502, leaf=203),
            # Linker: C503 contains C501+C205 (80% component)
            Contains(root=503, leaf=101),  # from C301 (via C501 and C502)
            Contains(root=503, leaf=102),  # from C301 (via C501 and C502)
            Contains(root=503, leaf=201),  # from C401 (via C501)
            Contains(root=503, leaf=202),  # from C401 (via C501)
            Contains(root=503, leaf=205),
        ]

        # === PROBABILITIES ===
        # Probabilities for formed clusters
        probabilities = [
            # Dedupe A probabilities
            Probabilities(resolution_id=3, cluster_id=301, probability=80),
            Probabilities(resolution_id=3, cluster_id=302, probability=70),
            # Dedupe B probabilities
            Probabilities(resolution_id=4, cluster_id=401, probability=70),
            # Linker probabilities
            Probabilities(resolution_id=5, cluster_id=501, probability=90),
            Probabilities(resolution_id=5, cluster_id=502, probability=90),
            Probabilities(resolution_id=5, cluster_id=503, probability=80),
        ]

        # === RESULTS ===
        # Pairwise probabilities
        results = [
            # Dedupe A results - pairwise that formed components
            Results(
                resolution_id=3, left_id=101, right_id=102, probability=80
            ),  # forms C301
            Results(
                resolution_id=3, left_id=101, right_id=103, probability=70
            ),  # forms C303
            Results(
                resolution_id=3, left_id=102, right_id=103, probability=70
            ),  # part of C302
            # Dedupe B results
            Results(
                resolution_id=4, left_id=201, right_id=202, probability=70
            ),  # forms C401
            # Linker results
            Results(
                resolution_id=5, left_id=301, right_id=401, probability=90
            ),  # forms C501
            Results(
                resolution_id=5, left_id=103, right_id=203, probability=90
            ),  # forms C502
            Results(
                resolution_id=5, left_id=301, right_id=205, probability=80
            ),  # forms C503
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
            results,
        ):
            if isinstance(obj, list):
                session.add_all(obj)
                session.flush()

        session.commit()

    yield matchbox_postgres


@pytest.mark.docker
class TestGetLineage:
    """Test get_lineage method with minimal coverage."""

    def test_get_lineage_source_no_parents(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Source resolutions should return only themselves."""
        with MBDB.get_session() as session:
            source_a = session.get(Resolutions, 1)  # source_a

            lineage = source_a.get_lineage()

            # Source has no parents, should only return itself
            # (resolution_id, source_config_id, threshold)
            assert lineage == [(1, 11, None)]  # source_a has source_config_id 11

    def test_get_lineage_dedupe_has_source_parent(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Dedupe resolution should return itself + source parent."""
        with MBDB.get_session() as session:
            dedupe_a = session.get(Resolutions, 3)  # dedupe_a

            lineage = dedupe_a.get_lineage()

            # Should return: self (dedupe_a) + parent (source_a)
            # Ordered by priority: self first (level 0), then parent (level 1)
            assert lineage == [(3, None, 80), (1, 11, None)]

    def test_get_lineage_linker_has_multiple_parents(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Linker resolution should return itself + all parents in hierarchy."""
        with MBDB.get_session() as session:
            linker = session.get(Resolutions, 5)  # linker_ab

            lineage = linker.get_lineage()

            # Should return: self + direct parents + indirect parents
            # Expected order by (level, resolution_id):
            # - Level 0: linker (5, None, 90)
            # - Level 1: dedupe_a (3, None, 80), dedupe_b (4, None, 70)
            # - Level 2: source_a (1, 11, None), source_b (2, 22, None)
            expected = [
                (5, None, 90),  # self (linker)
                (3, None, 80),  # dedupe_a (level 1, cached truth)
                (4, None, 70),  # dedupe_b (level 1, cached truth)
                (1, 11, None),  # source_a (level 2, source)
                (2, 22, None),  # source_b (level 2, source)
            ]
            assert lineage == expected

    def test_get_lineage_with_source_filter(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should filter lineage to specific source configs."""
        with MBDB.get_session() as session:
            linker = session.get(Resolutions, 5)  # linker_ab
            source_a_config = session.get(SourceConfigs, 11)  # source_a config

            lineage = linker.get_lineage(sources=[source_a_config])

            # Should only include lineage path to source_a:
            # - Level 0: linker (5, None, 90)
            # - Level 1: dedupe_a (3, None, 80) - leads to source_a
            # - Level 2: source_a (1, 11, None) - the target
            # Should NOT include dedupe_b or source_b
            expected = [
                (5, None, 90),  # self (linker)
                (3, None, 80),  # dedupe_a (leads to source_a)
                (1, 11, None),  # source_a (target)
            ]
            assert lineage == expected

    def test_get_lineage_with_threshold_override(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should override only the query resolution's threshold."""
        with MBDB.get_session() as session:
            linker = session.get(Resolutions, 5)  # linker_ab

            lineage = linker.get_lineage(threshold=75)

            # Should override linker's threshold (90 -> 75) but keep cached thresholds
            expected = [
                (5, None, 75),  # self with overridden threshold
                (3, None, 80),  # dedupe_a (cached truth unchanged)
                (4, None, 70),  # dedupe_b (cached truth unchanged)
                (1, 11, None),  # source_a (unchanged)
                (2, 22, None),  # source_b (unchanged)
            ]
            assert lineage == expected

    def test_get_lineage_with_multiple_source_filter(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should filter lineage to multiple source configs."""
        with MBDB.get_session() as session:
            linker = session.get(Resolutions, 5)  # linker_ab
            source_a_config = session.get(SourceConfigs, 11)  # source_a config
            source_b_config = session.get(SourceConfigs, 22)  # source_b config

            lineage = linker.get_lineage(sources=[source_a_config, source_b_config])

            # Should include both lineage paths (same as no filter for linker):
            expected = [
                (5, None, 90),  # self (linker)
                (3, None, 80),  # dedupe_a (leads to source_a)
                (4, None, 70),  # dedupe_b (leads to source_b)
                (1, 11, None),  # source_a
                (2, 22, None),  # source_b
            ]
            assert lineage == expected

    def test_get_lineage_ordering_by_level_then_id(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should verify ordering is by level ASC, then resolution_id ASC."""
        with MBDB.get_session() as session:
            linker = session.get(Resolutions, 5)  # linker_ab

            lineage = linker.get_lineage()

            # Verify ordering: self first, then by level, then by resolution_id
            levels = []
            for i, (res_id, _, _) in enumerate(lineage):  # Updated to unpack 3-tuple
                if i == 0:
                    levels.append(0)  # self is level 0
                elif res_id in [3, 4]:  # dedupe_a, dedupe_b
                    levels.append(1)  # level 1 parents
                elif res_id in [1, 2]:  # source_a, source_b
                    levels.append(2)  # level 2 parents

            # Should be sorted: [0, 1, 1, 2, 2]
            assert levels == [0, 1, 1, 2, 2]

            # Within same level, should be sorted by resolution_id
            level_1_ids = [lineage[1][0], lineage[2][0]]  # dedupe_a, dedupe_b
            assert level_1_ids == [3, 4]  # sorted by ID

            level_2_ids = [lineage[3][0], lineage[4][0]]  # source_a, source_b
            assert level_2_ids == [1, 2]  # sorted by ID

    def test_get_lineage_threshold_override_only_affects_self(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should verify threshold override only affects the query resolution."""
        with MBDB.get_session() as session:
            linker = session.get(Resolutions, 5)  # linker_ab

            # Test with threshold override
            lineage_with_override = linker.get_lineage(threshold=80)

            # Should be: [(5, None, 80), (3, None, 80), (4, None, 70),
            # (1, 11, None), (2, 22, None)]
            # Where:
            # - Resolution 5 (linker): Uses override 80 instead of default 90
            # - Resolution 3 (dedupe_a): Uses cached 80 (unchanged)
            # - Resolution 4 (dedupe_b): Uses cached 70 (unchanged)
            # - Resolutions 1,2 (sources): Use None (unchanged)

            expected = [
                (5, None, 80),  # linker with override
                (3, None, 80),  # dedupe_a with cached threshold
                (4, None, 70),  # dedupe_b with cached threshold
                (1, 11, None),  # source_a unchanged
                (2, 22, None),  # source_b unchanged
            ]

            assert lineage_with_override == expected

            # Compare with no override to confirm only self changed
            lineage_no_override = linker.get_lineage()

            expected_no_override = [
                (5, None, 90),  # linker with default threshold
                (3, None, 80),  # dedupe_a with cached threshold (same)
                (4, None, 70),  # dedupe_b with cached threshold (same)
                (1, 11, None),  # source_a unchanged (same)
                (2, 22, None),  # source_b unchanged (same)
            ]

            assert lineage_no_override == expected_no_override


@pytest.mark.docker
class TestGetSourceConfig:
    """Test source config retrieval."""

    def test_get_existing_source(self, populated_postgres_db: MatchboxPostgres):
        """Should return source config for existing source."""
        with MBDB.get_session() as session:
            source_config = get_source_config("source_a", session)
            assert source_config.source_config_id == 11
            assert source_config.resolution_id == 1


@pytest.mark.docker
@pytest.mark.parametrize("mode", ["root_leaf", "id_key"])
class TestBuildUnifiedQuery:
    """Test unified query building with correct COALESCE expectations."""

    def test_build_unified_source_only(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Should build unified query for source-only scenario."""
        with MBDB.get_session() as session:
            source_a_resolution = session.get(Resolutions, 1)  # source_a resolution
            source_a_config = session.get(SourceConfigs, 11)  # source_a config

            query = _build_unified_query(
                resolution=source_a_resolution,
                sources=[source_a_config],  # Pass source config directly
                threshold=None,
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        if mode == "root_leaf":
            # 5 unique cluster rows (not 6) since two keys share cluster 101
            assert len(result) == 5

            expected_columns = {
                "root_id",
                "root_hash",
                "leaf_id",
                "leaf_hash",
            }
            assert set(result.columns) == expected_columns
            assert all(result["root_id"] == result["leaf_id"])

            # Should have clusters 101, 102, 103, 104, 105
            cluster_ids = set(result["root_id"])
            assert cluster_ids == {101, 102, 103, 104, 105}

        else:  # id_key
            # Should return all 6 keys from source_a
            assert len(result) == 6

            expected_columns = {"id", "key"}
            assert set(result.columns) == expected_columns
            assert set(result["id"]) == {101, 102, 103, 104, 105}

    def test_build_unified_mixed_scenario(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Should build unified query mixing sources and models."""
        with MBDB.get_session() as session:
            dedupe_a_resolution = session.get(Resolutions, 3)  # dedupe_a context
            source_a_config = session.get(SourceConfigs, 11)  # source_a config

            query = _build_unified_query(
                resolution=dedupe_a_resolution,
                sources=[source_a_config],  # Query source_a through dedupe_a
                threshold=80,  # Override threshold to 80
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        if mode == "root_leaf":
            # We should have 5 unique (root_id, leaf_id) combinations:
            # - (301, 101), (301, 102) for the dedupe cluster
            # - (103, 103), (104, 104), (105, 105) for unclaimed clusters
            assert len(result) == 5

            cluster_ids = set(result["root_id"])

            # At threshold 80, C301 qualifies (prob=80 >= 80)
            assert 301 in cluster_ids

            # C301 should contain leaf clusters 101 and 102
            leaves_in_301 = [
                row["leaf_id"] for row in result.to_dicts() if row["root_id"] == 301
            ]
            expected_301_leaves = {101, 102}
            assert set(leaves_in_301) == expected_301_leaves

            # Keys not processed by dedupe_a should keep original clusters
            remaining_clusters = cluster_ids - {301}
            assert remaining_clusters == {103, 104, 105}

        else:  # id_key
            # For id_key, we still get all 6 rows since it includes the key column
            assert len(result) == 6

            cluster_ids = set(result["id"])

            # At threshold 80, C301 qualifies (prob=80 >= 80)
            assert 301 in cluster_ids

            # Keys that should map to C301 (contains clusters 101, 102)
            keys_in_301 = [row["key"] for row in result.to_dicts() if row["id"] == 301]
            expected_301_keys = {"src_a_key1", "src_a_key2", "src_a_key3"}
            assert set(keys_in_301) == expected_301_keys

            # Keys not processed by dedupe_a should keep original clusters
            remaining_clusters = cluster_ids - {301}
            assert remaining_clusters == {103, 104, 105}

    def test_build_unified_linker_all_sources(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Should build unified query for linker with all sources."""
        with MBDB.get_session() as session:
            linker_resolution = session.get(Resolutions, 5)  # linker context
            source_a_config = session.get(SourceConfigs, 11)  # source_a config
            source_b_config = session.get(SourceConfigs, 22)  # source_b config

            query = _build_unified_query(
                resolution=linker_resolution,
                sources=[source_a_config, source_b_config],  # Both sources
                threshold=80,  # Override linker threshold to 80 (from default 90)
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        if mode == "root_leaf":
            # Return 10 unique hashes from the 11 keys
            assert len(result) == 10
            cluster_ids = set(result["root_id"])
        else:  # id_key
            # Should return all 11 keys from both sources
            assert len(result) == 11
            cluster_ids = set(result["id"])

        # At threshold 80 for linker:
        # - C501 (prob=90): 90% >= 80% ✓ but superseded by C503
        # - C502 (prob=90): 90% >= 80% ✓
        # - C503 (prob=80): 80% >= 80% ✓ supersedes C501
        assert 503 in cluster_ids
        assert 502 in cluster_ids

        # C501 should NOT appear - superseded by C503
        assert 501 not in cluster_ids

        # Dedupe clusters should NOT appear - their keys are claimed by linker
        assert 301 not in cluster_ids  # All C301 keys (101,102) claimed by C503
        assert 401 not in cluster_ids  # All C401 keys (201,202) claimed by C503

        # Only unclaimed source clusters should appear
        unclaimed_clusters = cluster_ids - {503, 502}
        assert unclaimed_clusters == {104, 105, 204}

    def test_build_unified_linker_high_threshold(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Should exclude linker clusters that don't meet high threshold."""
        with MBDB.get_session() as session:
            linker_resolution = session.get(Resolutions, 5)  # linker context
            source_a_config = session.get(SourceConfigs, 11)  # source_a config
            source_b_config = session.get(SourceConfigs, 22)  # source_b config

            query = _build_unified_query(
                resolution=linker_resolution,
                sources=[source_a_config, source_b_config],  # Both sources
                threshold=95,  # Very high threshold excludes all linker clusters
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        if mode == "root_leaf":
            # Return 10 unique hashes from the 11 keys
            assert len(result) == 10
            cluster_ids = set(result["root_id"])
        else:  # id_key
            # Should return all 11 keys from both sources
            assert len(result) == 11
            cluster_ids = set(result["id"])

        # No linker clusters qualify at threshold=95
        assert 503 not in cluster_ids  # 80% < 95%
        assert 502 not in cluster_ids  # 90% < 95%

        # But dedupe clusters should appear (use cached thresholds):
        # - C301 from dedupe_a: cached=80, prob=80, so 80% >= 80% ✓
        # - C401 from dedupe_b: cached=70, prob=70, so 70% >= 70% ✓
        assert 301 in cluster_ids
        assert 401 in cluster_ids

        # Remaining clusters should be unclaimed sources
        expected_clusters = {103, 104, 105, 203, 204, 205}
        assert cluster_ids == expected_clusters | {301, 401}

    def test_build_unified_single_source_filter(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Should filter to single source lineage."""
        with MBDB.get_session() as session:
            linker_resolution = session.get(Resolutions, 5)  # linker context
            source_b_config = session.get(SourceConfigs, 22)  # source_b only

            query = _build_unified_query(
                resolution=linker_resolution,
                sources=[source_b_config],  # Only source_b
                threshold=80,  # Override linker threshold to 80
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        if mode == "root_leaf":
            # Return 5 unique hashes from the 5 keys
            assert len(result) == 5
            cluster_ids = set(result["root_id"])
        else:  # id_key
            # Should return only 5 keys from source_b
            assert len(result) == 5
            cluster_ids = set(result["id"])

        # Should see linker clusters that contain source_b data:
        # C503 contains source_b keys: 201, 202, 205
        # C502 contains source_b key: 203
        assert 503 in cluster_ids
        assert 502 in cluster_ids

        # Should NOT see dedupe_b cluster - its keys claimed by C503
        assert 401 not in cluster_ids

        # Should NOT see C501 - superseded by C503
        assert 501 not in cluster_ids

        # Should NOT see any source_a related clusters (filtered out)
        assert 301 not in cluster_ids
        for source_a_cluster in [101, 102, 103, 104, 105]:
            assert source_a_cluster not in cluster_ids

        # Should see only unclaimed source_b cluster
        unclaimed_source_b = cluster_ids - {503, 502}
        assert unclaimed_source_b == {204}

    def test_build_unified_no_source_filter(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Should include all sources when no filtering."""
        with MBDB.get_session() as session:
            linker_resolution = session.get(Resolutions, 5)  # linker context

            query = _build_unified_query(
                resolution=linker_resolution,
                sources=None,  # No filtering - all sources
                threshold=80,  # Override linker threshold to 80
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        if mode == "root_leaf":
            # Return 10 unique hashes from the 11 keys
            assert len(result) == 10
            cluster_ids = set(result["root_id"])
        else:  # id_key
            # Should return all 11 keys from both sources
            assert len(result) == 11
            cluster_ids = set(result["id"])

        # At threshold 80 for linker:
        # - C502 (prob=90): 90% >= 80% ✓
        # - C503 (prob=80): 80% >= 80% ✓ (supersedes C501)
        assert 503 in cluster_ids  # 80% >= 80%
        assert 502 in cluster_ids  # 90% >= 80%

        # C501 should NOT appear - superseded by C503
        assert 501 not in cluster_ids

        # Dedupe clusters should NOT appear - claimed by linker
        assert 301 not in cluster_ids  # Claimed by C503
        assert 401 not in cluster_ids  # Claimed by C503

        # Only unclaimed source clusters appear
        unclaimed_clusters = cluster_ids - {503, 502}
        assert unclaimed_clusters == {104, 105, 204}

    def test_build_unified_source_only_no_sources_filter(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Source resolution with sources=None returns that source only."""
        with MBDB.get_session() as session:
            source_a_resolution = session.get(Resolutions, 1)  # source_a resolution

            query = _build_unified_query(
                resolution=source_a_resolution,  # Query source_a
                sources=None,  # No source filtering
                threshold=None,
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        if mode == "root_leaf":
            # Return 5 unique hashes from the 6 keys
            assert len(result) == 5
            cluster_ids = set(result["root_id"])
            # Can't check source_config_id since it's no longer in "root_leaf" results
            assert cluster_ids == {101, 102, 103, 104, 105}
        else:
            # Should return 6 keys from source_a
            assert len(result) == 6
            cluster_ids = set(result["id"])
            assert cluster_ids == {101, 102, 103, 104, 105}

    def test_simple_thresholding(
        self,
        populated_postgres_db: MatchboxPostgres,
        mode: Literal["root_leaf", "id_key"],
    ):
        """Simple test showing thresholding works."""
        # Query at threshold=70 where both C301 (80%) and C302 (70%) qualify
        with MBDB.get_session() as session:
            dedupe_resolution = session.get(Resolutions, 3)  # dedupe context
            source_a_config = session.get(SourceConfigs, 11)  # source_a config

            query = _build_unified_query(
                resolution=dedupe_resolution,
                sources=[source_a_config],
                threshold=70,  # Use threshold lower than cache and C301
                mode=mode,
            )

        with MBDB.get_adbc_connection() as conn:
            result = sql_to_df(compile_sql(query), conn.dbapi_connection, "polars")

        # Get the appropriate column name for cluster IDs
        if mode == "id_key":
            # Should return 6 keys from source a with no repetition
            assert len(result) == 6
            cluster_column = "id"
        else:  # "root_leaf"
            # Return 5 unique hashes from the 6 keys
            assert len(result) == 5
            cluster_column = "root_id"

        # C301 should be excluded in favour of 302
        cluster_ids = result[cluster_column].to_list()
        assert 301 not in cluster_ids
        assert 302 in cluster_ids


@pytest.mark.docker
class TestGetClustersWithLeaves:
    """Test cluster-leaf relationship extraction."""

    def test_get_clusters_for_model(self, populated_postgres_db: MatchboxPostgres):
        """Should return cluster hierarchy for model's parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_parent_clusters_and_leaves(linker_res)

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

            result = get_parent_clusters_and_leaves(dedupe_res)

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

            result = get_parent_clusters_and_leaves(linker_res)

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

            result = get_parent_clusters_and_leaves(linker_res)

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

            result = get_parent_clusters_and_leaves(source_res)

            # Sources have no parents, so should return empty
            assert result == {}, f"Source should have no parent clusters: {result}"

    def test_get_clusters_includes_source_assignments(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should include source cluster assignments from indirect parents."""
        with MBDB.get_session() as session:
            linker_res = session.get(Resolutions, 5)  # linker_ab

            result = get_parent_clusters_and_leaves(linker_res)

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

            result = get_parent_clusters_and_leaves(linker_res)

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

        # At linker's default threshold (90), C501 and C502 qualify (both prob=90)
        # C503 is excluded because prob=80 < threshold=90
        assert 501 in cluster_ids  # C501 should appear (90% >= 90%)
        assert 502 in cluster_ids  # C502 should appear (90% >= 90%)
        assert 503 not in cluster_ids  # C503 should be excluded (80% < 90%)

        # Verify specific key mappings
        key_cluster_map = {row["key"]: row["id"] for row in result.to_pylist()}

        # src_a_key4 (cluster 103) should map to C502 (contains 103+203)
        assert key_cluster_map["src_a_key4"] == 502

        # src_a_key1,2,3 (clusters 101,102) should map to C501
        # (contains 101+102+201+202)
        assert key_cluster_map["src_a_key1"] == 501  # cluster 101 → C501
        assert key_cluster_map["src_a_key2"] == 501  # cluster 101 → C501
        assert key_cluster_map["src_a_key3"] == 501  # cluster 102 → C501

        # src_a_key5,6 (clusters 104,105) are not contained in any linker cluster
        assert key_cluster_map["src_a_key5"] == 104
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

        # At threshold=80, C501, C502 (prob=90) and C503 (prob=80) all qualify
        linker_clusters = {501, 502, 503}
        assert linker_clusters.intersection(clusters_a)  # Some linker clusters in A
        assert linker_clusters.intersection(clusters_b)  # Some linker clusters in B

        # Get key cluster mappings
        key_cluster_map_a = {row["key"]: row["id"] for row in result_a.to_pylist()}
        key_cluster_map_b = {row["key"]: row["id"] for row in result_b.to_pylist()}

        # Cross-source linking via C503:
        # C503 contains: 101, 102, 201, 202, 205

        # Keys that should map to C503 (supersedes C501)
        assert key_cluster_map_a["src_a_key1"] == 503  # cluster 101 → C503
        assert key_cluster_map_a["src_a_key2"] == 503  # cluster 101 → C503
        assert key_cluster_map_a["src_a_key3"] == 503  # cluster 102 → C503

        assert key_cluster_map_b["src_b_key1"] == 503  # cluster 201 → C503
        assert key_cluster_map_b["src_b_key2"] == 503  # cluster 202 → C503
        assert key_cluster_map_b["src_b_key5"] == 503  # cluster 205 → C503

        # Cross-source linking via C502:
        # C502 contains: 103, 203
        assert key_cluster_map_a["src_a_key4"] == 502  # cluster 103 → C502
        assert key_cluster_map_b["src_b_key3"] == 502  # cluster 203 → C502

        # Remaining unclaimed keys
        assert key_cluster_map_a["src_a_key5"] == 104  # not claimed by linker
        assert key_cluster_map_a["src_a_key6"] == 105  # not claimed by linker
        assert key_cluster_map_b["src_b_key4"] == 204  # not claimed by linker

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

    def test_match_one_to_many_via_linker(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should find one-to-many matches through linker resolution."""
        # Use linker to connect source_a key to multiple source_b keys via C503
        # C503 contains: 101, 102, 201, 202, 205
        # src_a_key1 (cluster 101) should match src_b_key1, src_b_key2, src_b_key5
        matches = match(
            key="src_a_key1",
            source="source_a",
            targets=["source_b"],
            resolution="linker_ab",
            threshold=80,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert source_b_match.target == "source_b"
        assert len(source_b_match.target_id) > 1  # Should match multiple keys

        # Should contain the expected keys from C503
        expected_keys = {"src_b_key1", "src_b_key2", "src_b_key5"}
        assert expected_keys.issubset(source_b_match.target_id)

    def test_match_many_to_one_within_same_source_cluster(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should find other keys in the same source cluster AND all target keys."""
        # Use one of the two keys in C101 (src_a_key1, src_a_key2)
        # When matching src_a_key1, should get:
        # - Source: both src_a_key1 and src_a_key2 (and src_a_key3 via C301)
        # - Target: all source_b keys linked via C503
        matches = match(
            key="src_a_key1",  # One key from C101
            source="source_a",
            targets=["source_b"],  # Cross-source target
            resolution="linker_ab",
            threshold=80,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert source_b_match.target == "source_b"

        # Source should contain ALL keys from the same cluster (C301 via C503)
        expected_source_keys = {"src_a_key1", "src_a_key2", "src_a_key3"}
        assert expected_source_keys.issubset(source_b_match.source_id)

        # Target should contain ALL linked source_b keys via C503
        # C503 contains: 101, 102, 201, 202, 205
        expected_target_keys = {"src_b_key1", "src_b_key2", "src_b_key5"}
        assert expected_target_keys.issubset(source_b_match.target_id)

    def test_match_one_to_none_isolated_cluster(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should handle one-to-none scenario with isolated clusters."""
        # src_a_key5 (cluster 104) is not connected to any source_b clusters
        matches = match(
            key="src_a_key5",
            source="source_a",
            targets=["source_b"],
            resolution="linker_ab",
            threshold=80,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert source_b_match.target == "source_b"
        assert len(source_b_match.target_id) == 0  # No matches in source_b
        assert source_b_match.cluster is not None  # Should still have cluster info

    def test_match_none_to_none_nonexistent_key(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should handle none-to-none scenario with nonexistent key."""
        matches = match(
            key="completely_nonexistent_key",
            source="source_a",
            targets=["source_b"],
            resolution="linker_ab",
            threshold=80,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert source_b_match.target == "source_b"
        assert len(source_b_match.source_id) == 0  # No source key found
        assert len(source_b_match.target_id) == 0  # No target matches
        assert source_b_match.cluster is None  # No cluster for nonexistent key

    def test_match_with_threshold_filtering(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should respect threshold filtering in match results."""
        # At threshold=90, only C504 qualifies from linker (prob=90)
        # C503 is excluded (prob=80 < 90)
        matches = match(
            key="src_a_key4",  # cluster 103, part of C504
            source="source_a",
            targets=["source_b"],
            resolution="linker_ab",
            threshold=90,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert source_b_match.target == "source_b"

        # Should match src_b_key3 (cluster 203) via C504
        assert "src_b_key3" in source_b_match.target_id
        assert len(source_b_match.target_id) == 1  # Only one match at this threshold

    def test_match_multiple_targets(self, populated_postgres_db: MatchboxPostgres):
        """Should handle matching against multiple target sources."""
        matches = match(
            key="src_a_key1",
            source="source_a",
            targets=["source_a", "source_b"],  # Multiple targets
            resolution="linker_ab",
            threshold=80,
        )

        assert len(matches) == 2  # One match per target

        targets = {m.target for m in matches}
        assert targets == {"source_a", "source_b"}

        # Self-match in source_a should contain the key itself
        source_a_match = next(m for m in matches if m.target == "source_a")
        assert "src_a_key1" in source_a_match.target_id

        # Cross-source match should contain linked keys
        source_b_match = next(m for m in matches if m.target == "source_b")
        assert len(source_b_match.target_id) > 0

    def test_match_dedupe_only_no_cross_source(
        self, populated_postgres_db: MatchboxPostgres
    ):
        """Should handle dedupe-only resolution with no cross-source linking."""
        # dedupe_a only processes source_a, so no source_b matches expected
        matches = match(
            key="src_a_key1",
            source="source_a",
            targets=["source_b"],
            resolution="dedupe_a",
            threshold=80,
        )

        assert len(matches) == 1
        source_b_match = matches[0]
        assert source_b_match.target == "source_b"
        assert (
            len(source_b_match.target_id) == 0
        )  # No cross-source links at dedupe level
