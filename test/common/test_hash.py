import time
import uuid
from enum import Flag

import polars as pl
import pyarrow as pa
import pytest

from matchbox.common.hash import (
    Cluster,
    HashMethod,
    IntMap,
    hash_arrow_table,
    hash_rows,
)
from matchbox.common.logging import logger
from matchbox.common.transform import DisjointSet


def test_intmap_basic():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    assert len({a, b, c}) == 3
    assert max(a, b, c) < 0


def test_intmap_same():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    im2 = IntMap(salt=10)
    x = im2.index(1, 2)
    y = im2.index(3, 4)
    z = im2.index(a, b)

    assert (a, b, c) == (x, y, z)


def test_intmap_different():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2)
    b = im1.index(3, 4)
    c = im1.index(a, b)

    im2 = IntMap(salt=11)
    x = im2.index(1, 2)
    y = im2.index(3, 4)
    z = im2.index(a, b)

    for v1, v2 in zip([a, b, c], [x, y, z], strict=True):
        assert v1 != v2


def test_intmap_unordered():
    im1 = IntMap(salt=10)
    a = im1.index(1, 2, 3)
    b = im1.index(3, 1, 2)

    assert a == b


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_hash_rows(method: HashMethod):
    data = pl.DataFrame(
        {
            "string_col": ["abc", "def", "ghi"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "struct_col": [{"a": 1, "b": "x"}, {"a": 2, "b": None}, {"a": 3, "b": "z"}],
            "object_col": [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()],
            "binary_col": [b"data1", b"data2", b"data3"],
        }
    )

    assert isinstance(data["string_col"].dtype, pl.String)
    assert isinstance(data["int_col"].dtype, pl.Int64)
    assert isinstance(data["float_col"].dtype, pl.Float64)
    assert isinstance(data["struct_col"].dtype, pl.Struct)
    assert isinstance(data["object_col"].dtype, pl.Object)
    assert isinstance(data["binary_col"].dtype, pl.Binary)

    hash_rows(data, columns=data.columns, method=method)


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_hash_arrow_table(method: HashMethod):
    a = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    # Field order should not matter
    b = pa.Table.from_pydict(
        {
            "b": [4, 5, 6],
            "a": [1, 2, 3],
        }
    )
    # Row order should not matter
    c = pa.Table.from_pydict(
        {
            "a": [3, 2, 1],
            "b": [6, 5, 4],
        }
    )
    # Field and row order should not matter
    d = pa.Table.from_pydict(
        {
            "b": [6, 5, 4],
            "a": [3, 2, 1],
        }
    )
    # Empty table should have a different hash
    e = pa.Table.from_pydict(
        {
            "a": [],
            "b": [],
        }
    )
    # Different row data should have a different hash
    f = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 7],
        }
    )
    # If field name change their order, the hash should change
    g = pa.Table.from_pydict(
        {
            "b": [1, 2, 3],
            "a": [4, 5, 6],
        }
    )
    # List fields are handled
    h = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [[1, 2], [3, 4], [5, 6]],
        }
    )
    # List order doesn't matter
    i = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [[2, 1], [4, 3], [6, 5]],
        }
    )
    # Binary fields are handled, including non-UTF-8 bytes
    j = pa.Table.from_pydict(
        {
            "a": [1, 2, 3],
            "b": [b"abc", None, bytes([255, 254, 253])],
        }
    )

    h_a = hash_arrow_table(a, method=method)
    h_a1 = hash_arrow_table(a, method=method)
    h_b = hash_arrow_table(b, method=method)
    h_c = hash_arrow_table(c, method=method)
    h_d = hash_arrow_table(d, method=method)
    h_e = hash_arrow_table(e, method=method)
    h_f = hash_arrow_table(f, method=method)
    h_g = hash_arrow_table(g, method=method)
    h_h = hash_arrow_table(h, method=method)
    h_i = hash_arrow_table(i, method=method)
    h_j = hash_arrow_table(j, method=method)

    # Basic type check
    assert isinstance(h_a, bytes)
    # Basic invariance checks
    assert h_a == h_a1 == h_b == h_c == h_d
    # Different data = different hashes
    assert h_a != h_e
    assert h_a != h_f
    assert h_a != h_g
    assert h_a != h_j
    # List type table should be consistent regardless of field order
    assert h_h == h_i


@pytest.mark.parametrize(
    ["method"],
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)
def test_struct_json_hashing(method: HashMethod):
    """Test that struct/JSON data can be properly hashed."""

    # Basic struct test
    a = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )

    assert isinstance(pl.from_arrow(a)["metadata"].dtype, pl.Struct)

    # Same data but different struct serialization
    b = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"age": 30, "name": "Alice"},
                {"age": 25, "name": "Bob"},
                {"age": 35, "name": "Charlie"},
            ],
        }
    )

    # Different data in structs
    c = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "age": 31},  # Changed age
                {"name": "Bob", "age": 25},
                {"name": "Charlie", "age": 35},
            ],
        }
    )

    # Nested structs
    d = pa.Table.from_pydict(
        {
            "id": [1, 2, 3],
            "metadata": [
                {"name": "Alice", "details": {"city": "New York", "active": True}},
                {"name": "Bob", "details": {"city": "Boston", "active": False}},
                {"name": "Charlie", "details": {"city": "Chicago", "active": True}},
            ],
        }
    )

    # Test basic struct hashing
    h_a = hash_arrow_table(a, method=method)
    h_a1 = hash_arrow_table(a, method=method)
    h_b = hash_arrow_table(b, method=method)
    h_c = hash_arrow_table(c, method=method)
    h_d = hash_arrow_table(d, method=method)

    # Basic type check
    assert isinstance(h_a, bytes)

    # Basic equality check
    assert h_a == h_a1 == h_b
    # Difference checks
    assert h_a != h_c
    assert h_a != h_d


def collect_all_leaves(cluster: Cluster) -> set[Cluster]:
    """Recursively collect all leaf nodes from a cluster."""
    if cluster.leaves is None:
        return {cluster}

    leaves = set()
    for leaf in cluster.leaves:
        leaves.update(collect_all_leaves(leaf))
    return leaves


class TestClusterHierarchy:
    """Tests for using Cluster objects with DisjointSet to build hierarchies."""

    @pytest.fixture
    def intmap(self) -> IntMap:
        """Create a fresh IntMap for each test."""
        return IntMap(salt=10)

    @pytest.fixture
    def test_flag(self) -> Flag:
        """Create a dummy Flag enum for testing."""

        class TestFlag(Flag):
            PAIR = 1
            COMPONENT = 2

        return TestFlag

    @pytest.fixture
    def leaf_nodes(self, intmap: IntMap) -> list[Cluster]:
        """Create six leaf nodes for testing."""
        return [
            Cluster(
                intmap=intmap,
                probability=None,
                leaves=None,
                id=i,
                hash=f"hash{i}".encode(),
            )
            for i in range(1, 7)
        ]

    def test_create_leaf_nodes(self, intmap: IntMap):
        """Test that leaf nodes can be created properly."""
        # Create a leaf node
        node = Cluster(
            intmap=intmap, probability=None, leaves=None, id=1, hash=b"hash1"
        )

        # Verify its properties
        assert node.id == 1
        assert node.hash == b"hash1"
        assert node.leaves is None
        assert node.flag is None

    def test_create_leaf_node_with_flag(self, intmap: IntMap, test_flag: Flag):
        """Test that leaf nodes can be created with flags."""
        # Create a leaf node with a flag
        node = Cluster(
            intmap=intmap,
            probability=None,
            leaves=None,
            id=1,
            hash=b"hash1",
            flag=test_flag.PAIR,
        )

        # Verify its properties
        assert node.id == 1
        assert node.hash == b"hash1"
        assert node.leaves is None
        assert node.flag == test_flag.PAIR

    def test_add_leaves_to_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test adding leaf nodes to a DisjointSet."""
        # Create a DisjointSet and add leaves
        dsj = DisjointSet[Cluster]()
        for node in leaf_nodes:
            dsj.add(node)

        # Verify all nodes are in separate components
        components = dsj.get_components()
        assert len(components) == 6
        for component in components:
            assert len(component) == 1

    def test_union_leaves_in_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test creating unions of leaf nodes in DisjointSet."""
        dsj = DisjointSet[Cluster]()
        for node in leaf_nodes:
            dsj.add(node)

        # Create three pairs
        dsj.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        dsj.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        dsj.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        # Verify we now have three components
        components = dsj.get_components()
        assert len(components) == 3

        # Verify each component has the expected size
        component_sizes = [len(comp) for comp in components]
        assert sorted(component_sizes) == [2, 2, 2]

    def test_create_clusters_from_components(self, leaf_nodes: list[Cluster]):
        """Test creating new clusters from DisjointSet components."""
        # Add leaves to DisjointSet and create unions
        dsj = DisjointSet[Cluster]()
        for node in leaf_nodes:
            dsj.add(node)

        dsj.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        dsj.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        dsj.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        # Create clusters from components
        components = dsj.get_components()
        clusters: list[Cluster] = []
        for component in components:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(component_list, probability=100)
            clusters.append(cluster)

        # Verify we have three clusters
        assert len(clusters) == 3

        # Verify each non-leaf cluster has exactly two leaves
        for cluster in clusters:
            if cluster.leaves is not None:
                assert len(cluster.leaves) == 2

        # Verify probability is as expected for non-leaf clusters
        for cluster in clusters:
            if cluster.leaves is not None:
                assert cluster.probability == 100

    def test_create_clusters_from_components_with_flag(
        self, leaf_nodes: list[Cluster], test_flag: Flag
    ):
        """Test creating new clusters from DisjointSet components with flags."""
        # Add leaves to DisjointSet and create unions
        dsj = DisjointSet[Cluster]()
        for node in leaf_nodes:
            dsj.add(node)

        dsj.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        dsj.union(leaf_nodes[2], leaf_nodes[3])  # 3,4

        # Create clusters from components with flags
        components = dsj.get_components()
        clusters: list[Cluster] = []

        for component in components:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(
                    component_list, probability=100, flag=test_flag.PAIR
                )
            clusters.append(cluster)

        # Verify clusters with flags
        for cluster in clusters:
            if cluster.leaves is not None:
                assert cluster.flag == test_flag.PAIR
            else:
                assert cluster.flag is None  # Leaf nodes don't have flags in this test

    def test_level1_clusters_in_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test using level-1 clusters in another DisjointSet."""
        # Create level-1 clusters as in previous test
        dsj1 = DisjointSet[Cluster]()
        for node in leaf_nodes:
            dsj1.add(node)

        dsj1.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        dsj1.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        dsj1.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        components = dsj1.get_components()
        level1_clusters = []
        for component in components:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(component_list)
            level1_clusters.append(cluster)

        # Add these clusters to a new DisjointSet
        dsj2 = DisjointSet[Cluster]()
        for cluster in level1_clusters:
            dsj2.add(cluster)

        # Verify initial state
        assert len(dsj2.get_components()) == 3

        # Create a union of first two clusters
        dsj2.union(level1_clusters[0], level1_clusters[1])

        # Verify we now have two components
        components2 = dsj2.get_components()
        assert len(components2) == 2

        # One component should have 2 clusters, the other should have 1
        component_sizes = [len(comp) for comp in components2]
        assert sorted(component_sizes) == [1, 2]

    def test_leaf_preservation_in_hierarchy(self, leaf_nodes: list[Cluster]):
        """Test that leaf nodes are preserved in the cluster hierarchy."""
        # Follow the same steps as in previous tests to build a hierarchy
        # Step 1: Create level-1 clusters
        dsj1 = DisjointSet[Cluster]()
        for node in leaf_nodes:
            dsj1.add(node)

        dsj1.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        dsj1.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        dsj1.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        components1 = dsj1.get_components()
        level1_clusters = []
        for component in components1:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(component_list, probability=90)
            level1_clusters.append(cluster)

        # Verify probability as expected
        for cluster in level1_clusters:
            assert cluster.probability == 90

        # Step 2: Union level-1 clusters
        dsj2 = DisjointSet[Cluster]()
        for cluster in level1_clusters:
            dsj2.add(cluster)

        dsj2.union(level1_clusters[0], level1_clusters[1])

        # Find the component with multiple clusters
        larger_component = None
        for component in dsj2.get_components():
            if len(component) > 1:
                larger_component = component
                break

        assert larger_component is not None

        # Create a level-2 cluster from the larger component
        level2_cluster = Cluster.combine(larger_component, probability=80)

        # Verify the level-2 cluster contains exactly the leaf nodes 1, 2, 3, and 4
        expected_leaves = set(leaf_nodes[:4])  # Nodes 1-4

        # Collect all leaf nodes from the level-2 cluster
        actual_leaves = collect_all_leaves(level2_cluster)

        # Verify leaf nodes match
        assert actual_leaves == expected_leaves
        assert len(actual_leaves) == 4
        # Verify probability as expected
        assert level2_cluster.probability == 80

    def test_combine_with_single_cluster(self, leaf_nodes: list[Cluster]):
        """Test that combine works correctly with a single cluster."""
        # Create a single-element list
        clusters = [leaf_nodes[0]]

        # Combine
        result = Cluster.combine(clusters, probability=None)

        # Should return the original cluster
        assert result is leaf_nodes[0]
        # Verify probability as expected
        assert result.probability is None

    def test_combine_with_leaf_and_non_leaf(
        self, intmap: IntMap, leaf_nodes: list[Cluster]
    ):
        """Test combine with a mix of leaf and non-leaf clusters."""
        # Create a non-leaf cluster
        non_leaf = Cluster(intmap=intmap, leaves=[leaf_nodes[0], leaf_nodes[1]])

        # Combine with a leaf node
        result = Cluster.combine([non_leaf, leaf_nodes[2]], probability=70)

        # Should contain three leaves
        assert len(collect_all_leaves(result)) == 3

        # The specific leaves should be 1, 2, and 3
        expected = {leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]}
        assert collect_all_leaves(result) == expected

        # Verify probability as expected
        assert result.probability == 70

    def test_combine_with_flag(self, leaf_nodes: list[Cluster], test_flag: Flag):
        """Test the combine method with flag parameter."""
        # Combine clusters with a flag
        result = Cluster.combine(
            [leaf_nodes[0], leaf_nodes[1]], probability=80, flag=test_flag.PAIR
        )

        # Verify result has the flag set
        assert result.flag == test_flag.PAIR
        assert result.probability == 80
        assert len(result.leaves) == 2

    def test_combine_with_multiple_flags(
        self, leaf_nodes: list[Cluster], test_flag: Flag
    ):
        """Test the combine method with multiple flags active."""
        # Combine clusters with multiple flags
        combined_flags = test_flag.PAIR | test_flag.COMPONENT
        result = Cluster.combine(
            [leaf_nodes[0], leaf_nodes[1]], probability=80, flag=combined_flags
        )

        # Verify result has both flags set
        assert result.flag == combined_flags
        assert test_flag.PAIR in result.flag
        assert test_flag.COMPONENT in result.flag
        assert result.probability == 80
        assert len(result.leaves) == 2

    def test_combine_preserves_flag_from_input(
        self, intmap: IntMap, leaf_nodes: list[Cluster], test_flag: Flag
    ):
        """Test that combine uses the provided flag parameter over existing flags."""
        # Create a cluster with one flag
        cluster_with_flag = Cluster(
            intmap=intmap, leaves=[leaf_nodes[0], leaf_nodes[1]], flag=test_flag.PAIR
        )

        # Combine with a different flag
        result = Cluster.combine(
            [cluster_with_flag, leaf_nodes[2]], flag=test_flag.COMPONENT
        )

        # The result should have the new flag, not the existing one
        assert result.flag == test_flag.COMPONENT

    def test_flag_inheritance_in_hierarchical_combining(
        self, leaf_nodes: list[Cluster], test_flag: Flag
    ):
        """Test flag behavior when combining clusters that already have flags."""
        # Create clusters with different flags
        pair_cluster = Cluster.combine(
            [leaf_nodes[0], leaf_nodes[1]], flag=test_flag.PAIR
        )
        component_cluster = Cluster.combine(
            [leaf_nodes[2], leaf_nodes[3]], flag=test_flag.COMPONENT
        )

        # Combine clusters with both flags active
        both_flags = test_flag.PAIR | test_flag.COMPONENT
        result = Cluster.combine([pair_cluster, component_cluster], flag=both_flags)

        # Verify the result has both flags
        assert result.flag == both_flags
        assert test_flag.PAIR in result.flag
        assert test_flag.COMPONENT in result.flag

        # Verify all leaf nodes are preserved
        all_leaves = collect_all_leaves(result)
        expected_leaves = {leaf_nodes[0], leaf_nodes[1], leaf_nodes[2], leaf_nodes[3]}
        assert all_leaves == expected_leaves

    def test_flag_operations(self, leaf_nodes: list[Cluster], test_flag: Flag):
        """Test various flag operations and combinations."""
        # Test no flags
        no_flag_cluster = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        assert no_flag_cluster.flag is None

        # Test single flag
        pair_cluster = Cluster.combine(
            [leaf_nodes[0], leaf_nodes[1]], flag=test_flag.PAIR
        )
        assert pair_cluster.flag == test_flag.PAIR

        # Test checking if a flag is present
        combined_flags = test_flag.PAIR | test_flag.COMPONENT
        multi_flag_cluster = Cluster.combine(
            [leaf_nodes[0], leaf_nodes[1]], flag=combined_flags
        )

        # Test flag membership
        assert test_flag.PAIR in multi_flag_cluster.flag
        assert test_flag.COMPONENT in multi_flag_cluster.flag

        # Test that individual flags are different from combined
        assert multi_flag_cluster.flag != test_flag.PAIR
        assert multi_flag_cluster.flag != test_flag.COMPONENT

    def test_hash_consistency_regardless_of_order(self, leaf_nodes: list[Cluster]):
        """Test that the hash is consistent regardless of the order of leaves."""
        # Create two clusters with the same leaves but in different order
        cluster1 = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        cluster2 = Cluster.combine([leaf_nodes[1], leaf_nodes[0]])

        # Verify that both clusters have the same hash
        assert cluster1.hash == cluster2.hash

        # Test with more complex combinations
        cluster3 = Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]])
        cluster4 = Cluster.combine([leaf_nodes[2], leaf_nodes[0], leaf_nodes[1]])
        cluster5 = Cluster.combine([leaf_nodes[1], leaf_nodes[2], leaf_nodes[0]])

        # All permutations should have the same hash
        assert cluster3.hash == cluster4.hash
        assert cluster4.hash == cluster5.hash

    def test_hash_based_on_leaf_hashes_only(self, leaf_nodes: list[Cluster]):
        """Test that cluster hashes are based on leaf node hashes only."""
        # Create two different paths to the same set of leaf nodes

        # Path 1: Direct combination of three leaves
        direct = Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]])

        # Path 2: Hierarchical combination
        intermediate = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        hierarchical = Cluster.combine([intermediate, leaf_nodes[2]])

        # Both should have the same hash since they contain the same leaf nodes
        assert direct.hash == hierarchical.hash

        # Create a completely different structure with the same leaves
        # This time using combine with all leaves
        combined = Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]])

        # Should still have the same hash
        assert direct.hash == combined.hash

    def test_hash_with_provided_hash_value(
        self, intmap: IntMap, leaf_nodes: list[Cluster]
    ):
        """Test that providing a hash value overrides the calculated hash."""
        # Create a cluster with a provided hash
        custom_hash = b"custom_hash_value"
        cluster = Cluster(
            intmap=intmap,
            leaves=[leaf_nodes[0], leaf_nodes[1]],
            hash=custom_hash,
        )

        # Verify the hash is the custom one
        assert cluster.hash == custom_hash

        # Verify it's different from what would have been calculated
        auto_cluster = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        assert cluster.hash != auto_cluster.hash

    def test_hash_generation_for_complex_hierarchy(
        self, intmap: IntMap, leaf_nodes: list[Cluster]
    ):
        """Test that hash generation works correctly for complex hierarchies."""
        # Create level 1 clusters
        cluster_a = Cluster.combine([leaf_nodes[0], leaf_nodes[1]])
        cluster_b = Cluster.combine([leaf_nodes[2], leaf_nodes[3]])
        cluster_c = Cluster.combine([leaf_nodes[4], leaf_nodes[5]])

        # Create level 2 clusters
        level2_a = Cluster.combine([cluster_a, cluster_b])
        level2_b = cluster_c

        # Create top level cluster
        top_level = Cluster.combine([level2_a, level2_b])

        # Verify the top level hash is derived from all leaf nodes
        # by comparing with a direct combination
        direct_combination = Cluster.combine(leaf_nodes)

        # Both should have the same hash since they contain the same leaf nodes
        assert top_level.hash == direct_combination.hash

        # Check that the hash is consistent when using different intermediate groupings
        alternate_grouping = Cluster.combine(
            [
                Cluster.combine([leaf_nodes[0], leaf_nodes[1], leaf_nodes[2]]),
                Cluster.combine([leaf_nodes[3], leaf_nodes[4], leaf_nodes[5]]),
            ]
        )
        assert top_level.hash == alternate_grouping.hash

    def test_hash_independence_from_intermediate_changes(
        self, intmap: IntMap, leaf_nodes: list[Cluster]
    ):
        """Test that hashes depend only on leaf nodes, not intermediate structure.

        Custom intermediate hashes don't affect the final hash, showing that
        hash calculation traverses to leaf nodes.
        """
        # Create a cluster with all leaf nodes
        all_leaves = Cluster.combine(leaf_nodes)
        original_hash = all_leaves.hash

        # Create a new intermediate cluster with custom hash
        custom_intermediate = Cluster(
            intmap=intmap,
            leaves=[leaf_nodes[0], leaf_nodes[1]],
            hash=b"custom_intermediate_hash",
        )

        # Create another cluster combining the custom intermediate
        # with remaining leaves
        remaining_leaves = leaf_nodes[2:]
        combined_with_custom = Cluster.combine([custom_intermediate] + remaining_leaves)

        # The hash should be the SAME because hash calculation depends
        # only on leaf nodes, not on intermediate hashes
        assert combined_with_custom.hash == original_hash

        # Verify that the collections of leaf nodes are identical
        assert collect_all_leaves(all_leaves) == collect_all_leaves(
            combined_with_custom
        )

        # We can also confirm this isn't just coincidence by changing a leaf node
        # Create a modified leaf
        modified_leaf = Cluster(
            intmap=intmap,
            leaves=None,
            id=leaf_nodes[0].id,
            hash=b"modified_leaf_hash",
        )

        # Replace the first leaf with the modified one
        modified_group = [modified_leaf] + leaf_nodes[1:]
        modified_combined = Cluster.combine(modified_group)

        # This time the hash SHOULD be different because we changed a leaf node
        assert modified_combined.hash != original_hash

    def test_scale_performance(self, intmap: IntMap):
        """Test that the Cluster implementation performs well at scale."""
        # Number of terminal nodes to create
        num_nodes = 1_000_000

        # Create a large number of terminal nodes
        start_time = time.time()
        terminal_nodes = [
            Cluster(intmap=intmap, leaves=None, id=i, hash=f"hash{i}".encode())
            for i in range(num_nodes)
        ]
        creation_time = time.time() - start_time
        logger.debug(
            f"Created {num_nodes} terminal nodes in {creation_time:.2f} seconds"
        )

        # Test combining nodes in small batches first
        batch_size = 1000
        num_batches = num_nodes // batch_size

        start_time = time.time()
        level1_clusters = []
        for i in range(num_batches):
            batch = terminal_nodes[i * batch_size : (i + 1) * batch_size]
            # Create a cluster for each batch
            level1_cluster = Cluster.combine(batch)
            level1_clusters.append(level1_cluster)
        batching_time = time.time() - start_time
        logger.debug(
            f"Combined into {num_batches} level-1 clusters "
            f"in {batching_time:.2f} seconds"
        )

        # Verify we have the expected number of level-1 clusters
        assert len(level1_clusters) == num_batches

        # Now combine all level-1 clusters into a single top-level cluster
        start_time = time.time()
        top_cluster = Cluster.combine(level1_clusters)
        final_combination_time = time.time() - start_time
        logger.debug(
            "Combined all level-1 clusters into top-level cluster "
            f"in {final_combination_time:.2f} seconds"
        )

        # Verify the top cluster has a hash
        assert top_cluster.hash is not None

        # Verify the operation is consistent by creating a cluster directly
        # Combine a subset of terminal nodes (using a smaller sample for speed)
        sample_size = min(1000, num_nodes)
        sample_nodes = terminal_nodes[:sample_size]

        start_time = time.time()
        # Create a direct combination of the sample
        direct_combined = Cluster.combine(sample_nodes)
        direct_time = time.time() - start_time
        logger.debug(
            f"Directly combined {sample_size} nodes in {direct_time:.2f} seconds"
        )

        # Create a two-level combination of the same sample
        start_time = time.time()
        mid_batch_size = sample_size // 10
        mid_clusters = []
        for i in range(0, sample_size, mid_batch_size):
            batch = sample_nodes[i : i + mid_batch_size]
            mid_cluster = Cluster.combine(batch)
            mid_clusters.append(mid_cluster)
        hierarchical_combined = Cluster.combine(mid_clusters)
        hierarchical_time = time.time() - start_time
        logger.debug(
            f"Hierarchically combined {sample_size} nodes "
            f"in {hierarchical_time:.2f} seconds"
        )

        # Verify both approaches yield the same hash
        assert direct_combined.hash == hierarchical_combined.hash

        # Performance expectations (adjust as needed based on your hardware)
        # These are reasonable thresholds for modern hardware
        assert creation_time < 10.0, f"Creating {num_nodes:,} nodes took too long"
        assert batching_time < 30.0, "Batch combining took too long"
        assert final_combination_time < 20.0, "Final combination took too long"

        # Log a summary of total performance
        total_time = creation_time + batching_time + final_combination_time
        logger.debug(f"Total time for {num_nodes} nodes: {total_time:.2f} seconds")
        logger.debug(f"Time per node: {(total_time * 1000) / num_nodes:.3f} ms")
