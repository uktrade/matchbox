import pytest

from matchbox.common.hash import IntMap
from matchbox.common.transform import Cluster, DisjointSet


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

    def test_add_leaves_to_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test adding leaf nodes to a DisjointSet."""
        # Create a DisjointSet and add leaves
        djs = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs.add(node)

        # Verify all nodes are in separate components
        components = djs.get_components()
        assert len(components) == 6
        for component in components:
            assert len(component) == 1

    def test_union_leaves_in_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test creating unions of leaf nodes in DisjointSet."""
        djs = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs.add(node)

        # Create three pairs
        djs.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        # Verify we now have three components
        components = djs.get_components()
        assert len(components) == 3

        # Verify each component has the expected size
        component_sizes = [len(comp) for comp in components]
        assert sorted(component_sizes) == [2, 2, 2]

    def test_create_clusters_from_components(self, leaf_nodes: list[Cluster]):
        """Test creating new clusters from DisjointSet components."""
        # Add leaves to DisjointSet and create unions
        djs = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs.add(node)

        djs.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        # Create clusters from components
        components = djs.get_components()
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

    def test_level1_clusters_in_disjoint_set(self, leaf_nodes: list[Cluster]):
        """Test using level-1 clusters in another DisjointSet."""
        # Create level-1 clusters as in previous test
        djs1 = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs1.add(node)

        djs1.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs1.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs1.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        components = djs1.get_components()
        level1_clusters = []
        for component in components:
            component_list = list(component)
            if len(component_list) == 1:
                cluster = component_list[0]
            else:
                cluster = Cluster.combine(component_list)
            level1_clusters.append(cluster)

        # Add these clusters to a new DisjointSet
        djs2 = DisjointSet[Cluster]()
        for cluster in level1_clusters:
            djs2.add(cluster)

        # Verify initial state
        assert len(djs2.get_components()) == 3

        # Create a union of first two clusters
        djs2.union(level1_clusters[0], level1_clusters[1])

        # Verify we now have two components
        components2 = djs2.get_components()
        assert len(components2) == 2

        # One component should have 2 clusters, the other should have 1
        component_sizes = [len(comp) for comp in components2]
        assert sorted(component_sizes) == [1, 2]

    def test_leaf_preservation_in_hierarchy(self, leaf_nodes: list[Cluster]):
        """Test that leaf nodes are preserved in the cluster hierarchy."""
        # Follow the same steps as in previous tests to build a hierarchy
        # Step 1: Create level-1 clusters
        djs1 = DisjointSet[Cluster]()
        for node in leaf_nodes:
            djs1.add(node)

        djs1.union(leaf_nodes[0], leaf_nodes[1])  # 1,2
        djs1.union(leaf_nodes[2], leaf_nodes[3])  # 3,4
        djs1.union(leaf_nodes[4], leaf_nodes[5])  # 5,6

        components1 = djs1.get_components()
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
        djs2 = DisjointSet[Cluster]()
        for cluster in level1_clusters:
            djs2.add(cluster)

        djs2.union(level1_clusters[0], level1_clusters[1])

        # Find the component with multiple clusters
        larger_component = None
        for component in djs2.get_components():
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
        """Test that combine works correctly with a single cluster.

        When Cluster.combine() receives only one cluster but with a new probability,
        it should pass that probability on.
        """
        # Take a leaf node (which has probability=None)
        original_cluster = leaf_nodes[0]
        assert original_cluster.probability is None

        # Combine it with itself but provide a new probability
        result = Cluster.combine([original_cluster], probability=85)

        # Should get a new cluster with the specified probability
        assert result.probability == 85
        assert result.id == original_cluster.id
        assert result.hash == original_cluster.hash
        assert result.leaves == original_cluster.leaves

        # Should be a different object (not the original)
        assert result is not original_cluster

        # Test with probability=0 (edge case)
        result_zero = Cluster.combine([original_cluster], probability=0)
        assert result_zero.probability == 0

        # Test preserving original probability when None provided
        result_none = Cluster.combine([original_cluster], probability=None)
        assert result_none.probability is None

        assert result_none.id == original_cluster.id
        assert result_none.hash == original_cluster.hash
        assert result_none.leaves == original_cluster.leaves

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

    def test_hash_generation_for_complex_hierarchy(self, leaf_nodes: list[Cluster]):
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

    def test_combine_single_cluster_with_new_probability(
        self, leaf_nodes: list[Cluster]
    ):
        """Test that combine respects new probability even with single cluster.

        When Cluster.combine() receives only one cluster but with a new probability,
        it should create a new cluster with that probability rather than returning
        the original cluster unchanged.
        """
        # Take a leaf node (which has probability=None)
        original_cluster = leaf_nodes[0]
        assert original_cluster.probability is None

        # Combine it with itself but provide a new probability
        result = Cluster.combine([original_cluster], probability=85)

        # Should get a new cluster with the specified probability
        assert result.probability == 85
        assert result.id == original_cluster.id
        assert result.hash == original_cluster.hash
        assert result.leaves == original_cluster.leaves

        # Should be a different object (not the original)
        assert result is not original_cluster

        # Test with probability=0 (edge case)
        result_zero = Cluster.combine([original_cluster], probability=0)
        assert result_zero.probability == 0

        # Test preserving original probability when None provided
        result_none = Cluster.combine([original_cluster], probability=None)
        assert result_none.probability is None

        assert result_none.id == original_cluster.id
        assert result_none.hash == original_cluster.hash
        assert result_none.leaves == original_cluster.leaves
