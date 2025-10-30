from matchbox.common.transform import DisjointSet


class TestDisjointSet:
    def test_disjoint_set_empty(self) -> None:
        dsj = DisjointSet()

        assert dsj.get_components() == []

    def test_disjoint_set_same(self) -> None:
        dsj = DisjointSet()
        dsj.union(1, 1)

        assert dsj.get_components() == [{1}]

    def test_disjoint_set_redundant(self) -> None:
        dsj = DisjointSet()
        dsj.union(1, 2)

        assert dsj.get_components() == [{1, 2}]

        dsj.union(2, 1)

        assert dsj.get_components() == [{1, 2}]

    def test_disjoint_set_union(self) -> None:
        dsj = DisjointSet()
        dsj.union(1, 2)
        dsj.union(3, 4)
        dsj.union(5, 6)

        assert sorted(dsj.get_components()) == [{1, 2}, {3, 4}, {5, 6}]

        dsj.union(2, 3)
        dsj.union(4, 5)

        assert dsj.get_components() == [{1, 2, 3, 4, 5, 6}]

    def test_disjoint_set_add_single(self) -> None:
        dsj = DisjointSet()
        dsj.add(1)

        assert dsj.get_components() == [{1}]

    def test_disjoint_set_add_multiple(self) -> None:
        dsj = DisjointSet()
        dsj.add(1)
        dsj.add(2)
        dsj.add(3)

        # Get components and ensure we have the expected singletons
        components = dsj.get_components()
        assert len(components) == 3
        assert {1} in components
        assert {2} in components
        assert {3} in components

    def test_disjoint_set_add_existing(self) -> None:
        dsj = DisjointSet()
        dsj.add(1)
        dsj.add(1)  # Add the same element again

        assert dsj.get_components() == [{1}]  # Should still be just one component

    def test_disjoint_set_add_after_union(self) -> None:
        dsj = DisjointSet()
        dsj.union(1, 2)
        dsj.add(1)  # Add an element that's already in a union

        assert dsj.get_components() == [{1, 2}]  # Should not change existing structure

    def test_disjoint_set_union_after_add(self) -> None:
        dsj = DisjointSet()
        dsj.add(1)
        dsj.add(2)
        dsj.union(1, 2)

        assert dsj.get_components() == [{1, 2}]

    def test_disjoint_set_mixed_operations(self) -> None:
        dsj = DisjointSet()
        dsj.add(1)
        dsj.add(2)
        dsj.union(3, 4)  # Creates {3,4} implicitly
        dsj.add(5)

        # Check that the expected components are present
        components = dsj.get_components()
        assert len(components) == 4
        assert {1} in components
        assert {2} in components
        assert {3, 4} in components
        assert {5} in components
