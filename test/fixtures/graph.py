import pytest
from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
)
from matchbox.common.graph import (
    ResolutionNodeKind as ResKind,
)
from matchbox.common.hash import hash_to_str
from rustworkx import PyDiGraph


@pytest.fixture
def resolution_graph() -> ResolutionGraph:
    res_graph = ResolutionGraph(
        nodes={
            ResolutionNode(hash=bytes(1), name="1", kind=ResKind.DATASET),
            ResolutionNode(hash=bytes(2), name="2", kind=ResKind.DATASET),
            ResolutionNode(hash=bytes(3), name="3", kind=ResKind.MODEL),
            ResolutionNode(hash=bytes(4), name="4", kind=ResKind.MODEL),
            ResolutionNode(hash=bytes(5), name="5", kind=ResKind.MODEL),
        },
        edges={
            ResolutionEdge(parent=bytes(2), child=bytes(1)),
            ResolutionEdge(parent=bytes(4), child=bytes(3)),
            ResolutionEdge(parent=bytes(5), child=bytes(2)),
            ResolutionEdge(parent=bytes(5), child=bytes(4)),
        },
    )

    return res_graph


@pytest.fixture
def pydigraph() -> PyDiGraph:
    def make_id(n: int) -> str:
        return hash_to_str(bytes(n))

    G = PyDiGraph()
    n1 = G.add_node({"id": make_id(1), "name": "1", "kind": str(ResKind.DATASET)})
    n2 = G.add_node({"id": make_id(2), "name": "2", "kind": str(ResKind.DATASET)})
    n3 = G.add_node({"id": make_id(3), "name": "3", "kind": str(ResKind.MODEL)})
    n4 = G.add_node({"id": make_id(4), "name": "4", "kind": str(ResKind.MODEL)})
    n5 = G.add_node({"id": make_id(5), "name": "5", "kind": str(ResKind.MODEL)})
    G.add_edge(n2, n1, {})
    G.add_edge(n4, n3, {})
    G.add_edge(n5, n2, {})
    G.add_edge(n5, n4, {})
    return G
