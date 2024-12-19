import pytest
from rustworkx import PyDiGraph

from matchbox.common.graph import (
    ResolutionEdge,
    ResolutionGraph,
    ResolutionNode,
)
from matchbox.common.graph import (
    ResolutionNodeType as ResType,
)


@pytest.fixture
def resolution_graph() -> ResolutionGraph:
    res_graph = ResolutionGraph(
        nodes={
            ResolutionNode(id=2, name="2", type=ResType.DATASET),
            ResolutionNode(id=1, name="1", type=ResType.DATASET),
            ResolutionNode(id=3, name="3", type=ResType.MODEL),
            ResolutionNode(id=4, name="4", type=ResType.MODEL),
            ResolutionNode(id=5, name="5", type=ResType.MODEL),
        },
        edges={
            ResolutionEdge(parent=2, child=1),
            ResolutionEdge(parent=4, child=3),
            ResolutionEdge(parent=5, child=2),
            ResolutionEdge(parent=5, child=4),
        },
    )

    return res_graph


@pytest.fixture
def pydigraph() -> PyDiGraph:
    G = PyDiGraph()
    n1 = G.add_node({"id": 1, "name": "1", "type": str(ResType.DATASET)})
    n2 = G.add_node({"id": 2, "name": "2", "type": str(ResType.DATASET)})
    n3 = G.add_node({"id": 3, "name": "3", "type": str(ResType.MODEL)})
    n4 = G.add_node({"id": 4, "name": "4", "type": str(ResType.MODEL)})
    n5 = G.add_node({"id": 5, "name": "5", "type": str(ResType.MODEL)})
    G.add_edge(n2, n1, {})
    G.add_edge(n4, n3, {})
    G.add_edge(n5, n2, {})
    G.add_edge(n5, n4, {})
    return G
