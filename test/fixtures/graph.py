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
            ResolutionNode(id=1, name="1", type=ResType.SOURCE),
            ResolutionNode(id=2, name="2", type=ResType.SOURCE),
            ResolutionNode(id=3, name="3", type=ResType.MODEL),
            ResolutionNode(id=4, name="4", type=ResType.MODEL),
            ResolutionNode(id=5, name="5", type=ResType.MODEL),
        },
        edges={
            ResolutionEdge(parent=1, child=3),
            ResolutionEdge(parent=2, child=5),
            ResolutionEdge(parent=3, child=4),
            ResolutionEdge(parent=4, child=5),
        },
    )

    return res_graph


@pytest.fixture
def pydigraph() -> PyDiGraph:
    G = PyDiGraph()
    n1 = G.add_node({"id": 1, "name": "1", "type": str(ResType.SOURCE)})
    n2 = G.add_node({"id": 2, "name": "2", "type": str(ResType.SOURCE)})
    n3 = G.add_node({"id": 3, "name": "3", "type": str(ResType.MODEL)})
    n4 = G.add_node({"id": 4, "name": "4", "type": str(ResType.MODEL)})
    n5 = G.add_node({"id": 5, "name": "5", "type": str(ResType.MODEL)})
    G.add_edge(n1, n3, {})
    G.add_edge(n2, n5, {})
    G.add_edge(n3, n4, {})
    G.add_edge(n4, n5, {})
    return G
