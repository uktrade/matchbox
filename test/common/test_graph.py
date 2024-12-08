from typing import Any

from matchbox.common.graph import ResolutionGraph
from rustworkx import PyDiGraph, is_isomorphic


def test_res_graph_to_rx(resolution_graph: ResolutionGraph, pydigraph: PyDiGraph):
    G: PyDiGraph = resolution_graph.to_rx()

    def same(x: Any, y: Any) -> bool:
        return x == y

    assert is_isomorphic(G, pydigraph, node_matcher=same, edge_matcher=same)
