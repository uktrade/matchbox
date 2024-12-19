from enum import StrEnum

import rustworkx as rx
from pydantic import BaseModel


class ResolutionNodeType(StrEnum):
    DATASET = "dataset"
    MODEL = "model"
    HUMAN = "human"


class ResolutionNode(BaseModel):
    id: int
    name: str
    type: ResolutionNodeType

    def __hash__(self):
        return hash(self.id)


class ResolutionEdge(BaseModel):
    parent: int
    child: int

    def __hash__(self):
        return hash((self.parent, self.child))


class ResolutionGraph(BaseModel):
    nodes: set[ResolutionNode]
    edges: set[ResolutionEdge]

    def to_rx(self) -> rx.PyDiGraph:
        nodes = {}
        G = rx.PyDiGraph()
        for n in self.nodes:
            node_data = {
                "id": n.id,
                "name": n.name,
                "type": str(n.type),
            }
            nodes[n.id] = G.add_node(node_data)
        for e in self.edges:
            G.add_edge(nodes[e.parent], nodes[e.child], {})
        return G
