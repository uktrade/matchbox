"""Common data structures for resolution graphs."""

from enum import StrEnum
from typing import TypeAlias

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

SourceResolutionName: TypeAlias = str
"""Type alias for source resolution names."""

ModelResolutionName: TypeAlias = str
"""Type alias for model resolution names."""

ResolutionName = SourceResolutionName | ModelResolutionName
"""Type alias for resolution names."""

DEFAULT_RESOLUTION: ResolutionName = "__DEFAULT__"


class ResolutionType(StrEnum):
    """Types of nodes in a resolution."""

    SOURCE = "source"
    MODEL = "model"


class ResolutionNode(BaseModel):
    """A node in a resolution graph."""

    model_config = ConfigDict(frozen=True)

    id: int
    name: ResolutionName
    type: ResolutionType


class ResolutionEdge(BaseModel):
    """An edge in a resolution graph."""

    model_config = ConfigDict(frozen=True)

    parent: int
    child: int


class ResolutionGraph(BaseModel):
    """A directed graph of resolution nodes and edges."""

    nodes: set[ResolutionNode]
    edges: set[ResolutionEdge]

    def to_rx(self) -> rx.PyDiGraph:
        """Convert the resolution graph to a rustworkx directed graph."""
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
