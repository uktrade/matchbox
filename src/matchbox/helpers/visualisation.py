from collections import defaultdict
from itertools import count

import rustworkx as rx
from matplotlib.figure import Figure
from rustworkx.visualization import mpl_draw

from matchbox.server.base import MatchboxDBAdapter, inject_backend


@inject_backend
def draw_model_tree(backend: MatchboxDBAdapter) -> Figure:
    """
    Draws the model subgraph.
    """
    G = backend.get_model_subgraph()

    node_indices = G.node_indices()
    datasets = {
        G[node_indices[i]]["id"]: i
        for i in node_indices
        if G[node_indices[i]]["type"] == "dataset"
    }

    colours = []
    for i in node_indices:
        type = G[node_indices[i]]["type"]
        if type == "dataset":
            colours.append((0, 0, 1, 0.2))
        elif type == "model":
            colours.append((1, 0, 0, 0.2))

    return mpl_draw(
        G,
        pos=rx.spring_layout(
            G,
            pos={v: [0, i / 2] for v, i in enumerate(datasets.values())},
            fixed=set(datasets.values()),
        ),
        node_color=colours,
        with_labels=True,
        labels=lambda node: node["name"],
        edge_labels=lambda edge: edge["type"],
        font_size=8,
    )


def draw_data_tree(graph: rx.PyDiGraph) -> str:
    """
    Convert a rustworkx PyDiGraph to Mermaid graph visualization code.

    Args:
        graph (rx.PyDiGraph): A rustworkx directed graph with nodes containing 'id' and
            'type' attributes

    Returns:
        str: Mermaid graph definition code
    """
    mermaid_lines = ["graph LR"]

    counters = defaultdict(count, {"hash": count(1)})
    node_to_var = {}
    node_types = {}
    data_nodes = set()

    def format_id(id_value):
        """Format ID value, converting bytes to hex if needed."""
        if isinstance(id_value, bytes):
            return f"\\x{id_value.hex()}"
        return f"['{str(id_value)}']"

    for node_idx in graph.node_indices():
        node_data = graph.get_node_data(node_idx)
        if isinstance(node_data, dict):
            node_type = node_data.get("type", "")
            node_types[node_idx] = node_type
            if node_type == "data":
                data_nodes.add(node_idx)

    for node_idx, node_type in node_types.items():
        if node_type == "source":
            node_data = graph.get_node_data(node_idx)
            table_name = node_data["id"].split(".")[-1]
            node_to_var[node_idx] = table_name

            counter = count(1)
            for predecessor in graph.predecessor_indices(node_idx):
                if predecessor in data_nodes:
                    node_to_var[predecessor] = f"{table_name}{str(next(counter))}"
                    data_nodes.remove(predecessor)

    remaining_counter = count(len(node_to_var) + 1)
    for node_idx in data_nodes:
        node_to_var[node_idx] = str(next(remaining_counter))

    for node_idx, node_type in node_types.items():
        if node_type == "cluster":
            node_to_var[node_idx] = f"hash{next(counters['hash'])}"

    sources = []
    data_defs = []
    clusters = []

    for node_idx, node_type in node_types.items():
        node_data = graph.get_node_data(node_idx)
        var_name = node_to_var[node_idx]

        if node_type == "source":
            node_def = f'    {var_name}["{node_data["id"]}"]'
            sources.append(node_def)
        elif node_type == "data":
            node_label = format_id(node_data["id"])
            node_label = node_label.strip("[]'")
            node_def = f'    {var_name}["{node_label}"]'
            data_defs.append(node_def)
        elif node_type == "cluster":
            node_label = format_id(node_data["id"])
            node_def = f'    {var_name}["{node_label}"]'
            clusters.append(node_def)

    mermaid_lines.extend(sources)
    mermaid_lines.extend(data_defs)
    mermaid_lines.extend(clusters)

    mermaid_lines.append("")

    for edge in graph.edge_list():
        source = edge[0]
        target = edge[1]
        source_var = node_to_var[source]
        target_var = node_to_var[target]
        mermaid_lines.append(f"    {source_var} --> {target_var}")

    return "\n".join(mermaid_lines)
