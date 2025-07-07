"""Visualisation utilities."""

import matplotlib.pyplot as plt
import rustworkx as rx
from matplotlib.figure import Figure
from rustworkx.visualization import mpl_draw

from matchbox.client._handler import get_resolution_graph
from matchbox.common.dtos import ResolutionName
from matchbox.common.graph import ResolutionNodeType


def draw_resolution_graph(contains: ResolutionName | None = None) -> Figure:
    """Draws the resolution graph layer by layer, component by component.

    contains: An optional filter to only include components containing a
        specific source or model.
    """
    plt.ioff()  # Turn off interactive plotting to prevent auto-display

    G: rx.PyDiGraph = get_resolution_graph().to_rx()

    # Filter components if contains is specified
    components: list[set[int]] = []
    if contains is not None:
        target_nodes = G.filter_nodes(lambda node: node.get("name") == contains)

        if target_nodes:
            # Get all components and find those containing any target node
            all_components = rx.weakly_connected_components(G)
            components = [
                comp
                for comp in all_components
                if any(node in comp for node in target_nodes)
            ]
    else:
        components = rx.weakly_connected_components(G)
        if len(components) <= 1:
            components = [set(G.node_indices())]

    # Handle empty graph or no matching components
    if not components or not any(components):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        message = (
            f'No component found containing "{contains}"'
            if contains
            else "No nodes in resolution graph"
        )
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    else:
        n_components = len(components)
        n_cols = min(2, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for comp_idx, component in enumerate(components):
        subgraph = G.subgraph(list(component))

        source_nodes = [
            node
            for node in subgraph.node_indices()
            if subgraph[node]["type"] == ResolutionNodeType.SOURCE.value
        ]
        layers = rx.layers(subgraph, source_nodes, index_output=True)

        positions = {}
        for layer_idx, layer_nodes in enumerate(layers):
            y_pos = -layer_idx * 3.0
            if len(layer_nodes) == 1:
                positions[layer_nodes[0]] = (0, y_pos)
            else:
                for i, node in enumerate(layer_nodes):
                    x_pos = (
                        i - (len(layer_nodes) - 1) / 2
                    ) * 4.0  # Give text some space
                    positions[node] = (x_pos, y_pos)

        colors = [
            (0.2, 0.6, 1.0, 0.8)  # RGBA blue
            if subgraph[i]["type"] == ResolutionNodeType.SOURCE.value
            else (1.0, 0.4, 0.2, 0.8)  # RGBA orange
            for i in subgraph.node_indices()
        ]

        mpl_draw(
            subgraph,
            pos=positions,
            ax=axes[comp_idx],
            node_color=colors,
            with_labels=True,
            labels=lambda node: node.get("name", str(node.get("id", ""))),
            arrows=True,
        )

        axes[comp_idx].set_aspect("equal")

        # Calculate bounds and add padding for text
        if positions:
            x_coords = [pos[0] for pos in positions.values()]
            y_coords = [pos[1] for pos in positions.values()]
            x_range = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 4.0
            y_range = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 3.0

            # Add substantial padding for text labels
            x_padding = max(2.0, x_range * 0.3)
            y_padding = max(1.5, y_range * 0.2)

            axes[comp_idx].set_xlim(
                min(x_coords) - x_padding, max(x_coords) + x_padding
            )
            axes[comp_idx].set_ylim(
                min(y_coords) - y_padding, max(y_coords) + y_padding
            )

    for idx in range(len(components), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig
