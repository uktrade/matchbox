from unittest.mock import patch

from matchbox.client.visualisation import draw_resolution_graph
from matplotlib.figure import Figure


@patch("matchbox.client.visualisation._get_resolution_graph")
def test_draw_resolution_graph(handler_func, resolution_graph):
    handler_func.return_value = resolution_graph.model_dump_json()

    plt = draw_resolution_graph()
    assert isinstance(plt, Figure)
