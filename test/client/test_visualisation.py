from unittest.mock import patch

from matplotlib.figure import Figure

from matchbox.client.visualisation import draw_resolution_graph


@patch("matchbox.client.visualisation.get_resolution_graph")
def test_draw_resolution_graph(handler_func, resolution_graph):
    handler_func.return_value = resolution_graph

    plt = draw_resolution_graph()
    assert isinstance(plt, Figure)
