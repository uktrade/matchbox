from httpx import Response
from matplotlib.figure import Figure
from respx import MockRouter

from matchbox.client.visualisation import draw_resolution_graph
from matchbox.common.graph import ResolutionGraph


def test_draw_resolution_graph(
    matchbox_api: MockRouter, resolution_graph: ResolutionGraph
):
    matchbox_api.get("/report/resolutions").mock(
        return_value=Response(200, content=resolution_graph.model_dump_json()),
    )

    plt = draw_resolution_graph()
    assert isinstance(plt, Figure)


def test_filter_draw_resolution_graph(
    matchbox_api: MockRouter, resolution_graph: ResolutionGraph
):
    matchbox_api.get("/report/resolutions").mock(
        return_value=Response(200, content=resolution_graph.model_dump_json()),
    )

    plt = draw_resolution_graph(contains="1")
    assert isinstance(plt, Figure)


def test_no_result_draw_resolution_graph(
    matchbox_api: MockRouter, resolution_graph: ResolutionGraph
):
    matchbox_api.get("/report/resolutions").mock(
        return_value=Response(200, content=resolution_graph.model_dump_json()),
    )

    plt = draw_resolution_graph(contains="not_found")
    assert isinstance(plt, Figure)
