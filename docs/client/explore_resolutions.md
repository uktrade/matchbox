Matchbox lets you link many sources of data in many different ways. But when you query it, which way should you choose?

A *resolution*, or *point of resolution* represents a queriable state describing how to cluster entities from one or more data sources. A resolution can represent an original data source, a deduplicated data source, or the result of linking two or more resolutions.

In order to explore which resolutions are stored on Matchbox, you can use the following client method:

=== "Example"
    ```python
    from matchbox.client.visualisation import draw_resolution_graph

    draw_resolution_graph()
    ```
