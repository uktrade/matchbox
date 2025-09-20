Matchbox lets you link many sources of data in many different ways. But when you query it, which way should you choose?

A DAG (i.e. a Matchbox pipeline) represents a queriable state describing how to cluster entities from one or more data sources. It brings together deduplicated and linked data sources.

You can explore which DAGs are stored on Matchbox.

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    DAG.list_all()
    ```
