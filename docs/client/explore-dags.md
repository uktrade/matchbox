Matchbox lets you link many sources of data in many different ways. But when you query it, which way should you choose?

A DAG (i.e. a Matchbox pipeline) represents a queriable state describing how to cluster entities from one or more data sources. It brings together deduplicated and linked data sources.

You can explore which DAGs are stored on Matchbox.


```python
from matchbox.client.dags import DAG

DAG.list_all()
```


## Download a DAG locally

A DAG also stores serialisable descriptions of all the steps that produced a queriable state. You can [create your own DAGs](link-data.md) and push them to the Matchbox server. But you can also re-construct a DAG locally based on its server snapshot:

```python
dag = DAG(name="companies").load_default()
```

This will load the "default" version of a DAG, if available. A DAG can only have one version set as "default". There may also be a "pending" version of a DAG that's currently being worked on:

```
dag = DAG(name="companies").load_pending()
```

Once you have retrieved your DAG's structure, you can inspect it:

```python
print(dag.draw())
```

This will print the hierarchy of nodes making up the DAG. You can retrieve individual nodes and inspect their configuration:

```python
source1 = dag.get_source("source1")
dedupe_source1 = dag.get_model("dedupe_source1")
```
