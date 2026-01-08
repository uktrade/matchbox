When building and debugging a DAG for the first time, you might want to do it interactively, such as in a Python notebook.

## Manually running and syncing nodes
In an interactive setting, it can be useful to run and sync nodes manually, one by one, to inspect intermediate outputs. A node must be run before it's synced. Before a node can be run, its dependencies must be run and synced. Printing the DAG structure with `dag.draw()` helps you see a node's dependencies.

```python
source1.run()
source1.sync()
dedupe_source1.run()
dedupe_source1.sync()
...
```

You can also run only part of the DAG, based on the node execution order output by `dag.draw()`, which must be read from bottom to top:

```python
dag.run_and_sync(start="node1", finish="node2")
```

## Iterating on sources

After defining a source, you can validate your extract-transform logic by peeking at a small sample:

```python
source1 = dag.source(...)
source1.sample()

# Update source
source1.extract_transform = new_sql
source1.sample()
```

By default, a Polars dataframe is returned. Other return types are supported. For example:

```python
source1.sample(return_type="pandas")
```

Once you're happy, you can run the source node. Source data is cached to disk so that queries using that source don't need to retrieve data again. However, re-running the source node manually updates the disk cache. If you change the definition of a a node, including a source node, you must re-run it and sync it for downstream queries to be correct.

## Iterating on queries

After running your source node, you can create a query from it, and inspect its data:

```python
source1_query = source1.query()
source1_query.data() # query data and clean it
source1_query.data_raw() # query data without cleaning it
```

The `return_type` argument can also be passed to the `.data()` and `.data_raw()` query methods.

You can iterate on the cleaning logic:

```python
# Re-use raw query data to avoid fetching data from the Matchbox server repeatedly
raw_query_data = source1_query.data_raw() # must use the default return_type

# Update cleaning rules
source1_query.cleaning = new_cleaning

# Try out the new cleaning
source1_query.data(raw_query_data)
```

## Iterating on models

Running a model returns [`ModelResults`][matchbox.client.results.ModelResults] that can be inspected. The same results are accessible using the attribute `results` on a model node which has been run:

```python
results = dedupe_source1.run()
# After running the model, the following also works:
results = dedupe_source1.results
```

You can iterate on the model logic:

```python
# Re-use query data to avoid processing query repeatedly
dedupe_data = dedupe1.query(source1).data()

# Update model
dedupe_source1.model_settings["unique_fields"] = ["a", "b"]

# Try out the new model
dedupe_source1.run(dedupe_data)
```

If you're working on a linker and want to apply this pattern, you need to pass two dataframes:

```python
link_source1_source2.run(link_source1_data, link_source2_data)
```


## Wholesale node update

Instead of updating single node attributes, you can overwrite a whole node in a DAG by creating a new node with the same name:

```python
source1 = dag.source("source1", location=location1, ...)
source1 = dag.source("source1", location=location2, ...)
```