# Evaluate model performance

After running models to deduplicate and link your data, you’ll likely want to check how well they’re performing. Matchbox helps with this by using **precision** and **recall**:

* **Precision**: How many of the matches your model made are actually correct?
* **Recall**: How many of the correct matches did your model successfully find?

To calculate these, we need a **ground truth** - a set of correct matches created by people. This is called **validation data**.

## Creating a sample set

First, you need to create a local set of sample clusters that you want to evaluate. If you're evaluating a single model:

```python
dag = ... # your DAG, defined elsewhere 
dag.resolve().as_dump().write_parquet("samples.pq")
```

Or, if you're comparing two models:

```python
dag = ... # your DAG, defined elsewhere

resolved_model1 = dag.resolve("model1")
resolved_model2 = dag.resolve("model2")
# Forms clusters as the union of the clusters output by each model
resolved_joint = resolved_model1.merge(resolved_model2)
resolved_joint.as_dump().write_parquet("samples.pq")
```

## Creating validation data

Matchbox provides a terminal-based UI to help you create this validation data. Here's how it works:

1. **Launch the evaluation tool** using `matchbox eval --collection <collection_name> --user <your_username> --file <path_to_samples>`
    - Define `MB__CLIENT__DEFAULT_WAREHOUSE` (or use `--warehouse <connection_string>`) so the CLI can reach your warehouse.
    - Use `--tag <session_tag>` to tag your judgements so they can be filtered later
2. Matchbox will **load clusters** from your warehouse, for you to review.  
3. In the terminal interface, you will review each cluster.  

Once enough users have reviewed clusters, this data can be used to evaluate model performance.

## How precision and recall are calculated

Validation data is created at the **cluster level**, but precision and recall are calculated using **pairs of records**.

For example, if a model links records A, B, C, and D into one cluster, it implies these pairs:

> A-B, A-C, A-D, B-C, B-D, C-D

If a user says only A, B, and C belong together, the correct pairs are:

> A-B, A-C, B-C

So, the model found all the correct pairs (good recall), but also included some incorrect ones (lower precision).

If users disagree on a pair:

* If more users approve it than reject it → it’s considered **correct**.
* If more reject it → it’s **incorrect**.
* If it’s a tie → it’s **ignored** in the calculation.

Only pairs that appear in both the model and the validation data are used in the evaluation.

!!! tip "Relative vs. absolute scores"
    If your model builds on others, it may inherit some incorrect pairs it can’t control. So, precision and recall scores aren’t absolute - they’re best used to **compare models or thresholds**.

## Tuning your model’s threshold

Choosing the right threshold for your model involves balancing precision and recall. A higher threshold usually means:

* **Higher precision** (fewer false matches)
* **Lower recall** (more missed matches)

To evaluate this in code, first you need to build and run a model outside of a DAG:

```python
from matchbox.client.models.dedupers import NaiveDeduper
from sqlalchemy import create_engine

engine = create_engine('postgresql://')

dag = DAG(name="companies").new_run()

source = dag.source(...) # source parameters must be completed

model = source.query().deduper(
    name="model_name",
    description=f"description",
    model_class=NaiveDeduper,
    model_settings={
        "unique_fields": ["field1", "field2"],
    },
)

results = model.run()
```

Download validation data, filtering by tag:

```python
from matchbox.client.eval import EvalData
eval_data = EvalData(tag="companies__15_02_2025")
```

Check precision and recall at a specific threshold:

```python
precision, recall = eval_data.precision_recall(results, threshold=0.5)
```

!!! tip "Deterministic models"
    Some types of model (like the `NaiveDeduper` used in the example) only output 1s for the matches they make, hence **threshold truth tuning doesn't apply**:

    * You won't get a precision-recall curve, but a single point at threshold 1.0.
    * The precision and recall scores will be the same at all thresholds.

    On the other hand, probabilistic models (like `SplinkLinker`), can output **any value between 0.0 and 1.0**.
    
