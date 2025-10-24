# Evaluate model performance

After running models to deduplicate and link your data, you’ll likely want to check how well they’re performing. Matchbox helps with this by using **precision** and **recall**:

* **Precision**: How many of the matches your model made are actually correct?
* **Recall**: How many of the correct matches did your model successfully find?

To calculate these, we need a **ground truth** - a set of correct matches created by people. This is called **validation data**.

## Creating validation data

Matchbox provides a terminal-based UI to help you create this validation data. Here's how it works:

1. **Launch the evaluation tool** using `matchbox eval --collection <collection_name> --user <your_username>`
    a. You can also set your username with the `MB__CLIENT__USER` environment variable.
    b. Define `MB__CLIENT__DEFAULT_WAREHOUSE` (or use `--warehouse <connection_string>`) so the CLI can reach your warehouse.
    c. If your DAG isn't complete, include `--resolution <resolution_name>` to pick a specific run.
2. Matchbox will **download clusters** for you to review. It avoids clusters you've already judged and focuses on ones the model is unsure about.
3. In the terminal interface, you'll review each cluster:
   * Use keyboard commands like `b+1,3,5` to assign rows 1, 3, and 5 to group B
   * Press `space` to send your judgements to Matchbox
   * Skip to the next item without reviewing with `→` (right arrow)
   * Press `?` or `F1` for help with commands

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

Download validation data:

```python
from matchbox.client.eval import EvalData
eval_data = EvalData()
```

Check precision and recall at a specific threshold:

```python
precision, recall = eval_data.precision_recall(results, threshold=0.5)
```

Compare multiple model resolutions:

```python
from matchbox.client.eval import compare_models, ModelResolutionPath

comparison = compare_models([
    ModelResolutionPath(collection="companies", run=1, name="deduper"),
    ModelResolutionPath(collection="companies", run=2, name="deduper"),
])
```

!!! tip "Deterministic models"
    Some types of model (like the `NaiveDeduper` used in the example) only output 1s for the matches they make, hence **threshold truth tuning doesn't apply**:

    * You won't get a precision-recall curve, but a single point at threshold 1.0.
    * The precision and recall scores will be the same at all thresholds.

    On the other hand, probabilistic models (like `SplinkLinker`), can output **any value between 0.0 and 1.0**.
    
