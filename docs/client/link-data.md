# Building with Matchbox DAGs

Data matching and entity resolution are complex tasks that often require multiple processing steps. Matchbox provides a powerful Directed Acyclic Graph (DAG) framework that allows you to define and run sophisticated matching pipelines with clearly defined dependencies.

This guide walks through creating complete matching pipelines using the Matchbox DAG API, covering everything from [defining data sources](#2-defining-data-sources) to [executing complex multi-step matching processes](#advanced-use-cases). In our examples we'll be referencing publicly available datasets about UK companies, specifically [Companies House data](https://find-and-update.company-information.service.gov.uk), and [UK trade data](https://www.uktradeinfo.com).

## Understanding DAGs in Matchbox

A DAG (Directed Acyclic Graph) represents a sequence of operations where each step depends on the outputs of previous steps, without any circular dependencies. In Matchbox, a DAG consists of:

* [`Source`s][matchbox.client.sources.Source]: indexing data from sources
* [`Model`s][matchbox.client.models.Model]: Removing duplicates within one data input, or linking two data inputs. As data inputs, `Model`s can take `Source`s or other `Model`s.

`Source`s and `Model`s can form [`Query`][matchbox.client.models.Model] objects, which allow you to retrieve the version of the data implied by that DAG step. Querying a source gives you the records in that source, and querying from a model gives you the deduplicated or linked records at that point in the DAG. When querying from a model, you need to specify which sources you want to query from that model's lineage.

```python
source: Source
deduper: Model
# ... define your source and a model deduplicating it ...
source_query = source.query()
model_query = deduper.query(source)
```

`Model`s are formed from `Query` objects.

```python
other_source: Source
# ... define your second source ...
deduper = source.query().deduper(...)
linker = deduper.query().linker(other_source.query())
```

All these objects are lazy: they don't actually retrieve any data unless you run them, for example:

```python
queried_data = query.data()
deduper_results = deduper.run()
linker_results = linker.run()
```

The steps need to be run in order, but once you've finalised your DAG, it's better to automatically run all of them using a single DAG command, as is shown later.

## Setting up your environment

Before building a pipeline, it's worth configuring logging:

=== "Example"
    ```python
    import logging

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matchbox").setLevel(logging.INFO)
    ```

You will also need to define the engine to read your data sources:

=== "Example"
    ```python
    # Get your database engine
    from sqlalchemy import create_engine
    engine = create_engine("postgresql://user:password@host:port/")
    ```

## 1. Defining a new DAG

You're now ready to create your first [`DAG`][matchbox.client.dags.DAG].

=== "Example"
    ```python
    from matchbox.client.dags import DAG
    dag = DAG(name="companies").new_run()
    ```

A DAG needs a name, which will be used to identify this DAG once you publish it to the Matchbox server. You also need to use the `.new_run()` method to prepare the DAG to send results to the server.

This DAG will own all the sources and models you define later.

## 2. Defining data sources

Now you can define your data sources. Each source represents data that will be used in the matching process.

The `index_fields` are what Matchbox will use to store a reference to your data, and are the only fields it will permit you to match on.

The `key_field` is the field in your source that contains some unique code that identifies each entity. For example, in a relational database, this would typically be your primary key.

=== "Example"
    ```python
    from matchbox.client import RelationalDBLocation

    warehouse = RelationalDBLocation(name="dbname").set_client(engine)

    # Companies House data
    companies_house = dag.source(
        name="companies_house",
        location=warehouse,
        extract_transform="""
            select
                number::str as company_number,
                company_name,
                upper(postcode) as postcode,
            from
                companieshouse.companies;
        """,
        infer_types=True,
        index_fields=["company_name", "company_number", "postcode"],
        key_field="company_number",
    )
    
    # Exporters data
    exporters = dag.source(
        name="hmrc_exporters",
        location=warehouse,
        extract_transform="""
            select
                id,
                company_name,
                upper(postcode) as postcode,
            from
                hmrc.trade__exporters;
        """,
        infer_types=True,
        index_fields=["company_name", "postcode"],
        key_field="id",
    )
    ```

Each [`Source`][matchbox.client.sources.Source] object requires:

- A `location`, such as [`RelationalDBLocation`][matchbox.client.locations.RelationalDBLocation]. This will need a `name`, and `client`
    - The name of a location is a way of tagging it, such that later on you can filter sources you want to retrieve from the server
    - For a relational database, a SQLAlchemy engine is your client
- An `extract_transform` string, which will take data from the location and transform it into your key and index fields. Its syntax will depend on the type of location
    - For most a relational database location, this will be SQL
- A list of `index_fields` that will be used for matching
    - These must be found in the result of the `extract_transform` logic
- A key field (`key_field`) that uniquely identifies each record
    - This must be found in the result of the `extract_transform` logic

## 3. Creating dedupers

Dedupe steps identify and resolve duplicates within a single source.

=== "Example"
    ```python
    from matchbox.client.models.dedupers.naive import NaiveDeduper

    dedupe_companies_house = companies_house.query(
        cleaning={
            "company_name": f"lower({companies_house.f('company_name')})",
            "company_number": companies_house.f("company_number"),
        }
    ).deduper(
        name="naive_companieshouse_companies",
        description="Deduplication based on company name",
        model_class=NaiveDeduper,
        model_settings={
            "unique_fields": ["company_name", "company_number],
        },
        truth=1.0,
    )
    ```

A query can optionally take instructions on how to clean the data. These are defined using a dictionary where:

* the dictionary **key** is the desired column name that will be output
* the dictionary **value** is a SQL expression in DuckDB format

!!! warning "Only cleaned columns are passed through"
    
    When you specify a `cleaning` dictionary, **only the columns you explicitly include** (plus `id`, `leaf_id`, and key columns) will be passed through to the next step. Any columns not mentioned in the cleaning dictionary will be dropped.
    
    This means you must include **all fields** you need for your model in the cleaning dictionary, even if you're just passing them through unchanged.

A deduper takes:

- A unique `name` for the step
- An optional `description` explaining the purpose of the step
- The deduplication algorithm to use (`model_class`)
- Configuration settings (`model_settings`) for the algorithm
- Optionally, a `truth` threshold (a float between `0.0` and `1.0`) above which a match is considered "true". By default, this is set to `1.0`. This value is only relevant when using a model that can output matches with different confidence scores, which is not the case for a `NaiveDeduper`.

### On cleaning

!!! tip "Always clean all fields you need"
    
    Since only columns in the cleaning dictionary are passed through, you should include **all fields** required by your model - even if you're just aliasing them without transformation. This makes your configuration explicit and prevents fields from being accidentally dropped.
    
    ```python
    # ❌ BAD: company_number will be dropped
    cleaning={
        "company_name": f"lower({companies_house.f('company_name')})",
        # company_number not included - will be DROPPED!
    }
    model_settings={
        "unique_fields": [
            "company_name",
            "company_number",  # This field won't exist!
        ],
    }
    
    # ✅ GOOD: Include all fields you need
    cleaning = {
        "company_name": f"lower({companies_house.f('company_name')})",
        "company_number": companies_house.f("company_number"),  # Pass through
    }

    model_settings = {
        "unique_fields": [
            "company_name",
            "company_number",  # Both fields available
        ],
    }
    ```
    
    This approach makes your configuration explicit and ensures all necessary fields are available to your models.

It's worth understanding how data moves through steps, as it helps knowing when or if to qualify column names. When would I use `"company_number"` vs. `companies_house.f("company_number")`, for example?

### Columns before and after cleaning

When you query a source without cleaning, all columns are qualified with the source name:

```python
# No cleaning - all columns qualified
companies_house.query().data()
# Columns: id, companies_house_key, companies_house_company_name, companies_house_company_number, companies_house_postcode
```

When you apply cleaning, the columns become the aliases you specify in the cleaning dictionary:

```python
# With cleaning - only specified columns, with your aliases
companies_house.query(
    cleaning={
        "name": f"lower({companies_house.f('company_name')})",
        "number": companies_house.f("company_number"),
    }
).data()
# Columns: id, companies_house_key, name, number
# Note: postcode is DROPPED because it's not in cleaning dict
```

Special columns are always passed through:

- `id`: The Matchbox entity ID (always present)
- `leaf_id`: If present, identifies the source cluster (optional)
- Key columns: One per source (e.g., `companies_house_key`, `exporters_key`)

### Qualified field references

Use `source.f()` to create qualified references in SQL expressions:

```python
# Inside cleaning dictionary values (SQL expressions)
companies_house.f("company_name")  # → "companies_house_company_name"

# In model_settings, use the cleaned aliases
model_settings = {
    "unique_fields": ["company_name"]  # Use the alias from cleaning dict
}
```

## 4. Creating linkers

Link steps match records across different sources or models.

=== "Example"
    ```python
    from matchbox.client.models.linkers import DeterministicLinker

    link_ch_exporters = dedupe_companies_house.query(
        companies_house,
        cleaning={
            "company_name": f"lower({companies_house.f('company_name')})",
            "postcode": companies_house.f("postcode"),
        },
    ).linker(
        dedupe_exporters.query(
            exporters,
            cleaning={
                "company_name": f"lower({exporters.f('company_name')})",
                "postcode": exporters.f("postcode"),
            },
        ),
        name="deterministic_ch_exporters",
        description="Link Companies House to exporters",
        model_class=DeterministicLinker,
        model_settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": [
                "l.company_name = r.company_name and l.postcode = r.postcode"
            ],
        },
    )
    ```

A linker requires:

- A second query which represents the data to link on the right side
- A unique `name` for the step
- An optional `description` explaining the purpose of the step
- The linking algorithm to use (`model_class`)
- Configuration (`model_settings`) for the algorithm
- An optional `truth` threshold (a float between `0.0` and `1.0`) above which a match is considered "true", the default being `1.0`.

As with deduplication, the `cleaning` dictionary maps field aliases to DuckDB SQL expressions that can reference input columns. See [On cleaning](#on-cleaning) for how to specify this functionality.

### Available linker types

Matchbox provides several linking methodologies:

1. [`DeterministicLinker`][matchbox.client.models.linkers.DeterministicLinker]: Links records based on exact matches of specified fields

    ```python
    from matchbox.client.models.linkers import DeterministicLinker
    
    model_settings = {
        "left_id": "id",
        "right_id": "id",
        "comparisons": "l.name = r.name and l.postcode = r.postcode"
    }
    ```

2. [`WeightedDeterministicLinker`][matchbox.client.models.linkers.WeightedDeterministicLinker]: Assigns different weights to different comparison fields

    ```python
    from matchbox.client.models.linkers import WeightedDeterministicLinker
    
    model_settings = {
        "left_id": "id",
        "right_id": "id",
        "weighted_comparisons": [
            {"comparison": "l.company_name = r.company_name", "weight": 0.7},
            {"comparison": "l.postcode = r.postcode", "weight": 0.3}
        ],
        "threshold": 0.8
    }
    ```

3. [`SplinkLinker`][matchbox.client.models.linkers.SplinkLinker]: Uses probabilistic matching with the [Splink](https://moj-analytical-services.github.io/splink/index.html) library

    ```python
    from matchbox.client.models.linkers import SplinkLinker
    from splink import SettingsCreator
    import splink.comparison_library as cl
    
    splink_settings = SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=["l.postcode = r.postcode"],
        comparisons=[
            cl.jaro_winkler_at_thresholds("company_name", [0.9, 0.6], term_frequency_adjustments=True)
        ]
    )
    
    model_settings = {
        "left_id": "id",
        "right_id": "id",
        "linker_settings": splink_settings,
        "linker_training_functions": [
            {
                "function": "estimate_probability_two_random_records_match",
                "arguments": {
                    "deterministic_matching_rules": "l.company_number = r.company_number",
                    "recall": 0.7
                }
            }
        ],
        "threshold": 0.8
    }
    ```

## 5. Running and publishing the DAG

Once you've defined all your steps, you can run and store the results of your [`DAG`][matchbox.client.dags.DAG].

=== "Example"
    ```python    
    # Run the entire DAG
    dag.run_and_sync()
    ```

Once you're happy with your results, you need to publish your DAG so that other users can query from it.

=== "Example"
    ```python    
    dag.set_default()
    ```



### Visualising DAG execution

When you run a DAG, Matchbox provides real-time status information:

=== "Output"
    ```
    ⏸️ deterministic_ch_hmrc
    └── ⏸️ naive_companieshouse_companies
    │   └── ⏸️ companieshouse.companies
    └── ⏸️ deterministic_exp_imp
        └── ⏸️ naive_hmrc_exporters
        │   └── ⏸️ hmrc.trade__exporters
        └── ⏸️ naive_hmrc_importers
            └── ⏸️ hmrc.trade__importers
    
    ...
    
    ⏸️ deterministic_ch_hmrc
    └── ⏸️ naive_companieshouse_companies
    │   └── ⏸️ companieshouse.companies
    └── 🔄 deterministic_exp_imp
        └── ✅ naive_hmrc_exporters
        │   └── ✅ hmrc.trade__exporters
        └── ⏭️ naive_hmrc_importers
            └── ⏭️ hmrc.trade__importers
    
    ...
    
    ✅ deterministic_ch_hmrc
    └── ✅ naive_companieshouse_companies
    │   └── ✅ companieshouse.companies
    └── ✅ deterministic_exp_imp
        └── ✅ naive_hmrc_exporters
        │   └── ✅ hmrc.trade__exporters
        └── ✅ naive_hmrc_importers
            └── ✅ hmrc.trade__importers
    ```

Status indicators:

- ⏸️ Awaiting execution
- 🔄 Currently executing
- ✅ Completed
- ⏭️ Skipped

## Advanced use cases

### Multi-source linking

You can link across multiple sources in a single step:

=== "Example"
    ```python
    # Link Companies House data with both exporters and importers
    link_ch_traders = dedupe_companies_house.query(
        companies_house,
        cleaning={
            "company_name": f"lower({companies_house.f('company_name')})",
            "postcode": companies_house.f("postcode"),
        },
    ).linker(
        link_exp_imp.query(
            importers,
            exporters,
            cleaning={
                "company_name": f"""
                    coalesce(
                        lower({exporters.f('company_name')}), 
                        lower({importers.f('company_name')})
                    )
                """,
                "postcode": f"""
                    coalesce(
                        {exporters.f('postcode')}, 
                        {importers.f('postcode')}
                    )
                """,
            },
        ),
        name="deterministic_ch_hmrc",
        description="Link Companies House to HMRC traders",
        model_class=DeterministicLinker,
        model_settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": [
                """
                    l.company_name = r.company_name
                        and l.postcode = r.postcode
                """
            ],
        },
    )
    ```

This example demonstrates how you can:

1. Use the results of a previous linking step as input
2. Select fields from multiple sources in a single query
3. Use SQL functions like `coalesce()` in your cleaning expressions to handle data from multiple sources
4. Create unified field names for comparison across sources

### Re-run a previous DAG

You might want to publish a new run of your DAG based on newer data. You can retrieve the old DAG and inspect it. You can't sync or publish it, as it will be read-only. However, you can generate a new run from it explicitly
```python    
# Create a new DAG identical to the previous default
dag = DAG(name="companies").load_default().set_client(engine).new_run()
# Run new DAG
dag.run_and_sync()
# Make the DAG the new default
dag.set_default()
```

## Best practices

### 1. Data preparation

Data cleaning is 90% of any record matching problem.

- Clean your data before matching
- Create appropriate indexes on your database tables
- Test your cleaning functions on sample data
- **Always include all fields needed by your models** in the cleaning dictionary

### 2. Pipeline design

- Break complex matching tasks into smaller steps
- Use appropriate batch sizes for large sources
- Create clear, descriptive names for your steps
- Be explicit about which fields you need at each step

### 3. Cleaning dictionaries

- **Include all fields** your model needs, even if just passing them through
- Use consistent aliases across left and right queries when linking
- Test your cleaning logic on sample data before running the full pipeline
- Remember that key columns are automatically passed through

### 4. Execution

- Start with small samples to test your pipeline
- Monitor performance and adjust batch sizes accordingly
- Use the `draw()` method to visualise and debug your DAG
- Pass `low_memory=True` to `dag.run_and_sync()` to reduce memory usage for a scheduled pipeline

## Conclusion

The Matchbox DAG API provides a powerful framework for building sophisticated data matching pipelines. By combining different types of steps (index, dedupe, link) with appropriate cleaning operations and matching algorithms, you can solve complex entity resolution problems efficiently.

For more information, explore the API reference for specific components:

- [DAG API](../api/client/dags.md)
- [Models](../api/client/models.md)
- [Results](../api/client/results.md)
