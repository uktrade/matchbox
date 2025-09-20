# Building with Matchbox DAGs

Data matching and entity resolution are complex tasks that often require multiple processing steps. Matchbox provides a powerful Directed Acyclic Graph (DAG) framework that allows you to define and run sophisticated matching pipelines with clearly defined dependencies.

This guide walks through creating complete matching pipelines using the Matchbox DAG API, covering everything from [defining data sources](#2-defining-data-sources) to [executing complex multi-step matching processes](#advanced-use-cases). In our examples we'll be referencing publicly available datasets about UK companies, specifically [Companies House data](https://find-and-update.company-information.service.gov.uk), and [UK trade data](https://www.uktradeinfo.com).

## Understanding DAGs in Matchbox

A DAG (Directed Acyclic Graph) represents a sequence of operations where each step depends on the outputs of previous steps, without any circular dependencies. In Matchbox, a DAG consists of:

* [`Source`s][matchbox.client.sources.Source]: indexing data from sources
* [`Model`s][matchbox.client.models.Model]: Removing duplicates within one data input, or linking two data inputs. As data inputs, `Model`s can take `Source`s or other `Model`s.

`Source`s and `Model`s can form [`Query`][matchbox.client.models.Model] objects, which allow to retrieve the version of the data implied by that DAG step. Querying a source gives you the records in that source, and querying from a model gives you the deduplicated or linked records at that point in the DAG. When querying from a model, you need to specify which sources you want to query from that model's lineage.

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
queried_data = query.run()
deduper_results = deduper.run()
linker_results = linker.run()
```

The steps need to be run in order, but once you've finalised your DAG, it's better to automatically run all of them using a single DAG command, as is shown later. When you run a step, either directly or through the DAG, its data is cached so that running it again won't do anything, unless you force a re-run.


## Setting up your environment

Before building a pipeline, it's worth configuring logging:

=== "Example"
    ```python
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger()

    # Reduce noise from HTTP libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("http").setLevel(logging.WARNING)
    ```

You will also need to define the engine to read your data sources:

=== "Example"
    ```python
    # Get your database engine
    from your_utils import get_database_engine
    engine = get_database_engine()
    ```

## 1. Defining a new DAG

You're now ready to create your first [`DAG`][matchbox.client.dags.DAG].

=== "Example"
    ```python
    from matchbox.client.dags import DAG
    dag = DAG(name="companies", new=True)
    ```

A DAG needs a name, which will be used to identify this DAG once you publish it to the Matchbox server. You also need to specify that this is a new, unpublished DAG. If you don't do that, Matchbox will look for a DAG with this name that was already published.

This DAG will own all the sources and models you define later.

## 2. Defining data sources

Now you can define your data sources. Each source represents data that will be used in the matching process.

The `index_fields` are what Matchbox will use to store a reference to your data, and are the only fields it will permit you to match on.

The `key_field` is the field in your source that contains some unique code that identifies each entitiy. For example, in a relational database, this would typically be your primary key.

=== "Example"
    ```python
    from matchbox.client import RelationalDBLocation

    # Companies House data
    companies_house = dag.source(
        name="companies_house",
        location=RelationalDBLocation(name="dbname", client=engine),
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
        location=RelationalDBLocation(name="dbname", client=engine),
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

- A `location`, such as [`RelationalDBLocation`][matchbox.client.sources.RelationalDBLocation]. This will need a `name`, and `client`
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
        }
    ).deduper(
        name="naive_companieshouse_companies",
        description="Deduplication based on company name",
        model_class=NaiveDeduper,
        model_settings={
            "unique_fields": ["company_name"],
        },
        truth=1.0,
    )

A query can optionally take instructions on how to clean the data. These are defined using a dictionary where:

* the dictionary **key** is the desired column name that will be output
* the dictionary **value** is a SQL expression in DuckDB format
    
A deduper takes:

- A unique `name` for the step
- An optional `description` explaining the purpose of the step
- The deduplication algorithm to use (`model_class`)
- Configuration settings (`model_settings`) for the algorithm
- Optionally, a `truth` threshold (a float between `0.0` and `1.0`) above which a match is considered "true". By default, this is set to `1.0`. This value is only relevant when using a model that can output matches with different confidence scores, which is not the case for a `NaiveDeduper`.

### On cleaning

!!! tip "Simplify field references by cleaning everything"
    
    To avoid confusion with qualified vs unqualified field names, consider "cleaning" every field you select - even if you're just aliasing it without transformation. This way, all your field references use simple, unqualified names throughout your configuration.
    
    ```python
    # Instead of mixing qualified and unqualified names
    cleaning={
        "company_name": f"lower({companies_house.f('company_name')})",
        # company_number not cleaned, so needs qualification later
    }
    model_settings={
        "unique_fields": [
            "company_name",
            companies_house.f("company_number"),  # Qualified!
        ],
    }
    
    # Clean everything for consistency
    cleaning = {
        "company_name": f"lower({companies_house.f('company_name')})",
        "company_number": companies_house.f("company_number") # Just aliasing
    }

    model_settings = {
        "unique_fields": [
            "company_name",
            "company_number",  # Both unqualified!
        ],
    }
    ```
    
    This approach makes your configuration much more readable and reduces errors from forgetting to qualify field names.

It's worth understanding how data moves through steps, as it helps knowing when or if to qualify column names. When would I use `"company_number"` vs. `companies_house.f("company_number")`, for example?

A [`Query`][matchbox.client.queries.Query] extracts data to a columnar format. Models will often query the same column names from multiple sources, so column names must be qualified with their source. 

For example, 

```python
companies_house.query().run()
```

will return a dataframe with the following columns:

* `id` (the Matchbox ID)
* `companies_house_company_number`
* `companies_house_company_name`
* `companies_house_postcode`

Note how the fields specified are "qualified" with the source they came from. When defining cleaning instructions, we need to refer to qualified source names too. `Source.f()` is provided as a convenient way to select fields qualified by a source.

The rules for the cleaning dictionary are:

* If a column _is_ mentioned in any cleaning SQL, its uncleaned version is automatically dropped from the output
* If a column _isn't_ mentioned in any cleaning SQL, it's automatically passed through with its qualified name

Here's the cleaning dictionary from the above example:

```python
cleaning_dict = {
    "company_name": f"lower({companies_house.f('company_name')})",
}
```

Note: 

* How we qualify the field we clean
* How we alias it to `company_name`
* `company_number` and `postcode` _aren't_ mentioned.

The columns output by the query will be:

* `id` (the Matchbox ID)
* `companies_house_company_number`
* `company_name` (unqualified)
* `companies_house_postcode`

Finally, cleaned fields typically need specifying in a model. Here's our example:

```python
model_settings = {
    "id": "id",
    "unique_fields": [
        "company_name",
        companies_house.f("company_number"),
    ],
}
```

Note that because we didn't clean `company_number` it needs to be qualified here, rather than in the cleaning dictionary.

## 4. Creating link steps

Link steps connect records between different sources.

=== "Example"
    ```python
    from matchbox.client.dags import LinkStep
    from matchbox.client.models.linkers import DeterministicLinker

    # Link exporters and importers based on name and postcode
    link_exp_imp = dedupe_exporters.query(
        exporters,
        cleaning={
                "company_name": f"lower({exporters.f('company_name')})",
                "postcode": exporters.f("postcode"),
            },
    ).linker(
        dedupe_importers.query(
            importers,
            cleaning={
                "company_name": f"lower({importers.f('company_name')})",
                "postcode": importers.f("postcode"),
            },
        )
        name="deterministic_exp_imp",
        description="Deterministic link on names and postcode",
        model_class=DeterministicLinker,
        settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": [
                """
                    l.company_name = r.company_name
                        and l.postcode= r.postcode
                """
            ],
        },
    )
    ```

A linker requires:

- A second query to link (against the first query, which generated the linker)
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
    # Optionally, you can force the re-run of all steps if some of them had already run
    dag.run_and_sync(full_rerun=True)
    ```

Once you're happy with your results, you need to publish your DAG so that other users can query from it.

=== "Example"
    ```python    
    dag.publish()
    ```



### Visualising DAG Execution

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
            exporters
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
        settings={
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
2. Select fields from multiple sources in a single step
3. Use SQL functions like `coalesce()` in your cleaning expressions to handle data from multiple sources
4. Create unified field names for comparison across sources

### Re-run a previous DAG

You might want to publish a new version of your DAG based on newer data. You can retrieve the old DAG and inspect it. You can't sync or publish it, as it will be read-only. However, you can generate a new version from it explicitly

=== "Example"
    ```python    
    # Retrieve published DAG
    old_dag = DAG(name="companies")
    # Generate a new DAG from it
    new_dag = old_dag.clone()
    new_dag.run_and_sync()
    new_dag.publish()
    ```

## Best practices

### 1. Data preparation

Data cleaning is 90% of any record matching problem.

- Clean your data before matching
- Create appropriate indexes on your database tables
- Test your cleaning functions on sample data

### 2. Pipeline design

- Break complex matching tasks into smaller steps
- Use appropriate batch sizes for large sources
- Create clear, descriptive names for your steps

### 3. Execution

- Start with small samples to test your pipeline
- Monitor performance and adjust batch sizes accordingly
- Use the `draw()` method to visualize and debug your DAG

## Conclusion

The Matchbox DAG API provides a powerful framework for building sophisticated data matching pipelines. By combining different types of steps (index, dedupe, link) with appropriate cleaning operations and matching algorithms, you can solve complex entity resolution problems efficiently.

For more information, explore the API reference for specific components:

- [DAG API](../api/client/dags.md)
- [Models](../api/client/models.md)
- [Results](../api/client/results.md)
