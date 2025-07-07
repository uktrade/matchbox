# Building with Matchbox DAGs

Data matching and entity resolution are complex tasks that often require multiple processing steps. Matchbox provides a powerful Directed Acyclic Graph (DAG) framework that allows you to define and run sophisticated matching pipelines with clearly defined dependencies.

!!! tip "DAG steps and resolutions"
    Once it is run, a step in a DAG corresponds to a resolution in the Matchbox database. Learn more by reading about the tutorial on [exploring resolutions](../client/explore_resolutions.md) or dive deeper by consulting the extended guide to [Matchbox concepts](../server/concepts.md). 


This guide walks through creating complete matching pipelines using the Matchbox DAG API, covering everything from [defining data sources](#1-defining-data-sources) to [executing complex multi-step matching processes](#advanced-use-cases). In our examples we'll be referencing publicly available datasets about UK companies, specifically [Companies House data](https://find-and-update.company-information.service.gov.uk), and [UK trade data](https://www.uktradeinfo.com).

## Understanding DAGs in Matchbox

A DAG (Directed Acyclic Graph) represents a sequence of operations where each step depends on the outputs of previous steps, without any circular dependencies. In Matchbox, a DAG consists of:

1. [`IndexStep`s][matchbox.client.dags.IndexStep]: Loading and indexing data from sources
2. [`DedupeStep`s][matchbox.client.dags.DedupeStep]: Removing duplicates within a source
3. [`LinkStep`s][matchbox.client.dags.LinkStep]: Connecting records between different sources

## Setting up your environment

Before building a pipeline, ensure you have Matchbox properly installed and configured:

```python
import logging
from matchbox.client import clean
from matchbox.client.dags import DAG, DedupeStep, IndexStep, LinkStep, StepInput
from matchbox.client.helpers.cleaner import cleaner, cleaners
from matchbox.client.models.dedupers.naive import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.common.sources import SourceConfig, RelationalDBLocation

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

# Get your database engine
from your_utils import get_database_engine
engine = get_database_engine()
```

## 1. Defining data sources

The first step in creating a matching pipeline is to define your data sources. Each source represents data that will be used in the matching process.

The `index_fields` are what Matchbox will use to store a reference to your data, and are the only fields it will permit you to match on.

The `key_field` is the field in your source that contains some unique code that identifies each entitiy. For example, in a relational database, this would typically be your primary key.

=== "Example"
    ```python
    from matchbox.common.sources import SourceConfig, RelationalDBLocation
    
    # Companies House data
    companies_house = SourceConfig.new(
        location=RelationalDBLocation.from_engine(engine),
        extract_transform="""
            select
                pk as id,
                company_name,
                number::str as company_number,
                upper(postcode) as postcode,
            from
                companieshouse.companies;
        """,
        index_fields=["company_name", "company_number", "postcode"],
        key_field="id",
    )
    
    # Exporters data
    exporters = SourceConfig.new(
        location=RelationalDBLocation.from_engine(engine),
        extract_transform="""
            select
                id,
                company_name,
                upper(postcode) as postcode,
            from
                hmrc.trade__exporters;
        """,
        index_fields=["company_name", "postcode"],
        key_field="id",
    )
    ```

Each [`SourceConfig`][matchbox.common.sources.SourceConfig] object requires:

- A `location`, such as [`RelationalDBLocation`][matchbox.common.sources.RelationalDBLocation]. This will need a `type`, a `uri`, and `credentials`, the type of which changes depending on the type of location you're using
    - For most users [`RelationalDBLocation`][matchbox.common.sources.RelationalDBLocation] and its `.from_engine()` constructor is all you need
    - For a relational database, a SQLAlchemy engine is your credentials
- An `extract_transform` string, which will take data from the location and transform it into your key and index fields. Its syntax will depend on the type of location
    - For most users, using a relational database location, this will be SQL
- A list of `index_fields` that will be used for matching
    - These must be found in the result of the `extract_transform` logic
- A key field (`key_field`) that uniquely identifies each record
    - This must be found in the result of the `extract_transform` logic

## 2. Defining data cleaners

Data cleaning is essential for effective matching. Matchbox provides a flexible, reusable way to define cleaning operations for your data.

=== "Example"
    ```python
    from matchbox.client import clean
    from matchbox.client.helpers.cleaner import cleaner, cleaners
    
    # Cleaner for Companies House data
    ch_clean = cleaners(
        cleaner(
            clean.company_name,
            {"column": "companieshouse_companies_company_name"},
        ),
        cleaner(
            clean.company_number,
            {"column": "companieshouse_companies_company_number"},
        ),
        cleaner(clean.postcode, {"column": "companieshouse_companies_postcode"}),
    )
    
    # Cleaner for Exporters data
    ex_clean = cleaners(
        cleaner(clean.company_name, {"column": "hmrc_trade__exporters_company_name"}),
        cleaner(clean.postcode, {"column": "hmrc_trade__exporters_postcode"}),
    )
    ```

The [`cleaners()`][matchbox.client.helpers.cleaner.cleaners] function combines multiple [`cleaner()`][matchbox.client.helpers.cleaner.cleaner] operations. Each `cleaner()` specifies:

- A cleaning function (e.g., `clean.company_name`)
- Configuration options (including which column to clean)

Matchbox includes standard cleaning functions for common fields like company names, addresses, and identifiers. You can also create custom cleaning functions.

See the [full cleaning API](../api/client/clean.md) for more information, including using [`matchbox.client.clean.steps`][matchbox.client.clean.steps] to compose your own cleaning function from atomic cleaning steps.

## 3. Creating index steps

Index steps load data from your sources into Matchbox. Matchbox never sees your data, storing only a reference to it.

Only data indexed in Matchbox can we used to match.

=== "Example"
    ```python
    from matchbox.client.dags import IndexStep
    
    # Define batch size
    batch_size = 250_000
    
    # Create index steps
    i_companies = IndexStep(source_config=companies_house, batch_size=batch_size)
    i_exporters = IndexStep(source_config=exporters, batch_size=batch_size)
    ```

Each [`IndexStep`][matchbox.client.dags.IndexStep] requires:

- A `source` object
- An optional `batch_size` for processing large data in chunks

## 4. Creating dedupe steps

Dedupe steps identify and resolve duplicates within a single source.

=== "Example"
    ```python
    from matchbox.client.dags import DedupeStep, StepInput
    from matchbox.client.models.dedupers.naive import NaiveDeduper
    
    # Deduplicate Companies House data based on company number
    dedupe_companies = DedupeStep(
        left=StepInput(
            prev_node=i_companies,
            select={companies_house: ["company_name", "company_number", "postcode"]},
            cleaners=ch_clean,
            batch_size=batch_size,
        ),
        name="naive_companieshouse_companies",
        description="Deduplication based on company number",
        model_class=NaiveDeduper,
        settings={
            "id": "id",
            "unique_fields": [
                "companieshouse_companies_company_number",
            ],
        },
        truth=1.0,
    )
    ```

A [`DedupeStep`][matchbox.client.dags.DedupeStep] requires:

- A `left` input, defined as a [`StepInput`][matchbox.client.dags.StepInput] that specifies:
    - The previous step (`prev_node`)
    - Which fields to select (`select`)
    - Cleaning operations to apply (`cleaners`)
    - Optional batch size
- A unique `name` for the step
- A `description` explaining the purpose of the step
- The deduplication algorithm to use (`model_class`)
- Configuration `settings` for the algorithm
- A `truth` threshold (a float between `0.0` and `1.0`) above which a match is considered "true"

## 5. Creating link steps

Link steps connect records between different sources.

=== "Example"
    ```python
    from matchbox.client.dags import LinkStep
    from matchbox.client.models.linkers import DeterministicLinker
    
    # Link exporters and importers based on name and postcode
    link_exp_imp = LinkStep(
        left=StepInput(
            prev_node=dedupe_exporters,
            select={exporters: ["company_name", "postcode"]},
            cleaners=ex_clean,
            batch_size=batch_size,
        ),
        right=StepInput(
            prev_node=dedupe_importers,
            select={importers: ["company_name", "postcode"]},
            cleaners=im_clean,
            batch_size=batch_size,
        ),
        name="deterministic_exp_imp",
        description="Deterministic link on names and postcode",
        model_class=DeterministicLinker,
        settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": """
                l.hmrc_trade__exporters_company_name
                    = r.hmrc_trade__importers_company_name
                and l.hmrc_trade__exporters_postcode
                    = r.hmrc_trade__importers_postcode
            """,
        },
        truth=1.0,
    )
    ```

A [`LinkStep`][matchbox.client.dags.LinkStep] requires:

- A `left` and `right` input, defined as a [`StepInput`][matchbox.client.dags.StepInput] that specifies:
    - The previous step (`prev_node`)
    - Which fields to select (`select`)
    - Cleaning operations to apply (`cleaners`)
    - Optional batch size
- A unique `name` for the step
- A `description` explaining the purpose of the step
- The linking algorithm to use (`model_class`)
- Configuration `settings` for the algorithm
- A `truth` threshold (a float between `0.0` and `1.0`) above which a match is considered "true"

### Available linker types

Matchbox provides several linking methodologies:

1. [`DeterministicLinker`][matchbox.client.models.linkers.DeterministicLinker]: Links records based on exact matches of specified fields

    ```python
    from matchbox.client.models.linkers import DeterministicLinker
    
    settings = {
        "left_id": "id",
        "right_id": "id",
        "comparisons": "l.name = r.name and l.postcode = r.postcode"
    }
    ```

2. [`WeightedDeterministicLinker`][matchbox.client.models.linkers.WeightedDeterministicLinker]: Assigns different weights to different comparison fields

    ```python
    from matchbox.client.models.linkers import WeightedDeterministicLinker
    
    settings = {
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
    
    settings = {
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

## 6. Building and running the DAG

Once you've defined all your steps, you can build and run the complete [`DAG`][matchbox.client.dags.DAG].

=== "Example"
    ```python
    from matchbox.client.dags import DAG
    
    # Create the DAG
    my_dag = DAG()
    
    # Add index steps
    my_dag.add_steps(i_companies, i_exporters, i_importers)
    
    # Add dedupe steps
    my_dag.add_steps(dedupe_companies, dedupe_exporters, dedupe_importers)
    
    # Add link steps
    my_dag.add_steps(link_exp_imp, link_companies_traders)
    
    # Visualise the DAG
    print(my_dag.draw())
    
    # Run the entire DAG
    my_dag.run()
    
    # Alternatively, run from a specific step
    # my_dag.run(start="dedupe_exporters")
    ```

The key methods for working with DAGs are:

- `.add_steps()`: Add one or more steps to the DAG
- `.draw()`: Visualise the DAG structure
- `.run()`: Execute the entire DAG or from a specific step

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
    link_ch_traders = LinkStep(
        left=StepInput(
            prev_node=dedupe_companies,
            select={companies_house: ["company_name", "postcode"]},
            cleaners=ch_clean_simple,
        ),
        right=StepInput(
            prev_node=link_exp_imp,  # Using a previous link step as input
            select={
                importers: ["company_name", "postcode"],
                exporters: ["company_name", "postcode"],
            },
            cleaners={**ex_clean, **im_clean},
        ),
        name="deterministic_ch_hmrc",
        description="Link Companies House to HMRC traders",
        model_class=DeterministicLinker,
        settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": """
                l.companieshouse_companies_company_name =
                    coalesce(
                        r.hmrc_trade__exporters_company_name,
                        r.hmrc_trade__importers_company_name
                    )
                and l.companieshouse_companies_postcode =
                    coalesce(
                        r.hmrc_trade__exporters_postcode,
                        r.hmrc_trade__importers_postcode
                    )
            """,
        },
        truth=1.0,
    )
    ```

This example demonstrates how you can:

1. Use the results of a previous linking step as input
2. Select fields from multiple sources in a single step
3. Use SQL functions like `coalesce()` in your comparison logic

### Conditional matching

You can implement complex matching logic using SQL expressions:

=== "Example"
    ```python
    link_companies = LinkStep(
        # ... other parameters ...
        settings={
            "left_id": "id",
            "right_id": "id",
            "comparisons": """
                (
                    l.company_number = r.company_number
                ) OR (
                    l.company_name = r.company_name
                    AND l.postcode = r.postcode
                )
                """,
        },
        truth=1.0,
    )
    ```

This example matches records that either:

1. Have the same company number, OR
2. Have the same company name AND postcode

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
- [Linkers](../api/client/models.md)
- [Cleaners](../api/client/clean.md)
- [Results](../api/client/results.md)
