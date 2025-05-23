# Overview

::: matchbox.common.factories
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        extra:
            show_root_docstring: true
        filters:
            - "!^[A-Z]$"  # Excludes single-letter uppercase variables (like T, P, R)
            - "!^_"       # Excludes private attributes
            - "!_logger$"  # Excludes logger variables
            - "!_path$"    # Excludes path variables
            - "!model_config" # Excludes Pydantic configuration

## Using the system

The factory system aims to provide `*Testkit` objects that facilitate three groups of testing scenarios:

* Realistic mock `SourceConfig` and `Model` objects to test client-side connectivity functions
* Realistic mock data to test server-side adapter functions
* Realistic mock pipelines with controlled completeness to test client-side methodologies

Three broad functions are provided:

* [`source_factory()`][matchbox.common.factories.sources.source_factory] generates [`SourceTestkit`][matchbox.common.factories.sources.SourceTestkit] objects, which contain dummy `SourceConfig`s and associated data
* [`linked_sources_factory()`][matchbox.common.factories.sources.linked_sources_factory] generates [`LinkedSourcesTestkit`][matchbox.common.factories.sources.LinkedSourcesTestkit] objects, which contain a collection of interconnected `SourceTestkit` objects, and the true entities this data describes
* [`model_factory()`][matchbox.common.factories.models.model_factory] generates [`ModelTestkit`][matchbox.common.factories.models.ModelTestkit] objects, which mock probabilities that can connect both `SourceTestkit` and other `ModelTestkit` objects in ways that fail and succeed predictably

Underneath, these factories and objects use a system of [`SourceEntity`][matchbox.common.factories.entities.SourceEntity] and [`ClusterEntity`][matchbox.common.factories.entities.ClusterEntity]s to share data. The source is the true answer, and the clusters are the merging data as it moves through the system. A comprehensive set of comparators have been implemented to make this simple to implement, understand, and read in unit testing.

All factory functions are configured to provide a sensible, useful default.

The system has been designed to be as hashable as possible to enable caching. Often you'll need to provide tuples where you might normally provide lists.

There are some common patterns you might consider using when editing or extending tests.

## Client-side connectivity

We can use the factories to test inserting or retrieving isolated `SourceConfig` or `Model` objects.

Perhaps you're testing the API and want to put a realistic `SourceConfig` in the ingestion pipeline.

```python
source_testkit = source_factory()

# Setup store
store = MetadataStore()
update_id = store.cache_source(source_testkit.source_config)
```

Or you're testing the client handler and want to mock the API.

```python
@patch("matchbox.client.helpers.index.SourceConfig")
def test_my_api(MockSource: Mock, matchbox_api: MockRouter):
    source_testkit = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}]
    )
    MockSource.return_value = source_testkit.mock
```

`source_factory()` can be configured with a powerful range of [`FeatureConfig`][matchbox.common.factories.entities.FeatureConfig] objects, including a [variety of rules][matchbox.common.factories.entities.VariationRule] which distort and duplicate the data in predictable ways. These use [Faker](https://faker.readthedocs.io/) to generate data.

```python
source_factory(
    n_true_entities=1_000,
    features=(
        FeatureConfig(
            name="name",
            base_generator="first_name_female",
            drop_base=False,
            variations=(PrefixRule(prefix="Ms "),),
        ),
        FeatureConfig(
            name="title",
            base_generator="job",
            drop_base=True,
            variations=(
                SuffixRule(suffix=" MBE"),
                ReplaceRule(old="Manager", new="Leader"),
            ),
    ),
    repetition=3,
)
```

## Server-side adapters

The factories can generate data suitable for `MatchboxDBAdapter.index()`, `MatchboxDBAdapter.insert_model()`, or `MatchboxDBAdapter.set_model_results()`. Between these functions, we can set up any backend in any configuration we need to test the other adapter methods.

Adding a `SourceConfig`.

```python
source_testkit = source_factory()
backend.index(
    source_config=source_testkit.source_config
    data_hashes=source_testkit.data_hashes
)
```

Adding a `Model`.

```python
model_testkit = model_factory()
backend.insert_model(model_config=model_testkit.model.model_config)
```

Inserting results.

```python
model_testkit = model_factory()
backend.set_model_results(
    name=model_testkit.model.model_config.name, 
    results=model_testkit.probabilities
)
```

`linked_sources_factory()` and `model_factory()` can be used together to create broader systems of data that connect -- or don't -- in controlled ways.

```python
linked_testkit = linked_sources_factory()

for source_testkit in linked_testkit.sources.values():
    backend.index(
        source_config=source_testkit.source_config
        data_hashes=source_testkit.data_hashes
    )

model_testkit = model_factory(
    left_testkit=linked_testkit.sources["crn"],
    true_entities=linked_testkit.true_entities,
)

backend.insert_model(model_config=model_testkit.model.model_config)
backend.set_model_results(
    name=model_testkit.model.model_config.name, 
    results=model_testkit.probabilities
)
```

## Methodologies

Configure the true state of your data with `linked_sources_factory()`. Its default is a set of three tables of ten unique company entites.

* CRN (company name, CRN ID) contains all entities with three unique variations of the company's name
* CDMS (CRN ID, DUNS ID) contains all entities repeated twice
* DUNS (company name, DUNS ID) contains half the entities

`linked_sources_factory()` can be configured using tuples of [`SourceTestkitParameters`][matchbox.common.factories.sources.SourceTestkitParameters] objects. Using these you can create complex sets of interweaving sources for methodologies to be tested against.

The `model_factory()` is designed so you can chain together known processes in any order, before using your real methodology. [`LinkedSourcesTestkit.diff_results()`][matchbox.common.factories.sources.LinkedSourcesTestkit.diff_results] will make any probabilistic output comparable with the true source entities, and give a detailed diff to help you debug.

```python
linked_testkit: LinkedSourcesTestkit = linked_sources_factory()

# Create perfect deduped models first
left_deduped: ModelTestkit = model_factory(
    left_testkit=linked_testkit.sources["crn"],
    true_entities=linked_testkit.true_entities,
)
right_deduped: ModelTestkit = model_factory(
    left_testkit=linked_testkit.sources["cdms"],
    true_entities=linked_testkit.true_entities,
)

# Create a model and generate probabilities
model: Model = make_model(
    left_data=left_deduped.query,
    right_data=right_deduped.query
    ...
)
results: Results = model.run()

# Diff, assert, and log the message if it fails
identical, report = linked_testkit.diff_results(
    probabilities=results.probabilities,  # Your methodology's output
    left_clusters=left_deduped.entities,  # Output of left deduper -- left input to your methodology
    right_clusters=right_deduped.entities,  # Output of right deduper -- left input to your methodology
    sources=("crn", "cdms"),
    threshold=0,
)

assert identical, report
```