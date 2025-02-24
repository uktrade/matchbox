# Overview

::: matchbox.common.factories
    options:
        show_root_heading: true
        show_root_full_path: true
        show_root_docstring: true
        show_root_docstring: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"  # Excludes single-letter uppercase variables (like T, P, R)
            - "!^_"       # Excludes private attributes
            - "!_logger$"  # Excludes logger variables
            - "!_path$"    # Excludes path variables
            - "!model_config" # Excludes Pydantic configuration

## Using the system

The factory system aims to provide objects that facilitate three groups of testing scenarios:

* Realistic mock `Source` and `Model` objects to test client-side connectivity functions
* Realistic mock data to test server-side adapter functions
* Realistic mock pipeliness with controlled completeness to test client-side methodologies

Three broad functions are provided:

* [`source_factory()`][matchbox.common.factories.sources.source_factory] generates [`SourceDummy`][matchbox.common.factories.sources.SourceDummy] objects, which contain dummy `Source`s and associated data, such as the true entities this data describes
* [`linked_sources_factory()`][matchbox.common.factories.sources.linked_sources_factory] generates [`LinkedSourcesDummy`][matchbox.common.factories.sources.LinkedSourcesDummy] objects, which contain a collection of interconnected `SourceDummy` objects, and the true entities this data describes
* [`model_factory()`][matchbox.common.factories.models.model_factory] generates [`ModelDummy`][matchbox.common.factories.models.ModelDummy] objects, which mock probabilities that can connect both `SourceDummy` and other `ModelDummy` objects in ways that fail and succeed predictably

Underneath, these factories and objects use a system of [`SourceEntity`][matchbox.common.factories.entities.SourceEntity] and [`ResultsEntity`][matchbox.common.factories.entities.ResultsEntity]s to share data. The source is the true answer, and the results are the merging data as it moves through the system. A comprehensive set of comparitors have been implements to make this simple to implement, understand, and read in unit testing.

All factory methods aim to be useful in three ways:

* When used without arguments, produce a sensible, useful default
* A set of arguments that allow more generalised generation
* A set of arguments that allow specific control, such as using a `SourceDummy` as an argument to `model_factory()`

The system has been designed to be as hashable as possible to enable caching. Often you'll need to provide tuples where you might normally provide lists.

There are some common patterns you might consider using when editing or extending tests.

## Client-side connectivity

We can use the factories to test inserting or retrieving isolated `Source` or `Model` objects.

Perhaps you're testing the API and want to put a realistic `Source` in the ingestion pipeline.

```python
dummy_source = source_factory()

# Setup store
store = MetadataStore()
update_id = store.cache_source(dummy_source.source)
```

Or you're testing the client handler and want to mock the API.

```python
@patch("matchbox.client.helpers.index.Source")
def test_my_api(MockSource: Mock, matchbox_api: MockRouter):
    source = source_factory(
        features=[{"name": "company_name", "base_generator": "company"}]
    )
    mock_source_instance = source.to_mock()
    MockSource.return_value = mock_source_instance
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

Adding a `Source`.

```python
dummy_source = source_factory()
backend.index(
    source=dummy_source.source
    data_hashes=dummy_source.data_hashes
)
```

Adding a `Model`.

```python
dummy_model = model_factory()
backend.insert_model(model=dummy_model.model.metadata)
```

Inserting results.

```python
dummy_model = model_factory()
backend.set_model_results(
    model=dummy_model.model.metadata.full_name, 
    results=dummy_model.probabilities
)
```

`linked_sources_factory()` and `model_factory()` can be used together to create broader systems of data that connect -- or don't -- in controlled ways.

```python
linked = linked_sources_factory()

for dummy_source in linked.sources.values():
    backend.index(
        source=dummy_source.source
        data_hashes=dummy_source.data_hashes
    )

dummy_model = model_factory(
    left_source=linked.sources["crn"],
    true_entities=all_true_sources,
)

backend.insert_model(model=dummy_model.model.metadata)
backend.set_model_results(
    model=dummy_model.model.metadata.full_name, 
    results=dummy_model.probabilities
)
```

## Methodologies

Configure the true state of your data with `linked_sources_factory()`. Its default is a set of three tables of ten unique company entites.

* CRN (company name, CRN ID) contains all entities with three unique variations of the company's name
* CDMS (CRN ID, DUNS ID) contains all entities repeated twice
* DUNS (company name, DUNS ID) contains half the entities

Give a tuple of [`SourceConfig`][matchbox.common.factories.sources.SourceConfig] objects to set up any scenario you desire.

The `model_factory()` is designed so you can chain together known processes in any order, before using your real methodology. [`LinkedSourcesDummy.diff_results()`][matchbox.common.factories.sources.LinkedSourcesDummy.diff_results] will make any probabilistic output comparable with the true source entities, and give a detailed diff to help you debug.

```python
linked = linked_sources_factory()

# Create perfect deduped models first
left_deduped = model_factory(
    left_source=linked.sources["crn"],
    true_entities=linked.true_entities.values(),
)
right_deduped = model_factory(
    left_source=linked.sources["cdms"],
    true_entities=linked.true_entities.values(),
)

# Create a model and generate probabilities
model = make_model(...)
results = model.run()

# Diff, assert, and log the message if it fails
identical, msg = linked.diff_results(
    probabilities=results.probabilities,
    left_results=left_deduped.entities,
    right_results=right_deduped.entities
    sources=("crn", "cdms"),
    threshold=0,
    verbose=True,
)

assert identical, msg
```