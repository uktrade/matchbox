<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/uktrade/matchbox/refs/heads/main/docs/assets/matchbox-logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/uktrade/matchbox/refs/heads/main/docs/assets/matchbox-logo-light.svg">
      <img alt="Shows the Matchbox logo in light or dark color mode." src="https://raw.githubusercontent.com/uktrade/matchbox/refs/heads/main/docs/assets/matchbox-logo-light.svg">
    </picture>
</p>

Record matching is a chore. 🔥Matchbox is a match pipeline orchestration tool that aims to:

* Make matching an iterative, collaborative, measurable problem
* Compose sources, dedupers and linkers and make the results very easy to query
* Allow organisations to know they have matching records without having to share the data
* Allow matching pipelines to run iteratively
* Support batch and real-time matching 

Matchbox doesn't store raw data, instead indexing the data in your warehouse and leaving permissioning at the level of the user, service or pipeline.

To get started, read our [full documentation](https://uktrade.github.io/matchbox/).

## Installation

To install the matchbox client:

```shell
pip install "matchbox-db"
```

To install the full package, including the server features:

```shell
pip install "matchbox-db[server]"
```

To run the server, see the [server installation documentation](https://uktrade.github.io/matchbox/server/install/).

## Use cases

### Data architects and engineers

* Reconcile entities across disparate data
* Rationalise about the quality of different entity matching pipelines and serve up the best
* Run matching pipelines without recomputing them every time
* Lay the foundation for the nouns of a semantic layer

### Data analysts and scientists

* Use your team's best matching methods when retrieving entities, always
* Measurably improve methodologies when they don't work for you
* When you link new data, allow others to use your work easily and securely

### Service owners

* Understand the broader business entities in your service, not just what you have
* Enrich other services with data generated in yours without giving away any permissioning powers
* Empower your users to label matched entities and let other services use that information

## Development

See our full development guide and coding standards on our [contribution guide](https://uktrade.github.io/matchbox/contributing/).

## Local development with Datadog

When iterating the Datadog configuration, environment variables can be set in several ways:

1. **Datadog configuration**: Create a `.datadog.env` file with your Datadog API key and other agent settings
2. **Compose override**: Use `docker-compose.override.yml` for local-specific variable overrides

Variables in `.datadog.env` will override any defaults set in the compose file.

Example `.datadog.env`:

```
DD_API_KEY=your_api_key_here
```

The Docker Compose file will automatically set `DD_ENV=local-{username}` for local development isolation.
