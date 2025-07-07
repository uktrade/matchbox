Matchbox is a tool to make record matching iterative, collaborative and measurable.

The server allows any link or deduplication job on any warehouse to be stored in a common format. It holds user validation of clusters, and measures the quality of competing processes. This allows the organisation to designate the match pipelines that most accurately reflect the "nouns" of their data: the "people", "companies" or "products" who exist across a dozen datasets.

## Running the server locally

To build and run the server locally:

```shell
just build
```

### With Datadog integration

[Datadog](https://www.datadoghq.com) is an monitoring tool integrated with Matchbox.

Create a Datadog environment variable file.

```shell
cp ./environments/datadog-agent-private-sample.env ./environments/.datadog-agent-private.env
```

Populate the newly-created ` ./environments/.datadog-agent-private.env` with a Datadog API key.

Run the server using:

```shell
docker compose --profile monitoring up --build
```
