Matchbox is a tool to make record matching iterative, collaborative and measurable.

The server allows any link or deduplication job on any warehouse to be stored in a common format. It holds user validation of clusters, and measures the quality of competing processes. This allows you to designate the match pipelines that most accurately reflect the "nouns" of your data: the "people", "companies" or "products" who exist across a dozen datasets.

## Running the server locally

To build and run the server locally for development:

```shell
just build
```

## Deploying to production

At a minimum, you will need to provision the following infrastructure:

* An instance running the API
* The main Matchbox database (like Postgres)
* A bucket in object storage (like S3)

This requires that the processing of files uploaded by clients happens on the same instance as the API. Thus, it is not advised to run the Matchbox server in this fashion, except for development or very small setups. The Matchbox server variable `MB__SERVER__TASK_RUNNER` needs to be set to "api" for this minimal setup.


For a more robust deployment, set `MB__SERVER__TASK_RUNNER` to "celery", and you will also need:

* A Celery worker
* Redis

Not only does using Redis enable you to offload the long jobs created by the API to another instance, it also lets you horizontally scale the API itself.