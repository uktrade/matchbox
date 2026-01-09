# Command Line Interface

Matchbox comes with a Command Line Interface (CLI) that allows users to perform certain management tasks.

To run the CLI:
```shell
matchbox [OPTIONS] COMMAND [ARGS]
```

To get help:
```shell
matchbox --help
```

To get help on specific commands:
```shell
matchbox COMMAND --help
```

## Information commands

### Version
To get the Matchbox client version, run:
```shell
matchbox version
```

### Server status
To get the status of the server and the Matchbox server version, run:
```shell
matchbox server health
```

### Auth status
To get the authentication status of the client, run:
```shell
matchbox auth status
```

## Database maintenance commands

### Delete orphans
When resolutions are modified or deleted, it is possible that the database ends up having clusters which are not related to any table containing sources, models or evaluations. These clusters are considered orphaned, and they should be deleted regularly to reduce bloat.

To do this, run:
```shell
matchbox server delete-orphans
```

This command will print the number of orphaned clusters deleted.