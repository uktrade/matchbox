# Command Line Interface

Matchbox comes with a Command Line Interface (CLI) that allows users to perform certain management tasks. It also allows users to access the interactive entity evaluation tool, which is documented [here](evaluation.md).

To run the CLI:
```bash
matchbox [OPTIONS] COMMAND [ARGS]
```

To get help:
```bash
matchbox --help
```

To get help on specific commands:
```bash
matchbox COMMAND --help
```

## Information commands

### Version
To get the Matchbox client version, run:
```bash
matchbox version
```

### Server status
To get the status of the server and the Matchbox server version, run:
```bash
matchbox server health
```

### Auth status
To get the authentication status of the client, run:
```bash
matchbox auth status
```

## Database maintenance commands

### Delete orphans
When resolutions are modified or deleted, it is possible that the `Clusters` table in the database ends up having clusters which are not related to any table containing sources, models or evaluations (see database structure [here](../api/server/backends/postgresql.md)). These clusters are considered orphaned, and they should be deleted regularly to reduce bloat.

To do this, run:
```bash
matchbox server delete-orphans
```

This command will delete clusters from the `Clusters` table whose id does not appear referenced in the following tables:
- `ClusterSourceKey`
- `EvalJudgements`
- `Results`
- `Probabilities`

This deletion will cascade to the `Contains` table.
