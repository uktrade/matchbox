# Command line interface

Matchbox comes with a command line interface (CLI) that allows users to perform certain management tasks.

To run the CLI:

```shell
mbx [OPTIONS] COMMAND [ARGS]
```

To get help:

```shell
mbx --help
```

To get help on specific commands:

```shell
mbx COMMAND --help
```

## How do I?

### Bootstrap Matchbox

The first user is always the system admin. As soon as you run the server for the first time, log in to automatically be added to the admins group.

```
mbx login
```

### Manage permissions

Permissions are defined between groups and resources. To manage groups:

```shell
mbx groups
```

For example, to manage permissions on a collection resource, see its related commands:

```shell
mbx collections -c COLLECTION permissions
mbx collections -c COLLECTION grant -g GROUP -p PERMISSION
mbx collections -c COLLECTION revoke -g GROUP -p PERMISSION
```

See [permissions](permissions.md) for more information.

### Delete orphans

When resolutions are modified or deleted, it is possible that the database ends up having clusters which are not related to any table containing sources, models or evaluations. These clusters are considered orphaned, and they should be deleted regularly to reduce bloat.

```shell
mbx admin prune
```

This command will print the number of orphaned clusters deleted.
