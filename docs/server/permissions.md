!!! warning
    Permissions are an experimental feature in Matchbox. Disable them with `MB__SERVER__AUTHORISATION=False`.

    They currently assume an external identity provider uses a public, private and JWT generated elsewhere, and places them into the Matchbox client and server environments.

Matchbox uses a Role-Based Access Control (RBAC) system to manage security. Permissions are granted to **Groups**, and **Users** are assigned membership in those groups.

## Bootstrapping

When a Matchbox server is first deployed, it has no admins. To bootstrap setup:

```shell
mbx login
```

* The **first user** to log in is automatically assigned to the **admins** group
* This user becomes the initial system administrator
* All subsequent users are treated as standard users

Bootstrapping is not done at the user's risk.

## Defaults

* **admins**: Members of this group have full control over the system
* **public**: Every user who logs in is automatically added to the `public` group. This ensures baseline access control for all authenticated users

## Permission hierarchy

Permissions in Matchbox are hierarchical. Possessing a higher-level permission automatically grants all lower-level permissions for that resource.

| Level | Capability | Includes |
| :--- | :--- | :--- |
| **Admin** | Manage permissions, delete resources, full control. | Read, Write |
| **Write** | Create and modify data (e.g., upload sources, run models). | Read |
| **Read** | View data, configurations, and results. | *None* |

### Resources

Permissions are applied to specific resources:

* **System**: Permissions apply to the system as a whole, such as group creation or dumping the database
* **Collection**: Permissions apply only to a specific named collection, such as read access to the "finance" collection
