# 🔥 Matchbox PostgreSQL backend

A backend adapter for deploying Matchbox using PostgreSQL.

Currently implements the following architecture. See Confluence for [further details](https://uktrade.atlassian.net/wiki/spaces/CDL/pages/4282908700/Matchbox+0.2+architecture+ideas).

`Resolutions.type` is one of "model", "dataset" or "human".

There are two graph-like trees in place here.

* In the resolution subgraph the tree is implemented as closure table, enabling quick querying of root to leaf paths at the cost of redundancy
* In the data subgraph the tree is implemented as a hierarchy, which means recursive queries are required to resolve it, but less data is stored

```mermaid
erDiagram
    Sources {
        bigint resolution_id PK,FK
        string alias
        string full_name
        bytes warehouse_hash
        string id
        array[string] column_names
        array[string] column_aliases
        array[string] column_types
    }
    Clusters {
        bigint cluster_id PK,FK
        bytes cluster_hash
        bigint dataset FK
        array[string] source_pk
    }
    Contains {
        bigint parent PK,FK
        bigint child PK,FK
    }
    Probabilities {
        bigint resolution PK,FK
        bigint cluster PK,FK
        smallint probability
    }
    Resolutions {
        bigint resolution_id PK,FK
        bytes resolution_hash
        enum type
        string name
        string description
        int truth
    }
    ResolutionFrom {
        bigint parent PK,FK
        bigint child PK,FK
        int level
        int truth_cache
    }

    Sources |o--|| Resolutions : ""
    Sources ||--o{ Clusters : ""
    Clusters ||--o{ Contains : "parent, child"
    Clusters ||--o{ Probabilities : ""
    Resolutions ||--o{ Probabilities : ""
    Resolutions ||--o{ ResolutionFrom : "child, parent"
```
