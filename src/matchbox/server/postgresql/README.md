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
        string resolution_name
        string full_name
        bytes warehouse_hash
        string db_pk
    }
    SourceColumns {
        bigint column_id PK
        bigint source_id FK
        int column_index
        string column_name
        string column_alias
        string column_type
    }
    Clusters {
        bigint cluster_id PK
        bytes cluster_hash
    }
    ClusterSourcePK {
        bigint pk_id PK
        bigint cluster_id FK
        bigint source_id FK
        string source_pk
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
        bigint resolution_id PK
        bytes resolution_hash
        string type
        string name
        string description
        smallint truth
    }
    ResolutionFrom {
        bigint parent PK,FK
        bigint child PK,FK
        int level
        smallint truth_cache
    }

    Sources |o--|| Resolutions : ""
    Sources ||--o{ SourceColumns : ""
    Sources ||--o{ ClusterSourcePK : ""
    Clusters ||--o{ ClusterSourcePK : ""
    Clusters ||--o{ Probabilities : ""
    Clusters ||--o{ Contains : "parent"
    Contains }o--|| Clusters : "child"
    Resolutions ||--o{ Probabilities : ""
    Resolutions ||--o{ ResolutionFrom : "parent"
    ResolutionFrom }o--|| Resolutions : "child"
```
