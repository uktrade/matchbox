# PostgreSQL

A backend adapter for deploying Matchbox using PostgreSQL.

There are two graph-like trees in place here.

* In the resolution subgraph the tree is implemented as closure table, enabling quick querying of root to leaf paths at the cost of redundancy
* In the data subgraph the tree is implemented as an adjacency list, which means recursive queries are required to resolve it, but less data is stored

```mermaid
erDiagram
    SourceConfigs {
        bigint source_id PK
        bigint resolution_id FK
        string location_type
        string location_uri
        string extract_transform
        bigint identifier_id FK
    }
    SourceFields {
        bigint field_id PK
        bigint source_id FK
        int field_index
        string field_name
        string field_type
    }
    Clusters {
        bigint cluster_id PK
        bytes cluster_hash
    }
    ClusterSourceIdentifiers {
        bigint identifier_id PK
        bigint cluster_id FK
        bigint source_id FK
        string identifier
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
        bytes content_hash
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
    PKSpace {
        bigint id PK
        bigint next_cluster_id
        bigint next_cluster_source_identifier_id
    }

    SourceConfigs |o--|| Resolutions : ""
    SourceConfigs ||--o{ SourceFields : ""
    SourceConfigs ||--|| SourceFields : ""
    SourceConfigs ||--o{ ClusterSourceIdentifiers : ""
    Clusters ||--o{ ClusterSourceIdentifiers : ""
    Clusters ||--o{ Probabilities : ""
    Clusters ||--o{ Contains : "parent"
    Contains }o--|| Clusters : "child"
    Resolutions ||--o{ Probabilities : ""
    Resolutions ||--o{ ResolutionFrom : "parent"
    ResolutionFrom }o--|| Resolutions : "child"
```


::: matchbox.server.postgresql
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        show_submodules: true
        extra:
            show_root_docstring: true