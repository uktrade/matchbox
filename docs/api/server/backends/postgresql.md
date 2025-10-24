# PostgreSQL

A backend adapter for deploying Matchbox using PostgreSQL.

There are two graph-like trees in place here.

* In the resolution subgraph the tree is implemented as closure table, enabling quick querying of root to leaf paths at the cost of redundancy
* In the data subgraph the tree is implemented as a modified closure table which only stores the "root" and "leaf" relationships for each model
    * The leaf IDs
    * The model's proposed cluster IDs at that threshold -- the roots

```mermaid
erDiagram
    Collections {
        bigint collection_id PK
        string name
    }
    Runs {
        bigint run_id PK
        bigint collection_id FK
        bool is_mutable
        bool is_default
    }
    SourceConfigs {
        bigint source_config_id PK
        bigint resolution_id FK
        string location_type
        string location_name
        string extract_transform
    }
    SourceFields {
        bigint field_id PK
        bigint source_config_id FK
        int index
        string name
        string type
        bool is_key
    }
    ModelConfigs {
        bigint model_config_id PK
        bigint resolution_id FK
        string model_class
        jsonb model_settings
        jsonb left_query
        jsonb right_query
    }
    Clusters {
        bigint cluster_id PK
        bytes cluster_hash
    }
    ClusterSourceKey {
        bigint key_id PK
        bigint cluster_id FK
        bigint source_config_id FK
        string key
    }
    Contains {
        bigint root PK,FK
        bigint leaf PK,FK
    }
    PKSpace {
        bigint id
        bigint next_cluster_id
        bigint next_cluster_keys_id
    }
    Probabilities {
        bigint resolution_id PK,FK
        bigint cluster_id PK,FK
        smallint probability
    }
    Results {
        bigint result_id PK
        bigint resolution_id FK
        bigint left_id FK
        bigint right_id FK
        smallint probability
    }
    Resolutions {
        bigint resolution_id PK
        bigint run_id FK
        string name
        string description
        string type
        bytes hash
        smallint truth
    }
    ResolutionFrom {
        bigint parent PK,FK
        bigint child PK,FK
        int level
        smallint truth_cache
    }
    Users {
        bigint user_id PK
        string name
    }
    EvalJudgements {
        bigint judgement_id PK
        bigint user_id FK
        bigint endorsed_cluster_id FK
        bigint shown_cluster_id FK
        datetime timestamp
    }

    Collections ||--o{ Runs : ""
    Runs ||--o{ Resolutions : ""
    SourceConfigs |o--|| Resolutions : ""
    ModelConfigs |o--|| Resolutions : ""
    SourceConfigs ||--o{ SourceFields : ""
    SourceConfigs ||--o{ ClusterSourceKey : ""
    Clusters ||--o{ ClusterSourceKey : ""
    Clusters ||--o{ Probabilities : ""
    Clusters ||--o{ Results : "left_id"
    Clusters ||--o{ Results : "right_id"
    Clusters ||--o{ EvalJudgements : "endorsed_cluster_id"
    Clusters ||--o{ EvalJudgements : "shown_cluster_id" 
    Clusters ||--o{ Contains : "root"
    Contains }o--|| Clusters : "leaf"
    Resolutions ||--o{ Probabilities : ""
    Resolutions ||--o{ Results : ""
    Resolutions ||--o{ ResolutionFrom : "parent"
    ResolutionFrom }o--|| Resolutions : "child"
    Users ||--o{ EvalJudgements : ""
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