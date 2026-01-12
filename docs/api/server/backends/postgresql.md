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
        text name
    }
    Runs {
        bigint run_id PK
        bigint collection_id FK
        boolean is_mutable
        boolean is_default
    }
    Resolutions {
        bigint resolution_id PK
        bigint run_id FK
        text name
        text description
        text type
        bytea fingerprint
        smallint truth
        enum upload_stage
    }
    ResolutionFrom {
        bigint parent PK,FK
        bigint child PK,FK
        integer level
        smallint truth_cache
    }
    SourceConfigs {
        bigint source_config_id PK
        bigint resolution_id FK
        text location_type
        text location_name
        text extract_transform
    }
    SourceFields {
        bigint field_id PK
        bigint source_config_id FK
        integer index
        text name
        text type
        boolean is_key
    }
    ModelConfigs {
        bigint model_config_id PK
        bigint resolution_id FK
        text model_class
        jsonb model_settings
        jsonb left_query
        jsonb right_query
    }
    Clusters {
        bigint cluster_id PK
        bytea cluster_hash
    }
    ClusterSourceKey {
        bigint key_id PK
        bigint cluster_id FK
        bigint source_config_id FK
        text key
    }
    Contains {
        bigint root PK,FK
        bigint leaf PK,FK
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
    Users {
        bigint user_id PK
        text name
        text email
    }
    Groups {
        bigint group_id PK
        text name
        text description
        boolean is_system
    }
    UserGroups {
        bigint user_id PK,FK
        bigint group_id PK,FK
    }
    Permissions {
        bigint permission_id PK
        text permission
        bigint group_id FK
        bigint collection_id FK
        boolean is_system
    }
    EvalJudgements {
        bigint judgement_id PK
        bigint user_id FK
        bigint endorsed_cluster_id FK
        bigint shown_cluster_id FK
        datetime timestamp
    }

    Collections ||--o{ Runs : ""
    Collections ||--o{ Permissions : ""
    Runs ||--o{ Resolutions : ""
    Resolutions ||--o{ ResolutionFrom : "parent"
    ResolutionFrom }o--|| Resolutions : "child"
    Resolutions |o--|| SourceConfigs : ""
    Resolutions |o--|| ModelConfigs : ""
    Resolutions ||--o{ Probabilities : ""
    Resolutions ||--o{ Results : ""
    SourceConfigs ||--o{ SourceFields : ""
    SourceConfigs ||--o{ ClusterSourceKey : ""
    Clusters ||--o{ ClusterSourceKey : ""
    Clusters ||--o{ Contains : "root"
    Contains }o--|| Clusters : "leaf"
    Clusters ||--o{ Probabilities : ""
    Clusters ||--o{ Results : "left_id"
    Clusters ||--o{ Results : "right_id"
    Clusters ||--o{ EvalJudgements : "endorsed_cluster_id"
    Clusters ||--o{ EvalJudgements : "shown_cluster_id"
    Users ||--o{ UserGroups : ""
    Users ||--o{ EvalJudgements : ""
    Groups ||--o{ UserGroups : ""
    Groups ||--o{ Permissions : ""
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