"""Shared Core table metadata for the five cluster tables.

Follows the declarative with imperative table pattern: these are plain
SQLAlchemy Core Table objects on one shared MetaData, used directly by
every backend. Relational backends adopt them via __table__ on their own
ORM classes, or use them as-is with Core.

Rules for this module:

- Generic types only (BigInteger, LargeBinary, Text, REAL). No
    dialect-specific types, no native enums. If a column genuinely needs a
    dialect type, use type.with_variant(PGType, "postgresql") on the shared
    column rather than forking the definition.
- The schema is the symbolic token "mb", never rendered literally. Each
    consuming engine resolves it to a real schema, or none, via
    execution_options(schema_translate_map={"mb": ...}). See
    matchbox.server.postgresql.db for the server-side wiring.
- Postgres-only indexes and ingest DDL stay in the server package, not here.

Names match each backend's ORM class exactly (Clusters, ClusterSourceKey,
Contains, ModelEdges, ResolverClusters), so import this module and
reference tables.Clusters rather than importing names directly. These are
Table instances, not classes: the PascalCase name signals the pairing,
not that they can be subclassed or instantiated.

TODO: once a second backend exists, add dedicated tests for this module
and the rest of matchbox.common.adapters.sql, rather than relying solely
on the PostgreSQL adapter test suite.
"""

from sqlalchemy import (
    REAL,
    BigInteger,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    LargeBinary,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

METADATA = MetaData(schema="mb")
"""Shared metadata for the five cluster tables.

schema="mb" is a symbolic token, translated to a real schema, or none, by
each consuming engine via schema_translate_map. Never rendered as a
literal schema name.
"""

Clusters = Table(
    "clusters",
    METADATA,
    Column("cluster_id", BigInteger, primary_key=True),
    Column("cluster_hash", LargeBinary, nullable=False),
    UniqueConstraint("cluster_hash", name="clusters_hash_key"),
)
"""Table of indexed data and clusters that match it."""

ClusterSourceKey = Table(
    "cluster_keys",
    METADATA,
    Column("key_id", BigInteger, primary_key=True),
    Column(
        "cluster_id",
        BigInteger,
        ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "source_config_id",
        BigInteger,
        ForeignKey("source_configs.source_config_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("key", Text, nullable=False),
    Index("ix_cluster_keys_cluster_id", "cluster_id"),
    Index("ix_cluster_keys_keys", "key"),
    Index("ix_cluster_keys_source_config_id", "source_config_id"),
    UniqueConstraint("key_id", "source_config_id", name="unique_keys_source"),
)
"""Table for storing source primary keys for clusters."""

Contains = Table(
    "contains",
    METADATA,
    Column(
        "root",
        BigInteger,
        ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "leaf",
        BigInteger,
        ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    CheckConstraint("root != leaf", name="no_self_containment"),
    UniqueConstraint("root", "leaf"),
    Index("ix_contains_root_leaf", "root", "leaf"),
    Index("ix_contains_leaf_root", "leaf", "root"),
)
"""Cluster lineage table."""

ModelEdges = Table(
    "model_edges",
    METADATA,
    Column("result_id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "step_id",
        BigInteger,
        ForeignKey("steps.step_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "left_id",
        BigInteger,
        ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "right_id",
        BigInteger,
        ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("score", REAL, nullable=False),
    Index("ix_model_edges_step", "step_id"),
    CheckConstraint("score >= 0.0 AND score <= 1.0", name="valid_score"),
    UniqueConstraint("step_id", "left_id", "right_id"),
)
"""Table of results for a model step.

Stores the raw left/right scores created by a model.
"""

ResolverClusters = Table(
    "resolver_clusters",
    METADATA,
    Column(
        "step_id",
        BigInteger,
        ForeignKey("steps.step_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "cluster_id",
        BigInteger,
        ForeignKey("clusters.cluster_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Index("ix_resolver_clusters_step", "step_id"),
)
"""Association table linking resolver steps to cluster IDs."""
