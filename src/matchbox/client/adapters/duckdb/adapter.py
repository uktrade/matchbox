"""DuckDB implementation of the local (client-side) Matchbox cluster store.

Shares SQL builders from matchbox.common.adapters.sql with the Postgres
backend. Engine-specific code lives in db.py and orm.py.
"""

import base64
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
from pyarrow import Table as ArrowTable
from sqlalchemy import (
    BigInteger,
    column,
    delete,
    exists,
    func,
    insert,
    literal,
    literal_column,
    select,
    text,
)
from sqlalchemy import (
    table as sa_table,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.selectable import Select

from matchbox.client.adapters.duckdb import db, orm
from matchbox.client.base import MatchboxLocalDBAdapter
from matchbox.common.adapters.protocol import MatchboxLocalBackends, MatchboxSnapshot
from matchbox.common.adapters.sql import tables
from matchbox.common.adapters.sql.insert import (
    select_cluster_map,
    select_contains_pairs,
    select_key_expansion,
    select_new_cluster_hashes,
    select_resolver_membership,
)
from matchbox.common.adapters.sql.query import (
    assemble_matches,
    build_matching_leaves_cte,
    build_target_cluster_cte,
    build_unified_query,
    resolver_membership_subquery,
)
from matchbox.common.adapters.sql.resolver import (
    build_expanded_leaves_subquery,
    build_leaf_hash_groups_query,
    hash_resolver_parents,
)
from matchbox.common.adapters.sql.snapshot import dump_tables, restore_tables
from matchbox.common.arrow import (
    SCHEMA_CLUSTERS,
    SCHEMA_MODEL_EDGES,
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
)
from matchbox.common.dtos import (
    Match,
    ModelStepPath,
    ResolverStepPath,
    SourceStepPath,
    Step,
    StepPath,
    StepType,
)
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
)


def _translate_lineage(
    session: Session, lineage_names: list[str]
) -> list[tuple[int, int | None]]:
    """Translate step names from self.lineage() into (step_id, source_config_id).

    Model steps are dropped: they have no source_config_id and aren't
    queryable, unlike source and resolver steps.
    """
    translated: list[tuple[int, int | None]] = []
    for name in lineage_names:
        step = orm.Steps.from_name(session, name)
        if step.type == StepType.MODEL.value:
            continue
        source_config_id = (
            step.source_config.source_config_id if step.source_config else None
        )
        translated.append((step.step_id, source_config_id))
    return translated


_RAW_DATA_TABLE_PREFIX = "mb_raw_data__"
_QUERY_CACHE_TABLE_PREFIX = "mb_query_cache__"


def _raw_data_table_name(source_step_id: int) -> str:
    """Physical table name for a source step's raw data."""
    return f"{_RAW_DATA_TABLE_PREFIX}{source_step_id}"


def _query_cache_table_name(cache_id: int) -> str:
    """Physical table name for a cached query result."""
    return f"{_QUERY_CACHE_TABLE_PREFIX}{cache_id}"


def _materialise(
    session: Session, physical_name: str, view_name: str, table: ArrowTable
) -> None:
    """Zero-copy materialise an ArrowTable as a physical DuckDB table."""
    raw_conn = session.connection().connection.driver_connection
    raw_conn.register(view_name, table)
    ddl = f'CREATE OR REPLACE TABLE "{physical_name}" AS SELECT * FROM "{view_name}"'
    session.execute(text(ddl))


def _arrow_query(session: Session, stmt: Select) -> ArrowTable:
    """Execute a read-only select via DuckDB's native Arrow export."""
    connection = session.connection()
    schema_translate_map = connection.get_execution_options().get(
        "schema_translate_map"
    )
    compiled = stmt.compile(
        dialect=connection.engine.dialect,
        compile_kwargs={"literal_binds": True},
        schema_translate_map=schema_translate_map,
        render_schema_translate=True,
    )
    raw_conn = connection.connection.driver_connection
    return raw_conn.sql(str(compiled)).to_arrow_table()


def _dump_dynamic_table(session: Session, name: str) -> list[dict]:
    """Fetch a physical table's rows as JSON-safe dicts, bytes wrapped as base64.

    Matches dump_tables' convention, so the snapshot format stays uniform.
    """
    raw_conn = session.connection().connection.driver_connection
    cursor = raw_conn.sql(f'SELECT * FROM "{name}"')
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return [
        {
            col: {"base64": base64.b64encode(val).decode("ascii")}
            if isinstance(val, bytes)
            else val
            for col, val in zip(columns, row, strict=True)
        }
        for row in rows
    ]


def _restore_dynamic_table(session: Session, name: str, rows: list[dict]) -> None:
    """Rebuild a physical table from _dump_dynamic_table's output."""
    decoded = [
        {
            col: base64.b64decode(val["base64"]) if isinstance(val, dict) else val
            for col, val in row.items()
        }
        for row in rows
    ]
    _materialise(session, name, "restore_tmp", pa.Table.from_pylist(decoded))


def _count_source_clusters(session: Session) -> int:
    """Count distinct clusters that have at least one source key."""
    stmt = (
        select(func.count(func.distinct(tables.Clusters.c.cluster_id)))
        .select_from(tables.Clusters)
        .join(
            tables.ClusterSourceKey,
            tables.ClusterSourceKey.c.cluster_id == tables.Clusters.c.cluster_id,
        )
    )
    return session.execute(stmt).scalar_one()


def _count_model_clusters(session: Session) -> int:
    """Count distinct clusters proposed by at least one resolver."""
    stmt = (
        select(func.count(func.distinct(tables.Clusters.c.cluster_id)))
        .select_from(tables.Clusters)
        .join(
            tables.ResolverClusters,
            tables.ResolverClusters.c.cluster_id == tables.Clusters.c.cluster_id,
        )
    )
    return session.execute(stmt).scalar_one()


class _Countable:
    """Binds a session-taking counter to this adapter's session factory.

    Local has no global singleton session like Postgres's MBDB - a
    process may hold several stores - so each count needs its own engine.
    """

    def __init__(
        self, session_factory: sessionmaker, counter: Callable[[Session], int]
    ) -> None:
        self._session_factory = session_factory
        self._counter = counter

    def count(self) -> int:
        """Counts the number of rows matching the statement."""
        with self._session_factory() as session:
            return self._counter(session)


class MatchboxLocalDuckDBQueryMixin:
    """Query mixin for the local DuckDB adapter."""

    def query(  # noqa: D102
        self,
        source: SourceStepPath,
        resolver: ResolverStepPath | None = None,
        return_leaf_id: bool = False,
        limit: int | None = None,
    ) -> ArrowTable:
        with self._session() as session:
            source_step = orm.Steps.from_name(session, source.name, StepType.SOURCE)
            # Fail loudly if the source has no data, rather than return nothing.
            if source_step.source_config is None:
                raise MatchboxDataNotFound(table="source_configs", data=[source.name])

            self_step = (
                source_step
                if resolver is None
                else orm.Steps.from_name(session, resolver.name, StepType.RESOLVER)
            )

            lineage = _translate_lineage(
                session,
                self.lineage(self_step.name, sources=[source_step.name]),
            )

            query_stmt = build_unified_query(
                lineage=lineage, level="key", include_source_config_id=False
            ).order_by(
                literal_column("root_id"),
                literal_column("leaf_id"),
                tables.ClusterSourceKey.c.key,
            )
            if limit is not None:
                query_stmt = query_stmt.limit(limit)

            arrow_table = _arrow_query(session, query_stmt)

        if return_leaf_id:
            return (
                arrow_table.select(["root_id", "key", "leaf_id"])
                .rename_columns(["id", "key", "leaf_id"])
                .cast(SCHEMA_QUERY_WITH_LEAVES)
            )
        return (
            arrow_table.select(["root_id", "key"])
            .rename_columns(["id", "key"])
            .cast(SCHEMA_QUERY)
        )

    def match(  # noqa: D102
        self,
        key: str,
        source: SourceStepPath,
        targets: list[SourceStepPath],
        resolver: ResolverStepPath,
    ) -> list[Match]:
        with self._session() as session:
            resolver_step = orm.Steps.from_name(
                session, resolver.name, StepType.RESOLVER
            )
            source_step = orm.Steps.from_name(session, source.name, StepType.SOURCE)
            if source_step.source_config is None:
                raise MatchboxDataNotFound(table="source_configs", data=[source.name])
            source_config_id = source_step.source_config.source_config_id

            target_config_ids = []
            for target in targets:
                target_step = orm.Steps.from_name(session, target.name, StepType.SOURCE)
                if target_step.source_config is None:
                    raise MatchboxDataNotFound(
                        table="source_configs", data=[target.name]
                    )
                target_config_ids.append(target_step.source_config.source_config_id)

            source_and_target_ids = [source_config_id, *target_config_ids]

            lineage = _translate_lineage(session, self.lineage(resolver_step.name))

            target_cluster_cte = build_target_cluster_cte(
                key=key, source_config_id=source_config_id, lineage=lineage
            )
            matching_leaves_cte = build_matching_leaves_cte(
                source_and_target_ids=source_and_target_ids,
                lineage=lineage,
                target_cluster_cte=target_cluster_cte,
            )

            matched_rows = session.execute(
                select(
                    matching_leaves_cte.c.cluster_id,
                    matching_leaves_cte.c.source_config_id,
                    matching_leaves_cte.c.key,
                )
            ).all()

        return assemble_matches(
            matched_rows=matched_rows,
            source=source,
            source_config_id=source_config_id,
            targets=targets,
            target_config_ids=target_config_ids,
        )


class MatchboxLocalDuckDBDataMixin:
    """Data mixin for the local DuckDB adapter."""

    def create_step(self, step: Step, path: StepPath) -> None:  # noqa: D102
        with self._session() as session:
            existing = session.execute(
                select(orm.Steps).where(orm.Steps.name == path.name)
            ).scalar_one_or_none()

            if existing is None:
                new_step = orm.Steps(
                    name=path.name,
                    type=step.step_type.value,
                    fingerprint=step.fingerprint,
                )
                session.add(new_step)
                session.flush()

                if step.step_type == StepType.SOURCE:
                    session.add(orm.SourceConfigs(step_id=new_step.step_id))
            else:
                existing.type = step.step_type.value
                existing.fingerprint = step.fingerprint

            session.commit()

    def insert_source_data(  # noqa: D102
        self, path: SourceStepPath, data_hashes: ArrowTable
    ) -> None:
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name, StepType.SOURCE)
            self._cascade_invalidate(session, step)
            if step.source_config is None:
                raise MatchboxDataNotFound(table="source_configs", data=[path.name])
            source_config_id = step.source_config.source_config_id

            if data_hashes.num_rows == 0:
                session.commit()
                return

            raw_conn = session.connection().connection.driver_connection
            raw_conn.register("incoming_hashes", data_hashes.select(["hash", "keys"]))
            incoming = sa_table("incoming_hashes", column("hash"), column("keys"))

            session.execute(
                insert(tables.Clusters).from_select(
                    ["cluster_hash"], select_new_cluster_hashes(incoming)
                )
            )
            session.execute(
                insert(tables.ClusterSourceKey).from_select(
                    ["cluster_id", "source_config_id", "key"],
                    select_key_expansion(incoming, source_config_id),
                )
            )
            session.commit()

    def insert_model_data(  # noqa: D102
        self, path: ModelStepPath, results: ArrowTable
    ) -> None:
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name, StepType.MODEL)
            self._cascade_invalidate(session, step)

            if results.num_rows == 0:
                session.commit()
                return

            raw_conn = session.connection().connection.driver_connection
            raw_conn.register(
                "incoming_edges", results.select(["left_id", "right_id", "score"])
            )
            incoming = sa_table(
                "incoming_edges", column("left_id"), column("right_id"), column("score")
            )

            session.execute(
                insert(tables.ModelEdges).from_select(
                    ["step_id", "left_id", "right_id", "score"],
                    select(
                        literal(step.step_id, BigInteger).label("step_id"),
                        incoming.c.left_id,
                        incoming.c.right_id,
                        incoming.c.score,
                    ),
                )
            )
            session.commit()

    def insert_resolver_data(  # noqa: D102
        self, path: ResolverStepPath, data: ArrowTable
    ) -> None:
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name, StepType.RESOLVER)
            self._cascade_invalidate(session, step)

            if data.num_rows == 0:
                session.commit()
                return

            raw_conn = session.connection().connection.driver_connection
            raw_conn.register(
                "incoming_resolver_assignments",
                data.select(["parent_id", "child_id"]),
            )
            incoming = sa_table(
                "incoming_resolver_assignments",
                column("parent_id"),
                column("child_id"),
            )

            expanded_leaves = build_expanded_leaves_subquery(incoming)
            hash_rows = session.execute(
                build_leaf_hash_groups_query(expanded_leaves)
            ).all()
            hash_table = hash_resolver_parents(hash_rows)

            raw_conn.register("resolver_hashes", hash_table)
            resolver_hashes = sa_table(
                "resolver_hashes", column("parent_id"), column("cluster_hash")
            )

            # 1) Bulk-insert new clusters: anti-join, ID omitted (sequence-assigned)
            session.execute(
                insert(tables.Clusters).from_select(
                    ["cluster_hash"],
                    select_new_cluster_hashes(resolver_hashes, hash_col="cluster_hash"),
                )
            )

            # 2) Map parent_id to canonical cluster_id, now hashes exist
            cluster_map = select_cluster_map(resolver_hashes)

            # 3) ResolverClusters: distinct (step_id, cluster_id)
            session.execute(
                insert(tables.ResolverClusters).from_select(
                    ["step_id", "cluster_id"],
                    select_resolver_membership(step.step_id, cluster_map),
                )
            )

            # 4) Contains: new (root, leaf) pairs, anti-joined against existing
            candidate_contains = select_contains_pairs(
                expanded_leaves, cluster_map
            ).subquery()
            session.execute(
                insert(tables.Contains).from_select(
                    ["root", "leaf"],
                    select(candidate_contains.c.root, candidate_contains.c.leaf).where(
                        ~exists(
                            select(1).where(
                                tables.Contains.c.root == candidate_contains.c.root,
                                tables.Contains.c.leaf == candidate_contains.c.leaf,
                            )
                        )
                    ),
                )
            )

            session.commit()

    def get_model_data(self, path: ModelStepPath) -> ArrowTable:  # noqa: D102
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name, StepType.MODEL)
            arrow_table = _arrow_query(
                session,
                select(
                    tables.ModelEdges.c.left_id,
                    tables.ModelEdges.c.right_id,
                    tables.ModelEdges.c.score,
                ).where(tables.ModelEdges.c.step_id == step.step_id),
            )
        return arrow_table.cast(SCHEMA_MODEL_EDGES)

    def get_resolver_data(self, path: ResolverStepPath) -> ArrowTable:  # noqa: D102
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name, StepType.RESOLVER)
            membership = resolver_membership_subquery(
                step_id=step.step_id, alias="assignments"
            )
            stmt = select(
                membership.c.root_id.label("parent_id"),
                membership.c.leaf_id.label("child_id"),
            ).order_by(membership.c.root_id, membership.c.leaf_id)
            arrow_table = _arrow_query(session, stmt)

        return arrow_table.cast(SCHEMA_CLUSTERS)

    def dump(self) -> MatchboxSnapshot:  # noqa: D102
        with self._session() as session:
            data = dump_tables(session, [*orm.LOCAL_TABLES, *orm.SHARED_TABLES])
            physical_names = [
                _raw_data_table_name(row["source_step_id"]) for row in data["raw_data"]
            ] + [
                _query_cache_table_name(row["cache_id"]) for row in data["query_cache"]
            ]
            # dump_tables only covers catalog rows - physical tables need
            # dumping separately, same row format.
            data["__dynamic_cache_tables__"] = {
                name: _dump_dynamic_table(session, name) for name in physical_names
            }
        return MatchboxSnapshot(backend_type=MatchboxLocalBackends.DUCKDB, data=data)

    def restore(self, snapshot: MatchboxSnapshot) -> None:  # noqa: D102
        if snapshot.backend_type != MatchboxLocalBackends.DUCKDB:
            raise TypeError(
                f"Cannot restore {snapshot.backend_type} snapshot to duckdb backend"
            )

        self.clear(certain=True)

        with self._session() as session:
            restore_tables(
                session, [*orm.LOCAL_TABLES, *orm.SHARED_TABLES], snapshot.data
            )
            for name, rows in snapshot.data.get("__dynamic_cache_tables__", {}).items():
                _restore_dynamic_table(session, name, rows)
            session.commit()

    def clear(self, certain: bool) -> None:  # noqa: D102
        if not certain:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop all rows in the database but not the "
                "tables themselves. Rerun with certain=True to continue."
            )
        with self._session() as session:
            for tbl in [*orm.SHARED_TABLES, *orm.LOCAL_TABLES]:
                session.execute(delete(tbl))
            session.commit()
        db.drop_dynamic_cache_tables(
            self._engine, (_RAW_DATA_TABLE_PREFIX, _QUERY_CACHE_TABLE_PREFIX)
        )

    def drop(self, certain: bool) -> None:  # noqa: D102
        if not certain:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database and recreate it. "
                "Rerun with certain=True to continue."
            )
        db.drop_dynamic_cache_tables(
            self._engine, (_RAW_DATA_TABLE_PREFIX, _QUERY_CACHE_TABLE_PREFIX)
        )
        db.drop_db(self._engine, [*orm.SHARED_TABLES, *orm.LOCAL_TABLES])
        db.create_db(self._engine, orm.LOCAL_TABLES)
        db.create_db(self._engine, orm.SHARED_TABLES)


class MatchboxLocalDuckDBCacheMixin:
    """Local-only mixin: raw data, query cache, cascade invalidation."""

    def insert_raw_data(self, path: SourceStepPath, table: ArrowTable) -> None:  # noqa: D102
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name, StepType.SOURCE)
            physical_name = _raw_data_table_name(step.step_id)
            _materialise(session, physical_name, "incoming_raw_data", table)
            session.execute(
                delete(orm.RawData).where(orm.RawData.source_step_id == step.step_id)
            )
            session.add(orm.RawData(source_step_id=step.step_id))
            session.commit()

    def get_raw_data(  # noqa: D102
        self, path: SourceStepPath, keys: list[str] | None = None
    ) -> ArrowTable:
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name, StepType.SOURCE)
            row = session.execute(
                select(orm.RawData).where(orm.RawData.source_step_id == step.step_id)
            ).scalar_one_or_none()

            if row is None:
                raise MatchboxDataNotFound(table="raw_data", data=[path.name])

            physical_name = _raw_data_table_name(row.source_step_id)
            raw_conn = session.connection().connection.driver_connection
            if keys is None:
                return raw_conn.sql(f'SELECT * FROM "{physical_name}"').to_arrow_table()
            raw_conn.register("key_filter", pa.table({"key": keys}))
            return raw_conn.sql(
                f'SELECT t.* FROM "{physical_name}" t '
                'SEMI JOIN "key_filter" f ON t.key = f.key'
            ).to_arrow_table()

    def cache_query(  # noqa: D102
        self, key: str, table: ArrowTable, depends_on: list[StepPath]
    ) -> None:
        with self._session() as session:
            existing = session.execute(
                select(orm.QueryCache).where(orm.QueryCache.cache_key == key)
            ).scalar_one_or_none()
            if existing is not None:
                old_name = _query_cache_table_name(existing.cache_id)
                session.execute(text(f'DROP TABLE IF EXISTS "{old_name}"'))
                session.execute(
                    delete(orm.QueryCacheStep).where(
                        orm.QueryCacheStep.cache_id == existing.cache_id
                    )
                )
                session.execute(
                    delete(orm.QueryCache).where(
                        orm.QueryCache.cache_id == existing.cache_id
                    )
                )

            row = orm.QueryCache(cache_key=key)
            session.add(row)
            session.flush()  # assigns cache_id from its duckdb sequence default

            physical_name = _query_cache_table_name(row.cache_id)
            _materialise(session, physical_name, "incoming_query_cache", table)

            step_ids = {
                orm.Steps.from_name(session, p.name).step_id for p in depends_on
            }
            session.add_all(
                orm.QueryCacheStep(cache_id=row.cache_id, step_id=step_id)
                for step_id in step_ids
            )
            session.commit()

    def get_cached_query(self, key: str) -> ArrowTable | None:  # noqa: D102
        with self._session() as session:
            row = session.execute(
                select(orm.QueryCache).where(orm.QueryCache.cache_key == key)
            ).scalar_one_or_none()
            if row is None:
                return None
            physical_name = _query_cache_table_name(row.cache_id)
            raw_conn = session.connection().connection.driver_connection
            return raw_conn.sql(f'SELECT * FROM "{physical_name}"').to_arrow_table()

    def drop_step_data(self, path: StepPath) -> None:  # noqa: D102
        with self._session() as session:
            step = orm.Steps.from_name(session, path.name)
            self._cascade_invalidate(session, step)
            session.commit()

    def _cascade_invalidate(self, session: Session, step: "orm.Steps") -> None:
        """Drop a step's data and its descendants', and their query cache.

        Responsibility for committing lies with the caller.

        Deletes are unconditional and type-agnostic: harmless no-ops for
        tables a step doesn't apply to. RawData and Clusters/Contains are
        left alone - canonical state and content-addressed rows, not
        caches. Query cache invalidation is selective, via QueryCacheStep.
        """
        step_ids = [step.step_id]
        for name in self.descendants(step.name):
            descendant = orm.Steps.from_name(session, name)
            step_ids.append(descendant.step_id)

        session.execute(
            delete(tables.ModelEdges).where(tables.ModelEdges.c.step_id.in_(step_ids))
        )
        session.execute(
            delete(tables.ResolverClusters).where(
                tables.ResolverClusters.c.step_id.in_(step_ids)
            )
        )
        session.execute(
            delete(tables.ClusterSourceKey).where(
                tables.ClusterSourceKey.c.source_config_id.in_(
                    select(orm.SourceConfigs.source_config_id).where(
                        orm.SourceConfigs.step_id.in_(step_ids)
                    )
                )
            )
        )

        affected_ids = (
            session.execute(
                select(orm.QueryCacheStep.cache_id)
                .where(orm.QueryCacheStep.step_id.in_(step_ids))
                .distinct()
            )
            .scalars()
            .all()
        )
        for cache_id in affected_ids:
            session.execute(
                text(f'DROP TABLE IF EXISTS "{_query_cache_table_name(cache_id)}"')
            )
        session.execute(
            delete(orm.QueryCacheStep).where(
                orm.QueryCacheStep.cache_id.in_(affected_ids)
            )
        )
        session.execute(
            delete(orm.QueryCache).where(orm.QueryCache.cache_id.in_(affected_ids))
        )


class MatchboxLocalDuckDB(
    MatchboxLocalDuckDBQueryMixin,
    MatchboxLocalDuckDBDataMixin,
    MatchboxLocalDuckDBCacheMixin,
    MatchboxLocalDBAdapter,
):
    """A DuckDB adapter for the local (client-side) Matchbox cluster store."""

    def __init__(self, path: Path | None = None) -> None:
        """Initialise the DuckDB adapter.

        Args:
            path: Path to the duckdb file. None means in-memory and
                ephemeral, dying with the process.
        """
        self._engine: Engine = db.create_local_engine(path)
        self._session_factory: sessionmaker = sessionmaker(bind=self._engine)

        db.create_db(self._engine, orm.LOCAL_TABLES)
        db.create_db(self._engine, orm.SHARED_TABLES)

        self.all_clusters = _Countable(self._session_factory, orm.Clusters.count)
        self.source_clusters = _Countable(self._session_factory, _count_source_clusters)
        self.model_clusters = _Countable(self._session_factory, _count_model_clusters)
        self.creates = _Countable(self._session_factory, orm.ResolverClusters.count)
        self.merges = _Countable(self._session_factory, orm.Contains.count)
        self.proposes = _Countable(self._session_factory, orm.ModelEdges.count)

    def _session(self) -> Session:
        """Return a new session bound to this adapter's engine."""
        return self._session_factory()
