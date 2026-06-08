"""Split model edges/resolver configs and drop truth cache.

Revision ID: e06333c2daaf
Revises: ecb39d1cc5c2
Create Date: 2026-02-19 11:18:35.699303

"""

import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e06333c2daaf"
down_revision: str | None = "ecb39d1cc5c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "mb"
MODEL_CONFIGS = sa.Table(
    "model_configs",
    sa.MetaData(),
    sa.Column("model_config_id", sa.BIGINT()),
    sa.Column("model_settings", postgresql.JSONB()),
    sa.Column("left_query", postgresql.JSONB()),
    sa.Column("right_query", postgresql.JSONB()),
    schema=SCHEMA,
)


def _rename_constraint(table_name: str, old_name: str, new_name: str) -> None:
    """Rename a table constraint."""
    op.execute(
        sa.text(
            f"ALTER TABLE {SCHEMA}.{table_name} "
            f"RENAME CONSTRAINT {old_name} TO {new_name}"
        )
    )


def _rename_sequence(old_name: str, new_name: str) -> None:
    """Rename a sequence in the Matchbox schema."""
    op.execute(sa.text(f"ALTER SEQUENCE {SCHEMA}.{old_name} RENAME TO {new_name}"))


def _load_json_payload(value: object | None) -> dict[str, object] | None:
    """Load JSON payloads stored either as JSON text or JSON objects."""
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object payload, got {type(value)!r}")
    return value


def _rewrite_model_query_configs() -> None:
    """Normalise legacy JSON text payloads and rewrite query config shape."""
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(
            MODEL_CONFIGS.c.model_config_id,
            MODEL_CONFIGS.c.model_settings,
            MODEL_CONFIGS.c.left_query,
            MODEL_CONFIGS.c.right_query,
        )
    ).mappings()

    for row in rows:
        model_settings = _load_json_payload(row["model_settings"])
        left_query = _load_json_payload(row["left_query"])
        right_query = _load_json_payload(row["right_query"])

        bind.execute(
            MODEL_CONFIGS.update()
            .where(MODEL_CONFIGS.c.model_config_id == row["model_config_id"])
            .values(
                model_settings=model_settings,
                left_query=(
                    None
                    if left_query is None
                    else left_query
                    if "sources" in left_query
                    else {
                        "sources": left_query["source_resolutions"],
                        "resolver": (
                            None
                            if left_query.get("model_resolution") is None
                            else f"{left_query['model_resolution']}_resolver"
                        ),
                        "combine_type": left_query.get("combine_type", "concat"),
                        "cleaning": left_query.get("cleaning"),
                    }
                ),
                right_query=(
                    None
                    if right_query is None
                    else right_query
                    if "sources" in right_query
                    else {
                        "sources": right_query["source_resolutions"],
                        "resolver": (
                            None
                            if right_query.get("model_resolution") is None
                            else f"{right_query['model_resolution']}_resolver"
                        ),
                        "combine_type": right_query.get("combine_type", "concat"),
                        "cleaning": right_query.get("cleaning"),
                    }
                ),
            )
        )


def upgrade() -> None:
    """Upgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Extend the resolution type constraint to include the new resolver type.
    op.drop_constraint(
        "resolution_type_constraints",
        "resolutions",
        schema=SCHEMA,
        type_="check",
    )
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source', 'resolver')",
        schema=SCHEMA,
    )

    # Keep existing model edge rows by renaming the results table. The index is
    # recreated immediately with its final name to avoid a rename step later.
    op.rename_table("results", "model_edges", schema=SCHEMA)
    op.drop_index(
        op.f("ix_results_resolution"), table_name="model_edges", schema=SCHEMA
    )
    op.create_index(
        "ix_model_edges_step",
        "model_edges",
        ["resolution_id"],
        unique=False,
        schema=SCHEMA,
    )

    # Convert model_edges.probability from a SMALLINT percentage (0–100) to a
    # REAL fraction (0.0–1.0). This is done before any column renames so the
    # postgresql_using expression can still reference the original column name.
    op.drop_constraint(
        op.f("valid_probability"), "model_edges", schema=SCHEMA, type_="check"
    )
    op.alter_column(
        "model_edges",
        "probability",
        existing_type=sa.SMALLINT(),
        type_=sa.REAL(),
        existing_nullable=False,
        postgresql_using="probability::real / 100.0",
        schema=SCHEMA,
    )

    # Rename probabilities → resolver_clusters
    op.rename_table("probabilities", "resolver_clusters", schema=SCHEMA)
    op.drop_index(
        op.f("ix_probabilities_resolution"),
        table_name="resolver_clusters",
        schema=SCHEMA,
    )
    op.create_index(
        "ix_resolver_clusters_step",
        "resolver_clusters",
        ["resolution_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.drop_constraint(
        op.f("valid_probability"), "resolver_clusters", schema=SCHEMA, type_="check"
    )
    op.drop_column("resolver_clusters", "probability", schema=SCHEMA)
    _rename_constraint(
        "resolver_clusters", "probabilities_pkey", "resolver_clusters_pkey"
    )
    _rename_constraint(
        "resolver_clusters",
        "probabilities_resolution_id_fkey",
        "resolver_clusters_step_id_fkey",
    )
    _rename_constraint(
        "resolver_clusters",
        "probabilities_cluster_id_fkey",
        "resolver_clusters_cluster_id_fkey",
    )

    # Create resolver_configs using step_id as the column name directly, so no
    # column rename is needed after the resolutions → steps table rename below.
    op.create_table(
        "resolver_configs",
        sa.Column(
            "resolver_config_id",
            sa.BIGINT(),
            sa.Identity(always=False, start=1),
            nullable=False,
        ),
        sa.Column("step_id", sa.BIGINT(), nullable=False),
        sa.Column("resolver_class", sa.TEXT(), nullable=False),
        sa.Column(
            "resolver_settings",
            postgresql.JSONB(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["step_id"], [f"{schema}.resolutions.resolution_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("resolver_config_id"),
        sa.UniqueConstraint("step_id", name="resolver_configs_step_key"),
        schema=SCHEMA,
    )

    # For each existing model resolution, create a paired resolver resolution.
    # Models keep their model_edges but lose their resolver_clusters, which move
    # to the new resolver.
    # The resolver builds its Components settings from the threshold stored in
    # resolutions.truth (read here before the column is dropped below).
    #
    # upload_stage is supplied explicitly because the READY default is ORM-level only
    # and will not fire on a raw SQL INSERT.
    #
    # Fingerprint is set to 32 zero bytes — a sentinel that will never match a real
    # upload hash, so all clients will be forced to re-upload and replace it. 32 bytes
    # matches the SHA-256 output of matchbox.common.hash.HASH_FUNC.
    op.execute(
        sa.text("""
            INSERT INTO {schema}.resolutions (
                name,
                type,
                description,
                fingerprint,
                run_id,
                upload_stage
            )
            SELECT
                r.name || '_resolver',
                'resolver',
                'Resolver for ' || r.name,
                decode(repeat('00', 32), 'hex'),
                r.run_id,
                'READY'
            FROM {schema}.resolutions r
            WHERE r.type = 'model'
        """)
    )

    op.execute(
        sa.text("""
            INSERT INTO {schema}.resolver_configs (
                step_id,
                resolver_class,
                resolver_settings
            )
            SELECT
                resolver.resolution_id,
                'Components',
                jsonb_build_object(
                    'thresholds',
                    jsonb_strip_nulls(
                        jsonb_build_object(
                            model.name,
                            CASE
                                WHEN model.truth IS NULL THEN NULL
                                ELSE model.truth::real / 100.0
                            END
                        )
                    )
                )
            FROM {schema}.resolutions model
            JOIN {schema}.resolutions resolver
                ON resolver.name = model.name || '_resolver'
                AND resolver.run_id = model.run_id
            WHERE model.type = 'model'
        """)
    )
    _rewrite_model_query_configs()

    # Move cluster associations from the model to its paired resolver.
    op.execute(
        sa.text("""
            UPDATE {schema}.resolver_clusters rc
            SET resolution_id = resolver.resolution_id
            FROM {schema}.resolutions model
            JOIN {schema}.resolutions resolver
                ON resolver.name = model.name || '_resolver'
                AND resolver.run_id = model.run_id
            WHERE rc.resolution_id = model.resolution_id
                AND model.type = 'model'
        """)
    )

    # Add the resolver into the closure table: first the direct model → resolver
    # edge, then all transitive ancestors so the closure table remains complete.
    op.execute(
        sa.text("""
            INSERT INTO {schema}.resolution_from (parent, child, level)
            SELECT
                model.resolution_id,
                resolver.resolution_id,
                1
            FROM {schema}.resolutions model
            JOIN {schema}.resolutions resolver
                ON resolver.name = model.name || '_resolver'
                AND resolver.run_id = model.run_id
            WHERE model.type = 'model'
        """)
    )
    op.execute(
        sa.text("""
            INSERT INTO {schema}.resolution_from (parent, child, level)
            SELECT
                rf.parent,
                resolver.resolution_id,
                rf.level + 1
            FROM {schema}.resolution_from rf
            JOIN {schema}.resolutions model ON model.resolution_id = rf.child
            JOIN {schema}.resolutions resolver
                ON resolver.name = model.name || '_resolver'
                AND resolver.run_id = model.run_id
            WHERE model.type = 'model'
        """)
    )

    op.drop_column("resolutions", "truth", schema=SCHEMA)
    op.drop_column("resolution_from", "truth_cache", schema=SCHEMA)

    # Rename tables, columns, the sequence, and all constraints to their final names.
    op.rename_table("resolutions", "steps", schema=SCHEMA)
    op.rename_table("resolution_from", "step_from", schema=SCHEMA)

    op.alter_column("steps", "resolution_id", new_column_name="step_id", schema=SCHEMA)
    op.alter_column(
        "source_configs", "resolution_id", new_column_name="step_id", schema=SCHEMA
    )
    op.alter_column(
        "model_configs", "resolution_id", new_column_name="step_id", schema=SCHEMA
    )
    op.alter_column(
        "model_edges", "resolution_id", new_column_name="step_id", schema=SCHEMA
    )
    op.alter_column(
        "resolver_clusters", "resolution_id", new_column_name="step_id", schema=SCHEMA
    )
    op.alter_column(
        "model_edges", "probability", new_column_name="score", schema=SCHEMA
    )

    op.create_check_constraint(
        op.f("valid_score"),
        "model_edges",
        "score >= 0.0 AND score <= 1.0",
        schema=SCHEMA,
    )

    _rename_sequence("resolutions_resolution_id_seq", "steps_step_id_seq")

    _rename_constraint("steps", "resolutions_pkey", "steps_pkey")
    _rename_constraint("steps", "resolutions_name_key", "steps_name_key")
    _rename_constraint("steps", "resolutions_run_fkey", "steps_run_fkey")
    _rename_constraint("steps", "resolution_type_constraints", "step_type_constraints")

    _rename_constraint("step_from", "resolution_from_pkey", "step_from_pkey")
    _rename_constraint(
        "step_from", "resolution_from_parent_fkey", "step_from_parent_fkey"
    )
    _rename_constraint(
        "step_from", "resolution_from_child_fkey", "step_from_child_fkey"
    )

    _rename_constraint(
        "model_configs",
        "model_configs_resolution_id_fkey",
        "model_configs_step_id_fkey",
    )
    _rename_constraint(
        "source_configs",
        "sources_resolution_id_fkey",
        "source_configs_step_id_fkey",
    )

    _rename_constraint("model_edges", "results_pkey", "model_edges_pkey")
    _rename_constraint(
        "model_edges", "results_resolution_id_fkey", "model_edges_step_id_fkey"
    )
    _rename_constraint(
        "model_edges", "results_left_id_fkey", "model_edges_left_id_fkey"
    )
    _rename_constraint(
        "model_edges", "results_right_id_fkey", "model_edges_right_id_fkey"
    )
    _rename_constraint(
        "model_edges",
        "results_resolution_id_left_id_right_id_key",
        "model_edges_step_id_left_id_right_id_key",
    )


def downgrade() -> None:
    """Downgrade schema."""
    schema = op.get_context().config.get_main_option("db_schema")
    # Reverse all constraint renames before touching columns or tables, so that
    # every subsequent operation can use the restored pre-migration names.

    _rename_constraint(
        "model_edges",
        "model_edges_step_id_left_id_right_id_key",
        "results_resolution_id_left_id_right_id_key",
    )
    _rename_constraint(
        "model_edges", "model_edges_right_id_fkey", "results_right_id_fkey"
    )
    _rename_constraint(
        "model_edges", "model_edges_left_id_fkey", "results_left_id_fkey"
    )
    _rename_constraint(
        "model_edges", "model_edges_step_id_fkey", "results_resolution_id_fkey"
    )
    _rename_constraint("model_edges", "model_edges_pkey", "results_pkey")

    _rename_constraint(
        "source_configs",
        "source_configs_step_id_fkey",
        "sources_resolution_id_fkey",
    )
    _rename_constraint(
        "model_configs",
        "model_configs_step_id_fkey",
        "model_configs_resolution_id_fkey",
    )

    _rename_constraint(
        "step_from", "step_from_child_fkey", "resolution_from_child_fkey"
    )
    _rename_constraint(
        "step_from", "step_from_parent_fkey", "resolution_from_parent_fkey"
    )
    _rename_constraint("step_from", "step_from_pkey", "resolution_from_pkey")

    _rename_constraint("steps", "step_type_constraints", "resolution_type_constraints")
    _rename_constraint("steps", "steps_run_fkey", "resolutions_run_fkey")
    _rename_constraint("steps", "steps_name_key", "resolutions_name_key")
    _rename_constraint("steps", "steps_pkey", "resolutions_pkey")

    # Rename resolver_clusters constraints directly to their probabilities_* names
    _rename_constraint(
        "resolver_clusters",
        "resolver_clusters_cluster_id_fkey",
        "probabilities_cluster_id_fkey",
    )
    _rename_constraint(
        "resolver_clusters",
        "resolver_clusters_step_id_fkey",
        "probabilities_resolution_id_fkey",
    )
    _rename_constraint(
        "resolver_clusters", "resolver_clusters_pkey", "probabilities_pkey"
    )

    _rename_sequence("steps_step_id_seq", "resolutions_resolution_id_seq")

    # Rename columns back.
    op.alter_column(
        "model_edges", "score", new_column_name="probability", schema=SCHEMA
    )
    op.alter_column(
        "resolver_clusters", "step_id", new_column_name="resolution_id", schema=SCHEMA
    )
    op.alter_column(
        "model_edges", "step_id", new_column_name="resolution_id", schema=SCHEMA
    )
    op.alter_column(
        "model_configs", "step_id", new_column_name="resolution_id", schema=SCHEMA
    )
    op.alter_column(
        "source_configs", "step_id", new_column_name="resolution_id", schema=SCHEMA
    )
    op.alter_column("steps", "step_id", new_column_name="resolution_id", schema=SCHEMA)

    # Rename tables back. resolver_clusters is handled separately below.
    op.rename_table("step_from", "resolution_from", schema=SCHEMA)
    op.rename_table("steps", "resolutions", schema=SCHEMA)

    # Reverse the REAL conversion: drop valid_score, convert probability back to a
    # SMALLINT percentage, and restore the original check constraint.
    op.drop_constraint(op.f("valid_score"), "model_edges", schema=SCHEMA, type_="check")
    op.alter_column(
        "model_edges",
        "probability",
        existing_type=sa.REAL(),
        type_=sa.SMALLINT(),
        existing_nullable=False,
        postgresql_using="ROUND(probability * 100.0)::smallint",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        op.f("valid_probability"),
        "model_edges",
        "probability BETWEEN 0 AND 100",
        schema=SCHEMA,
    )

    # Restore truth_cache to resolution_from before re-adding truth to resolutions.
    op.add_column(
        "resolution_from",
        sa.Column("truth_cache", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema=SCHEMA,
    )

    # Remove resolver type from the constraint before deleting resolver rows,
    # otherwise the DELETE would violate the check.
    op.drop_constraint(
        "resolution_type_constraints",
        "resolutions",
        schema=SCHEMA,
        type_="check",
    )
    # Downgrading to a schema without the resolver type, so remove resolver rows
    # first. Cascades clear resolver lineage, resolver_configs, and resolver_clusters.
    # The paired model resolutions are left intact but will have no cluster
    # associations. This split is not reversible.
    op.execute(sa.text(f"DELETE FROM {schema}.resolutions WHERE type = 'resolver'"))
    op.create_check_constraint(
        "resolution_type_constraints",
        "resolutions",
        "type IN ('model', 'source')",
        schema=SCHEMA,
    )

    # Restore truth column. Value is lost and will be NULL on downgrade.
    op.add_column(
        "resolutions",
        sa.Column("truth", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema=SCHEMA,
    )

    # resolver_configs rows are removed by the cascade above, but the table itself
    # must be dropped explicitly.
    op.drop_table("resolver_configs", schema=SCHEMA)

    # Restore probability column on resolver_clusters, defaulting existing rows to
    # 100 before tightening the constraint to NOT NULL.
    op.add_column(
        "resolver_clusters",
        sa.Column("probability", sa.SMALLINT(), autoincrement=False, nullable=True),
        schema=SCHEMA,
    )
    op.execute(
        sa.text(
            f"UPDATE {schema}.resolver_clusters "
            "SET probability = 100 "
            "WHERE probability IS NULL"
        )
    )
    op.alter_column(
        "resolver_clusters",
        "probability",
        existing_type=sa.SMALLINT(),
        nullable=False,
        schema=SCHEMA,
    )
    op.create_check_constraint(
        op.f("valid_probability"),
        "resolver_clusters",
        "probability >= 0 AND probability <= 100",
        schema=SCHEMA,
    )

    # Rename resolver_clusters back to probabilities. Constraints were already renamed
    # above. Drop the final index and recreate with the original name.
    op.drop_index(
        "ix_resolver_clusters_step", table_name="resolver_clusters", schema=SCHEMA
    )
    op.rename_table("resolver_clusters", "probabilities", schema=SCHEMA)
    op.create_index(
        op.f("ix_probabilities_resolution"),
        "probabilities",
        ["resolution_id"],
        unique=False,
        schema=SCHEMA,
    )

    # Rename model_edges back to results. Drop the final index and recreate with
    # the original name.
    op.drop_index("ix_model_edges_step", table_name="model_edges", schema=SCHEMA)
    op.rename_table("model_edges", "results", schema=SCHEMA)
    op.create_index(
        op.f("ix_results_resolution"),
        "results",
        ["resolution_id"],
        unique=False,
        schema=SCHEMA,
    )