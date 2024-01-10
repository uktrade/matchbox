from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import (
    LABEL_STYLE_TABLENAME_PLUS_COL,
    Engine,
    String,
    and_,
    func,
    or_,
    select,
)
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.selectable import Select

from cmf.data import (
    ENGINE,
    Clusters,
    DDupeContains,
    LinkContains,
    Models,
    SourceData,
    clusters_association,
)
from cmf.data.utils import get_schema_table_names, string_to_dataset, string_to_table


def selector(
    table: str, fields: List[str], engine: Engine = ENGINE
) -> Dict[str, List[str]]:
    """
    Takes the full name of a table and the fields you want to select from it,
    and arranges them in a dictionary parsable by query().

    Args:
        table: a table name in the form "schema.table".
        fields: a list of columns in the table
        engine: (optional) the engine to use to connect and validate the
            selector

    Returns:
        A dictionary of the validated table name and fields
    """
    db_schema, db_table = get_schema_table_names(table, validate=True)
    selected_table = string_to_table(
        db_schema=db_schema, db_table=db_table, engine=engine
    )

    all_cols = set(selected_table.c.keys())
    selected_cols = set(fields)
    if not selected_cols <= all_cols:
        raise ValueError(
            f"{selected_cols.difference(all_cols)} not found in "
            f"{selected_table.schema}.{selected_table.name}"
        )

    return {f"{selected_table.schema}.{selected_table.name}": fields}


def selectors(*selector: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {k: v for d in (selector) for k, v in d.items()}


def get_all_parents(model: Union[Models, List[Models]]) -> List[Models]:
    """
    Takes a Models object and returns all items in its parent tree.
    """
    result = []
    if isinstance(model, list):
        for mod in model:
            parents = get_all_parents(mod)
            result.append(parents)
        return result
    elif isinstance(model, Models):
        result.append(model)
        parent_neighbours = model.parent_neighbours()
        if len(parent_neighbours) == 0:
            return result
        else:
            for mod in parent_neighbours:
                parents = get_all_parents(mod)
                result += parents
            return result


def get_all_children(model: Union[Models, List[Models]]) -> List[Models]:
    """
    Takes a Models object and returns all items in its child tree.
    """
    result = []
    if isinstance(model, list):
        for mod in model:
            children = get_all_children(mod)
            result.append(children)
        return result
    elif isinstance(model, Models):
        result.append(model)
        child_neighbours = model.child_neighbours()
        if len(child_neighbours) == 0:
            return result
        else:
            for mod in child_neighbours:
                children = get_all_children(mod)
                result += children
            return result


def parent_to_tree(
    model_name: str, engine: Engine = ENGINE
) -> Tuple[bytes, List[bytes]]:
    """
    Takes the string name of a model and returns a tuple of its SHA-1,
    and the SHA-1 list of its children.
    """

    with Session(engine) as session:
        model = session.query(Models).filter_by(name=model_name).first()
        model_children = get_all_children(model)
        model_children.pop(0)  # includes original model

    return model.sha1, [m.sha1 for m in model_children]


def tree_to_reachable_stmt(model_tree: List[bytes]) -> Select:
    """
    Takes a list of models and returns a query to select their reachable
    edges.
    """
    c1 = aliased(Clusters)
    c2 = aliased(Clusters)

    dd_stmt = (
        select(DDupeContains.parent, DDupeContains.child)
        .join(c1, DDupeContains.parent == c1.sha1)
        .join(clusters_association, clusters_association.c.child == c1.sha1)
        .join(Models, clusters_association.c.parent == Models.sha1)
        .where(Models.sha1.in_(model_tree))
    )

    lk_stmt = (
        select(LinkContains.parent, LinkContains.child)
        .join(c1, LinkContains.parent == c1.sha1)
        .join(c2, LinkContains.child == c2.sha1)
        .join(clusters_association, clusters_association.c.child == c1.sha1)
        .join(Models, clusters_association.c.parent == Models.sha1)
        .where(Models.sha1.in_(model_tree))
    )

    return dd_stmt.union(lk_stmt)


def reachable_to_parent_data_stmt(reachable_stmt: Select, parent_sha1: bytes) -> Select:
    """
    Takes a select statement representing the reachable edges of a parent
    model and returns a statement to create a parent cluster to child data
    lookup
    """
    allowed = reachable_stmt.cte("allowed")

    root = (
        select(allowed.c.parent, allowed.c.child)
        .join(Clusters, Clusters.sha1 == allowed.c.parent)
        .join(clusters_association, clusters_association.c.child == Clusters.sha1)
        .join(Models, clusters_association.c.parent == Models.sha1)
        .where(Models.sha1 == parent_sha1)
        .cte("root")
    )

    recurse_top = select(root.c.parent, root.c.child).cte("recurse", recursive=True)
    recurse_bottom = select(recurse_top.c.parent, allowed.c.child).join(
        recurse_top, allowed.c.parent == recurse_top.c.child
    )
    recurse = recurse_top.union(recurse_bottom)

    return recurse


def selector_to_data(
    selector: Dict[str, List[str]],
    engine: Engine = ENGINE,
) -> Select:
    select_stmt = []
    join_stmts = []
    where_stmts = []
    for schema_table, fields in selector.items():
        db_schema, db_table = get_schema_table_names(schema_table)

        cmf_dataset = string_to_dataset(db_schema, db_table, engine=engine)
        db_table = string_to_table(db_schema, db_table, engine=engine)

        # To handle array column
        source_data_unested = select(
            SourceData.sha1, func.unnest(SourceData.id).label("id"), SourceData.dataset
        ).cte("source_data_unnested")

        select_stmt.append(db_table.c[tuple(fields)])
        join_stmts.append(
            {
                "target": db_table,
                "onclause": and_(
                    source_data_unested.c.id
                    == db_table.c[cmf_dataset.db_id].cast(String),
                    source_data_unested.c.dataset == cmf_dataset.uuid,
                ),
            }
        )
        where_stmts.append(db_table.c[cmf_dataset.db_id] != None)  # NoQA E711

    stmt = select(
        source_data_unested.c.sha1.label("data_sha1"), *select_stmt
    ).select_from(source_data_unested)

    for join_stmt in join_stmts:
        stmt = stmt.join(**join_stmt, isouter=True)

    stmt = stmt.where(or_(*where_stmts))
    stmt = stmt.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)

    return stmt


def query(
    selector: Dict[str, List[str]],
    model: Optional[str] = None,
    return_type: str = "pandas",
    engine: Engine = ENGINE,
    limit: int = None,
) -> Union[pd.DataFrame, ChunkedIteratorResult]:
    """
    Takes the dictionaries of tables and fields outputted by selectors and
    queries database for them. If a model "point of truth" is supplied, will
    attach the clusters this data belongs to.
    """
    if model is None:
        # We want raw data with no clusters
        final_stmt = selector_to_data(selector, engine=engine)
    else:
        # We want raw data with clusters attached
        parent, child = parent_to_tree(model, engine=engine)
        if len(parent) == 0:
            raise ValueError(f"Model {model} not found")
        tree = [parent] + child
        reachable_stmt = tree_to_reachable_stmt(tree)
        lookup_stmt = reachable_to_parent_data_stmt(reachable_stmt, parent)
        data_stmt = selector_to_data(selector, engine=engine).cte()

        final_stmt = select(lookup_stmt.c.parent.label("cluster_sha1"), data_stmt).join(
            lookup_stmt, lookup_stmt.c.child == data_stmt.c.data_sha1
        )

    if limit is not None:
        final_stmt = final_stmt.limit(limit)

    if return_type == "pandas":
        with Session(engine) as session:
            res = pd.read_sql(final_stmt, session.bind)
    elif return_type == "sqlalchemy":
        with Session(engine) as session:
            res = session.execute(final_stmt)
    else:
        ValueError(f"return_type of {return_type} not valid")

    return res
