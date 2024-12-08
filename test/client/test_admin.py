import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import dedent

import pytest
from matchbox.admin import load_datasets_from_config
from matchbox.common.db import Source, SourceWarehouse
from tomli_w import dumps


def warehouse_toml(warehouse: SourceWarehouse) -> str:
    return dedent(f"""
        [warehouses.{warehouse.alias}]
        db_type = "{warehouse.db_type}"
        user = "{warehouse.user}"
        password = "{warehouse.password}"
        host = "{warehouse.host}"
        port = {warehouse.port}
        database = "{warehouse.database}"
    """).strip()


def source_toml(source: Source, index: list[dict[str, str]]) -> str:
    index_str = dumps({"index": index}).replace("\n", "\n        ")
    return dedent(f"""
        [datasets.{re.sub(r"[^a-zA-Z0-9]", "", source.alias)}]
        database = "test_warehouse"
        db_schema = "{source.db_schema}"
        db_table = "{source.db_table}"
        db_pk = "{source.db_pk}"
        {index_str}
    """)


@pytest.mark.parametrize(
    "index",
    (
        [
            {"literal": "company_name"},
            {"literal": "crn"},
        ],
        [{"literal": "company_name", "type": "VARCHAR", "alias": "name"}],
        [
            {"literal": "company_name"},
            {"literal": "*"},
        ],
        [
            {"literal": "*"},
            {"literal": "company_name"},
        ],
    ),
    ids=["vanilla", "alias_and_type", "star_end", "star_start"],
)
def test_load_datasets_from_config(
    index: list[dict[str, str]],
    warehouse: SourceWarehouse,
    warehouse_data: list[Source],
):
    """Tests loading datasets from a TOML file."""
    # Construct TOML from CRN data
    # Columns: "id", "company_name", "crn"
    crn = warehouse_data[0]
    raw_toml = dedent(f"""
        {warehouse_toml(warehouse)}
        {source_toml(crn, index)}      
    """).strip()

    with NamedTemporaryFile(suffix=".toml", delete=False) as temp_file:
        temp_file.write(raw_toml.encode())
        temp_file.flush()
        temp_file_path = Path(temp_file.name)

    # Ingest
    config = load_datasets_from_config(temp_file_path)

    # Helper variables
    source = config.get(re.sub(r"[^a-zA-Z0-9]", "", crn.alias))
    named = [idx["literal"] for idx in index if idx["literal"] != "*"]
    has_star = any(idx["literal"] == "*" for idx in index)
    star_pos = next((i for i, idx in enumerate(index) if idx["literal"] == "*"), None)
    col_names = [col.literal.name for col in source.db_columns]

    # Test 1: Core attributes match
    assert source.database == warehouse
    assert source.alias == re.sub(r"[^a-zA-Z0-9]", "", crn.alias)
    assert source.db_schema == crn.db_schema
    assert source.db_table == crn.db_table
    assert source.db_pk == crn.db_pk

    # Test 2: All non-pk columns present
    assert set(col_names) == {"company_name", "crn", "id"} - {source.db_pk}

    # Test 3: Column indexing
    for col in source.db_columns:
        assert col.indexed == (has_star or col.literal.name in named)

    # Test 4: Aliases and types match
    for idx in index:
        if idx["literal"] == "*":
            continue
        col = next(c for c in source.db_columns if c.literal.name == idx["literal"])
        assert col.alias.name == idx.get("alias", idx["literal"])
        assert col.type == idx.get("type", col.type)

    # Test 5: Column ordering
    if star_pos is None:
        for i, name in enumerate(named):
            assert col_names[i] == name
    else:
        for i, idx in enumerate(index):
            if idx["literal"] != "*":
                if i < star_pos:
                    assert col_names[i] == idx["literal"]
                else:
                    star_col_count = len(col_names) - len(index) + 1
                    assert col_names[i + star_col_count - 1] == idx["literal"]

    # Test 6: column equalities

    assert source.db_columns[0] != source.db_columns[1]
    assert source.db_columns[0] == source.db_columns[0]
    assert source.db_columns[1].literal.hash == source.db_columns[1]
    assert source.db_columns[1].alias.hash == source.db_columns[1]
    assert source.db_columns[0].literal.hash != source.db_columns[1]
    assert source.db_columns[0].alias.hash != source.db_columns[1]
