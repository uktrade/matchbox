from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pytest
from sqlalchemy import text

from matchbox.common.db import Source
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.benchmark.generate_tables import (
    generate_all_tables,
    generate_result_tables,
)
from matchbox.server.postgresql.benchmark.init_schema import create_tables, empty_schema
from matchbox.server.postgresql.benchmark.query import compile_query_sql
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.utils.insert import HashIDMap

from ..fixtures.db import SetupDatabaseCallable


def test_benchmark_init_schema():
    schema = MBDB.MatchboxBase.metadata.schema
    count_tables = text(f"""
        select count(*)
        from information_schema.tables
        where table_schema = '{schema}';
    """)

    with MBDB.get_engine().connect() as con:
        con.execute(text(empty_schema()))
        con.commit()
        n_tables = int(con.execute(count_tables).scalar())
        assert n_tables == 0

        con.execute(text(create_tables()))
        con.commit()
        n_tables_expected = len(MBDB.MatchboxBase.metadata.tables)
        n_tables = int(con.execute(count_tables).scalar())
        assert n_tables == n_tables_expected


@pytest.mark.parametrize(
    "left_ids, right_ids, next_id, n_components, n_probs",
    (
        [range(10_000), None, 10_000, 8000, 2000],
        [range(8000), range(8000, 16_000), 16_000, 6000, 10_000],
    ),
    ids=["dedupe", "link"],
)
def test_benchmark_result_tables(left_ids, right_ids, next_id, n_components, n_probs):
    resolution_id = None

    (top_clusters, _, _, _, _) = generate_result_tables(
        left_ids, right_ids, resolution_id, next_id, n_components, n_probs
    )

    assert len(top_clusters) == n_components


def test_benchmark_generate_tables():
    schema = MBDB.MatchboxBase.metadata.schema

    def array_encode(array: Iterable[str]):
        if not array:
            return None
        escaped_l = [f'"{s}"' for s in array]
        list_rep = ", ".join(escaped_l)
        return "{" + list_rep + "}"

    with MBDB.get_engine().connect() as con:
        con.execute(text(empty_schema()))
        con.commit()

        results = generate_all_tables(20, 5, 25, 5, 25)

        assert len(results) == len(MBDB.MatchboxBase.metadata.tables)

        for table_name, table_arrow in results.items():
            df = table_arrow.to_pandas()
            # Pandas' `to_sql` dislikes arrays
            if "source_pk" in df.columns:
                df["source_pk"] = df["source_pk"].apply(array_encode)
            # Pandas' `to_sql` dislikes large unsigned ints
            for c in df.columns:
                if df[c].dtype == "uint64":
                    df[c] = df[c].astype("int64")
            df.to_sql(name=table_name, con=con, schema=schema)


def test_hash_id_map():
    """Test HashIDMap core functionality including basic operations."""
    # Initialize with some existing mappings
    lookup = pa.Table.from_arrays(
        [
            pa.array([1, 2], type=pa.uint64()),
            pa.array([b"hash1", b"hash2"], type=pa.binary()),
        ],
        names=["id", "hash"],
    )
    hash_map = HashIDMap(start=100, lookup=lookup)

    # Test getting existing hashes
    ids = pa.array([2, 1], type=pa.uint64())
    hashes = hash_map.get_hashes(ids)
    assert hashes.to_pylist() == [b"hash2", b"hash1"]

    # Test getting mix of existing and new hashes
    input_hashes = pa.array([b"hash1", b"new_hash", b"hash2"], type=pa.binary())
    returned_ids = hash_map.get_ids(input_hashes)

    # Verify results
    id_list = returned_ids.to_pylist()
    assert id_list[0] == 1  # Existing hash1
    assert id_list[2] == 2  # Existing hash2
    assert id_list[1] == 100  # New hash got next available ID

    # Verify lookup table was updated correctly
    assert hash_map.lookup.shape == (3, 3)
    assert hash_map.next_int == 101

    # Test error handling for missing IDs
    with pytest.raises(ValueError) as exc_info:
        hash_map.get_hashes(pa.array([999], type=pa.uint64()))
    assert "not found in lookup table" in str(exc_info.value)


@pytest.mark.parametrize(
    ("parameters"),
    [
        # Test case 1: CDMS/CRN linker, CRN dataset
        {
            "point_of_truth": "deterministic_naive_test.cdms_naive_test.crn",
            "source_index": 0,  # CRN
            "unique_ids": 1_000,
            "unique_pks": 3_000,
        },
        # Test case 2: CDMS/CRN linker, CDMS dataset
        {
            "point_of_truth": "deterministic_naive_test.cdms_naive_test.crn",
            "source_index": 2,  # CDMS
            "unique_ids": 1_000,
            "unique_pks": 2_000,
        },
        # Test case 3: CRN/DUNS linker, CRN dataset
        {
            "point_of_truth": "deterministic_naive_test.crn_naive_test.duns",
            "source_index": 0,  # CRN
            "unique_ids": 1_000,
            "unique_pks": 3_000,
        },
        # Test case 4: CRN/DUNS linker, DUNS dataset
        {
            "point_of_truth": "deterministic_naive_test.crn_naive_test.duns",
            "source_index": 1,  # DUNS
            "unique_ids": 500,
            "unique_pks": 500,
        },
    ],
    ids=["cdms-crn_crn", "cdms-crn_cdms", "crn-duns_crn", "crn-duns_duns"],
)
def test_benchmark_query_generation(
    setup_database: SetupDatabaseCallable,
    matchbox_postgres: MatchboxPostgres,
    warehouse_data: list[Source],
    parameters: dict[str, Any],
):
    setup_database(matchbox_postgres, warehouse_data, "link")

    engine = MBDB.get_engine()
    point_of_truth = parameters["point_of_truth"]
    idx = parameters["source_index"]
    dataset_name = f"{warehouse_data[idx].db_schema}.{warehouse_data[idx].db_table}"

    sql_query = compile_query_sql(
        point_of_truth=point_of_truth, dataset_name=dataset_name
    )

    assert isinstance(sql_query, str)

    with engine.connect() as conn:
        res = conn.execute(text(sql_query)).all()

    df = pd.DataFrame(res, columns=["id", "pk"])

    assert df.id.nunique() == parameters["unique_ids"]
    assert df.pk.nunique() == parameters["unique_pks"]
