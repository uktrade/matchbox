import itertools

import pandas as pd
import pyarrow as pa
import pytest
from sqlalchemy import Engine, text

from matchbox.common.factories.entities import SourceEntity
from matchbox.server.postgresql import MatchboxPostgres
from matchbox.server.postgresql.benchmark.query import (
    compile_match_sql,
    compile_query_sql,
)
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.utils.insert import HashIDMap

from ..fixtures.db import setup_scenario


def test_hash_id_map():
    """Test HashIDMap core functionality including basic operations."""
    # Initialize with some existing mappings
    lookup = pa.Table.from_arrays(
        [
            pa.array([1, 2], type=pa.uint64()),
            pa.array([b"hash1", b"hash2"], type=pa.large_binary()),
        ],
        names=["id", "hash"],
    )
    hash_map = HashIDMap(start=100, lookup=lookup)

    # Test getting existing hashes
    ids = pa.array([2, 1], type=pa.uint64())
    hashes = hash_map.get_hashes(ids)
    assert hashes.to_pylist() == [b"hash2", b"hash1"]

    # Test getting mix of existing and new hashes
    input_hashes = pa.array([b"hash1", b"new_hash", b"hash2"], type=pa.large_binary())
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
    ("point_of_truth", "source"),
    [
        # Test case 1: CDMS/CRN linker, CRN dataset
        pytest.param(
            "probabilistic_naive_test.crn_naive_test.cdms", "crn", id="cdms-crn_crn"
        ),
        # Test case 2: CDMS/CRN linker, CDMS dataset
        pytest.param(
            "probabilistic_naive_test.crn_naive_test.cdms", "cdms", id="cdms-crn_cdms"
        ),
        # Test case 3: CRN/DUNS linker, CRN dataset
        pytest.param(
            "deterministic_naive_test.crn_naive_test.duns", "crn", id="crn-duns_crn"
        ),
        # Test case 4: CRN/DUNS linker, DUNS dataset
        pytest.param(
            "deterministic_naive_test.crn_naive_test.duns", "duns", id="crn-duns_duns"
        ),
    ],
)
@pytest.mark.docker
def test_benchmark_query_generation(
    matchbox_postgres: MatchboxPostgres,
    postgres_warehouse: Engine,
    point_of_truth: str,
    source: str,
):
    with setup_scenario(matchbox_postgres, "link", warehouse=postgres_warehouse) as dag:
        engine = MBDB.get_engine()

        sources_dict = dag.get_sources_for_model(point_of_truth)
        assert len(sources_dict) == 1
        linked = dag.linked[next(iter(sources_dict))]

        true_entities = linked.true_entity_subset(source)
        true_pks = set(
            itertools.chain.from_iterable(
                s for e in true_entities for s in e.source_pks.values()
            )
        )

        sql_query = compile_query_sql(
            point_of_truth=point_of_truth,
            source_address=dag.sources[source].source.address,
        )

        assert isinstance(sql_query, str)

        with engine.connect() as conn:
            res = conn.execute(text(sql_query)).all()

        df = pd.DataFrame(res, columns=["id", "pk"])

        assert df.id.nunique() == len(true_entities)
        assert set(df.pk) == true_pks


@pytest.mark.docker
def test_benchmark_match_query_generation(
    matchbox_postgres: MatchboxPostgres,
    postgres_warehouse: Engine,
):
    with setup_scenario(matchbox_postgres, "link", warehouse=postgres_warehouse) as dag:
        engine = MBDB.get_engine()

        linker_name = "deterministic_naive_test.crn_naive_test.duns"
        duns_testkit = dag.sources.get("duns")

        sources_dict = dag.get_sources_for_model(linker_name)
        assert len(sources_dict) == 1
        linked = dag.linked[next(iter(sources_dict))]

        # A random one:many entity
        source_entity: SourceEntity = linked.find_entities(
            min_appearances={"crn": 2, "duns": 1},
            max_appearances={"duns": 1},
        )[0]

        sql_match = compile_match_sql(
            source_pk=next(iter(source_entity.source_pks["duns"])),
            resolution_name=str(duns_testkit.source.address),
            point_of_truth="deterministic_naive_test.crn_naive_test.duns",
        )

        assert isinstance(sql_match, str)

        with engine.connect() as conn:
            res = conn.execute(text(sql_match)).all()

        df = pd.DataFrame(res, columns=["cluster", "dataset", "source_pk"]).dropna()

        assert df.cluster.nunique() == 1
        assert df.dataset.nunique() == 2
        assert (
            set(df.source_pk)
            == source_entity.source_pks["duns"] | source_entity.source_pks["crn"]
        )
