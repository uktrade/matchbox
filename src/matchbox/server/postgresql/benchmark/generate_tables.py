"""Generate tables for benchmarking PostgreSQL backend."""

from itertools import chain
from pathlib import Path
from typing import Iterable

import click
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from matchbox.common.factories.models import generate_dummy_probabilities
from matchbox.common.hash import HASH_FUNC, hash_data, hash_values
from matchbox.common.logging import console
from matchbox.common.transform import (
    attach_components_to_probabilities,
    to_hierarchical_clusters,
)
from matchbox.server.postgresql.utils.insert import HashIDMap

PRESETS = {
    "xs": {
        "source_len": 10_000,
        "dedupe_components": 8000,
        "dedupe_len": 2000,
        "link_components": 6000,
        "link_len": 10_000,
    },
    "s": {
        "source_len": 100_000,
        "dedupe_components": 80_000,
        "dedupe_len": 20_000,
        "link_components": 60_000,
        "link_len": 100_000,
    },
    "m": {
        "source_len": 1_000_000,
        "dedupe_components": 800_000,
        "dedupe_len": 200_000,
        "link_components": 600_000,
        "link_len": 1_000_000,
    },
    "l": {
        "source_len": 10_000_000,
        "dedupe_components": 8_000_000,
        "dedupe_len": 2_000_000,
        "link_components": 6_000_000,
        "link_len": 10_000_000,
    },
    "xl": {
        "source_len": 100_000_000,
        "dedupe_components": 80_000_000,
        "dedupe_len": 20_000_000,
        "link_components": 60_000_000,
        "link_len": 100_000_000,
    },
}


def _hash_list_int(li: list[int]) -> list[bytes]:
    return [HASH_FUNC(str(i).encode("utf-8")).digest() for i in li]


def generate_sources(
    dataset_start_id: int = 1,
) -> tuple[pa.Table, pa.Table]:
    """Generate sources and source_columns tables.

    Args:
        dataset_start_id: Starting ID for dataset resolution IDs

    Returns:
        Tuple of (sources_table, source_columns_table) as PyArrow tables
    """
    # Sources data
    sources_id = [dataset_start_id, dataset_start_id + 1]
    sources_resolution_id = sources_id
    sources_resolution_names = ["source1@warehouse", "source2@warehouse"]
    sources_full_names = ["dbt.companies_house", "dbt.hmrc_exporters"]
    sources_db_pk = ["company_number", "id"]
    warehouse_hashes = [bytes("foo".encode("ascii"))] * 2

    # Create sources table without column arrays
    sources_table = pa.table(
        {
            "source_id": pa.array(sources_id, type=pa.uint64()),
            "resolution_id": pa.array(sources_resolution_id, type=pa.uint64()),
            "resolution_name": pa.array(sources_resolution_names, type=pa.string()),
            "full_name": pa.array(sources_full_names, type=pa.string()),
            "warehouse_hash": pa.array(warehouse_hashes, type=pa.large_binary()),
            "db_pk": pa.array(sources_db_pk, type=pa.string()),
        }
    )

    # Column data
    column_names = [["col1"], ["col2"]]
    column_types = [["TEXT"], ["TEXT"]]

    # Create flattened data for source_columns table
    column_ids = []
    source_ids = []
    column_indices = []
    column_name_values = []
    column_type_values = []

    # Start with column_id = 1
    next_column_id = 1

    # Process each source and its columns
    for i, source_id in enumerate(sources_id):
        for j, (name, type_val) in enumerate(
            zip(column_names[i], column_types[i], strict=True)
        ):
            column_ids.append(next_column_id)
            next_column_id += 1
            source_ids.append(source_id)
            column_indices.append(j)
            column_name_values.append(name)
            column_type_values.append(type_val)

    # Create source_columns table
    source_columns_table = pa.table(
        {
            "column_id": pa.array(column_ids, type=pa.uint64()),
            "source_id": pa.array(source_ids, type=pa.uint64()),
            "column_index": pa.array(column_indices, type=pa.int32()),
            "column_name": pa.array(column_name_values, type=pa.string()),
            "column_type": pa.array(column_type_values, type=pa.string()),
        }
    )

    return sources_table, source_columns_table


def generate_resolutions(dataset_start_id: int = 1) -> pa.Table:
    """Generate resolutions table.

    Args:
        dataset_start_id: Starting ID for dataset resolution IDs

    Returns:
        PyArrow resolutions table
    """
    base_id = dataset_start_id
    resolutions_resolution_id = [
        base_id,
        base_id + 1,
        base_id + 2,
        base_id + 3,
        base_id + 4,
    ]
    resolutions_name = ["source1", "source2", "dedupe1", "dedupe2", "link"]
    resolutions_resolution_hash = [
        HASH_FUNC(rid.encode("utf-8")).digest() for rid in resolutions_name
    ]
    resolutions_type = ["dataset", "dataset", "model", "model", "model"]
    resolutions_truth = [None, None, 80, 80, 90]

    return pa.table(
        {
            "resolution_id": pa.array(resolutions_resolution_id, type=pa.uint64()),
            "resolution_hash": pa.array(
                resolutions_resolution_hash, type=pa.large_binary()
            ),
            "type": pa.array(resolutions_type, type=pa.string()),
            "name": pa.array(resolutions_name, type=pa.string()),
            "description": pa.array(resolutions_name, type=pa.string()),
            "truth": pa.array(resolutions_truth, type=pa.int8()),
        }
    )


def generate_resolution_from(dataset_start_id: int = 1) -> pa.Table:
    """Generate resolution_from table.

    Args:
        dataset_start_id: Starting ID for dataset resolution IDs

    Returns:
        PyArrow resolution_from table
    """
    base_id = dataset_start_id
    # 1 and 2 are sources; 3 and 4 are dedupers; 5 is a linker
    resolution_parent = [
        base_id,
        base_id,
        base_id + 2,
        base_id + 1,
        base_id + 1,
        base_id + 3,
    ]
    resolution_child = [
        base_id + 2,
        base_id + 4,
        base_id + 4,
        base_id + 3,
        base_id + 4,
        base_id + 4,
    ]
    resolution_level = [1, 2, 1, 1, 2, 1]
    resolution_truth_cache = [None, None, 70, None, None, 70]

    return pa.table(
        {
            "parent": pa.array(resolution_parent, type=pa.uint64()),
            "child": pa.array(resolution_child, type=pa.uint64()),
            "level": pa.array(resolution_level, type=pa.int32()),
            "truth_cache": pa.array(resolution_truth_cache, type=pa.uint8()),
        }
    )


def generate_cluster_source(
    range_left: int,
    range_right: int,
    source_id: int,
    cluster_start_id: int = 0,
    pk_start_id: int = 0,  # Need to track the global pk_id to ensure uniqueness
) -> tuple[pa.Table, pa.Table]:
    """Generate both Clusters and ClusterSourcePK tables for source rows.

    Args:
        range_left: first ID to generate
        range_right: last ID to generate, plus one
        source_id: source ID for the source
        cluster_start_id: Starting ID for clusters
        pk_start_id: Starting ID for primary keys

    Returns:
        Tuple of (Clusters table, ClusterSourcePK table)
    """
    # Generate cluster IDs
    source = list(range(cluster_start_id + range_left, cluster_start_id + range_right))

    # Create the Clusters table (without dataset column)
    clusters_table = pa.table(
        {
            "cluster_id": pa.array(source, type=pa.uint64()),
            "cluster_hash": pa.array(_hash_list_int(source), type=pa.large_binary()),
        }
    )

    # Generate data and create ClusterSourcePK table
    pk_ids = []
    cluster_ids = []
    source_ids = []
    source_pks = []

    for i, cluster_id in enumerate(source):
        # Create a single source_pk entry for each cluster
        pk_ids.append(pk_start_id + i)
        cluster_ids.append(cluster_id)
        source_ids.append(source_id)
        source_pks.append(str(cluster_id))

    cluster_source_pks_table = pa.table(
        {
            "pk_id": pa.array(pk_ids, type=pa.int64()),
            "cluster_id": pa.array(cluster_ids, type=pa.uint64()),
            "source_id": pa.array(source_ids, type=pa.uint64()),
            "source_pk": pa.array(source_pks, type=pa.string()),
        }
    )

    return clusters_table, cluster_source_pks_table


def generate_result_tables(
    left_ids: Iterable[int],
    right_ids: Iterable[int] | None,
    resolution_id: int,
    next_id: int,
    n_components: int,
    n_probs: int,
    prob_min: float = 0.6,
    prob_max: float = 1,
) -> tuple[list[int], pa.Table, pa.Table, pa.Table, int]:
    """Generate probabilities, contains and clusters tables.

    Args:
        left_ids: list of IDs for rows to dedupe, or for left rows to link
        right_ids: list of IDs for right rows to link
        resolution_id: ID of resolution for this dedupe or link model
        next_id: the next ID to use when generating IDs
        n_components: number of implied connected components
        n_probs: total number of probability edges to be generated
        prob_min: minimum value for probabilities to be generated
        prob_max: maximum value for probabilities to be generated

    Returns:
        Tuple with 1 list of top-level clusters, 3 PyArrow tables, for probabilities,
        contains, and clusters, and the next ID to use for future calls
    """
    probs = generate_dummy_probabilities(
        tuple(left_ids),
        tuple(right_ids) if right_ids else None,
        (prob_min, prob_max),
        n_components,
        n_probs,
    )

    # Create a lookup table for hashes
    all_probs = pa.concat_arrays(
        [probs["left_id"].combine_chunks(), probs["right_id"].combine_chunks()]
    )

    lookup = pa.table(
        {
            "id": all_probs,
            "hash": pa.array(
                [hash_data(p) for p in all_probs.to_pylist()], type=pa.large_binary()
            ),
        }
    )

    hm = HashIDMap(start=next_id, lookup=lookup)

    # Join hashes, probabilities and components
    probs_with_ccs = attach_components_to_probabilities(
        pa.table(
            {
                "left_id": hm.get_hashes(probs["left_id"]),
                "right_id": hm.get_hashes(probs["right_id"]),
                "probability": probs["probability"],
            }
        )
    )

    # Calculate hierarchies
    hierarchy = to_hierarchical_clusters(
        probabilities=probs_with_ccs,
        hash_func=hash_values,
        dtype=pa.large_binary,
    )

    # Shape into tables
    new_hierarchy_schema = pa.schema(
        [
            pa.field("parent", pa.uint64()),
            pa.field("child", pa.uint64()),
            pa.field("probability", pa.uint8()),
        ]
    )
    hierarchy = pa.Table.from_arrays(
        [
            pa.array(hm.get_ids(hierarchy["parent"]), type=pa.uint64()),
            pa.array(hm.get_ids(hierarchy["child"]), type=pa.uint64()),
            hierarchy["probability"],
        ],
        schema=new_hierarchy_schema,
    ).sort_by([("parent", "ascending")])

    prev_parent = None
    unique_indices = []
    i = -1
    for batch in hierarchy.to_batches():
        d = batch.to_pydict()
        for parent in d["parent"]:
            i += 1
            if parent != prev_parent:
                unique_indices.append(i)
                prev_parent = parent

    mask = np.full((len(hierarchy)), False)
    mask[unique_indices] = True
    unique_parent_hierarchy = hierarchy.filter(mask=mask)
    unique_parent_ids = unique_parent_hierarchy["parent"].combine_chunks()

    parent_ids = hierarchy["parent"]
    child_ids = hierarchy["child"]
    unique_child_ids = pc.unique(child_ids)

    probabilities_table = pa.table(
        {
            "resolution": pa.array(
                [resolution_id] * len(unique_parent_ids), type=pa.uint64()
            ),
            "cluster": unique_parent_ids,
            "probability": unique_parent_hierarchy["probability"],
        }
    )

    contains_table = pa.table(
        {
            "parent": parent_ids,
            "child": child_ids,
        }
    )

    clusters_table = pa.table(
        {
            "cluster_id": unique_parent_ids,
            "cluster_hash": hm.get_hashes(unique_parent_ids),
        }
    )

    # Compute top clusters
    parents_not_children = pc.filter(
        unique_parent_ids, pc.invert(pc.is_in(unique_parent_ids, unique_child_ids))
    )
    right_ids_or_empty = [] if not right_ids else right_ids
    all_sources = pa.array(chain(left_ids, right_ids_or_empty), type=pa.uint64())

    sources_not_children = pc.filter(
        all_sources, pc.invert(pc.is_in(all_sources, unique_child_ids))
    )
    top_clusters = pc.unique(
        pa.concat_arrays([parents_not_children, sources_not_children])
    )

    return (
        top_clusters,
        probabilities_table,
        contains_table,
        clusters_table,
        hm.next_int,
    )


def generate_all_tables(
    source_len: int,
    dedupe_components: int,
    dedupe_len: int,
    link_components: int,
    link_len: int,
    cluster_start_id: int = 0,
    dataset_start_id: int = 1,
    pk_start_id: int = 0,
) -> dict[str, pa.Table]:
    """Make all six PostgreSQL backend tables.

    It will create two sources, one deduper for each, and one linker from
    each deduper.

    Args:
        source_len: length of each data source
        dedupe_components: number of connected components implied by each deduper
        dedupe_len: probabilities generated by each deduper
        link_components: number of connected components implied by each linker
        link_len: probabilities generated by each linker
        cluster_start_id: Starting ID for clusters
        dataset_start_id: Starting ID for dataset resolution IDs
        pk_start_id: Starting ID for primary keys (globally unique)

    Returns:
        A dictionary where keys are table names and values are PyArrow tables
    """
    console.log("Generating sources")
    resolutions = generate_resolutions(dataset_start_id)
    resolution_from = generate_resolution_from(dataset_start_id)
    sources, columns = generate_sources(dataset_start_id)

    clusters_source1, source_pks1 = generate_cluster_source(
        range_left=0,
        range_right=source_len,
        source_id=dataset_start_id,
        cluster_start_id=cluster_start_id,
        pk_start_id=pk_start_id,
    )

    current_pk_id = pk_start_id + len(source_pks1)

    clusters_source2, source_pks2 = generate_cluster_source(
        range_left=source_len,
        range_right=source_len * 2,
        source_id=dataset_start_id + 1,
        cluster_start_id=cluster_start_id,
        pk_start_id=current_pk_id,
    )

    current_pk_id += len(source_pks2)

    initial_next_id = cluster_start_id + (source_len * 2)

    console.log("Generating the deduplication tables")
    (
        top_clusters1,
        probabilities_dedupe1,
        contains_dedupe1,
        clusters_dedupe1,
        next_id1,
    ) = generate_result_tables(
        left_ids=clusters_source1["cluster_id"].to_pylist(),
        right_ids=None,
        resolution_id=dataset_start_id + 2,
        next_id=initial_next_id,
        n_components=dedupe_components,
        n_probs=dedupe_len,
    )

    (
        top_clusters2,
        probabilities_dedupe2,
        contains_dedupe2,
        clusters_dedupe2,
        next_id2,
    ) = generate_result_tables(
        left_ids=clusters_source2["cluster_id"].to_pylist(),
        right_ids=None,
        resolution_id=dataset_start_id + 3,
        next_id=next_id1,
        n_components=dedupe_components,
        n_probs=dedupe_len,
    )

    console.log("Generating the link tables")
    _, probabilities_link, contains_link, clusters_link, final_next_id = (
        generate_result_tables(
            left_ids=top_clusters1,
            right_ids=top_clusters2,
            resolution_id=dataset_start_id + 4,
            next_id=next_id2,
            n_components=link_components,
            n_probs=link_len,
        )
    )

    probabilities = pa.concat_tables(
        [probabilities_dedupe1, probabilities_dedupe2, probabilities_link]
    ).combine_chunks()
    contains = pa.concat_tables(
        [contains_dedupe1, contains_dedupe2, contains_link]
    ).combine_chunks()
    clusters = pa.concat_tables(
        [
            clusters_source1,
            clusters_source2,
            clusters_dedupe1,
            clusters_dedupe2,
            clusters_link,
        ]
    ).combine_chunks()

    # Combine the source primary keys tables
    cluster_source_pks = pa.concat_tables([source_pks1, source_pks2]).combine_chunks()

    console.log("Generation complete.")
    console.log(f"Next dataset id: {dataset_start_id + 5}")
    console.log(f"Next cluster id: {final_next_id}")
    console.log(f"Next pk id: {current_pk_id}")

    return {
        "resolutions": resolutions,
        "resolution_from": resolution_from,
        "sources": sources,
        "source_columns": columns,
        "clusters": clusters,
        "cluster_source_pks": cluster_source_pks,
        "contains": contains,
        "probabilities": probabilities,
    }


@click.command()
@click.option("-s", "--settings", type=str, required=True, help="Settings dict to use.")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory to save parquets to.",
)
@click.option(
    "-c", "--cluster-start-id", type=int, default=0, help="Starting ID for clusters"
)
@click.option(
    "-d",
    "--dataset-start-id",
    type=int,
    default=1,
    help="Starting ID for dataset resolution IDs",
)
def main(
    settings: str,
    output_dir: Path,
    cluster_start_id: int,
    dataset_start_id: int,
) -> None:
    """Command line tool for generating data.

    Args:
        settings: The key of the settings dictionary to use
        output_dir: Where to save the output files
        cluster_start_id: The first integer to use for clusters
        dataset_start_id: The first integer to use for datasets

    Examples:
        ```shell
        generate_tables.py \
            --settings xl \
            --output-dir data/v4 \
            --dataset-start-id 6742 \
            --cluster-start-id 7
        ```
        ```shell
        generate_tables.py -s s -o data/v4 -d 1 -c 0
        ```
    """
    if not output_dir:
        output_dir = Path.cwd() / "data" / "all_tables"
    if settings not in PRESETS:
        raise ValueError(f"Settings {settings} are invalid")

    config = PRESETS[settings]
    source_len = config["source_len"]
    dedupe_components = config["dedupe_components"]
    dedupe_len = config["dedupe_len"]
    link_len = config["link_len"]
    link_components = config["link_components"]

    all_tables = generate_all_tables(
        source_len=source_len,
        dedupe_components=dedupe_components,
        dedupe_len=dedupe_len,
        link_components=link_components,
        link_len=link_len,
        cluster_start_id=cluster_start_id,
        dataset_start_id=dataset_start_id,
    )

    output_dir /= settings
    output_dir.mkdir(parents=True, exist_ok=True)

    console.log("Writing to disk")
    for name, table in all_tables.items():
        pq.write_table(table, output_dir / f"{name}.parquet")

    console.log("Complete")


if __name__ == "__main__":
    main()
