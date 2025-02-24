import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
from rich.logging import RichHandler

from matchbox.common.factories.models import generate_dummy_probabilities
from matchbox.common.hash import hash_data, hash_values
from matchbox.common.transform import (
    attach_components_to_probabilities,
    to_hierarchical_clusters,
)
from matchbox.server.postgresql.benchmark.generate_tables import PRESETS
from matchbox.server.postgresql.utils.insert import HashIDMap

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
pipeline_logger = logging.getLogger("mb_pipeline")

ROOT = Path(__file__).parent.parent


@contextmanager
def timer(description: str):
    start = time.time()
    yield
    elapsed = time.time() - start

    if elapsed >= 60:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        time_str = f"{minutes} min {seconds:.1f} sec"
    else:
        time_str = f"{elapsed:.2f} seconds"

    pipeline_logger.info(f"{description} in {time_str}")


if __name__ == "__main__":
    config = PRESETS["l"]
    left_ids = tuple(range(config["dedupe_components"]))
    right_ids = tuple(
        range(config["dedupe_components"], config["dedupe_components"] * 2)
    )
    probs = generate_dummy_probabilities(
        left_ids, right_ids, [0.6, 1], config["link_components"], config["link_len"]
    )

    all_probs = pa.concat_arrays(
        [probs["left_id"].combine_chunks(), probs["right_id"].combine_chunks()]
    )

    lookup = pa.table(
        {
            "id": all_probs,
            "hash": pa.array(
                [hash_data(p) for p in all_probs.to_pylist()], type=pa.binary()
            ),
        }
    )

    hm = HashIDMap(start=max(right_ids) + 1, lookup=lookup)

    with timer("Full pipeline completed"):
        pipeline_logger.info(f"Processing {len(probs):,} records")

        with timer("Added components"):
            probs_with_ccs = attach_components_to_probabilities(
                pa.table(
                    {
                        "left_id": hm.get_hashes(probs["left_id"]),
                        "right_id": hm.get_hashes(probs["right_id"]),
                        "probability": probs["probability"],
                    }
                )
            )

        with timer("Built hierarchical clusters"):
            hierarchy = to_hierarchical_clusters(
                probabilities=probs_with_ccs,
                hash_func=hash_values,
                dtype=pa.binary,
            )
