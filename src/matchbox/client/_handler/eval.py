"""Evaluation functions for the client handler."""

import zipfile
from io import BytesIO

import polars as pl
from pyarrow import Table
from pyarrow.parquet import read_table

from matchbox.client._handler.main import CLIENT, http_retry, url_params
from matchbox.common.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_JUDGEMENTS,
    JudgementsZipFilenames,
    check_schema,
)
from matchbox.common.dtos import (
    ModelResolutionPath,
)
from matchbox.common.eval import Judgement
from matchbox.common.logging import logger


@http_retry
def sample_for_eval(n: int, resolution: ModelResolutionPath) -> Table:
    """Sample model results for evaluation."""
    res = CLIENT.get(
        "/eval/samples",
        params=url_params(
            {
                "n": n,
                "collection": resolution.collection,
                "run_id": resolution.run,
                "resolution": resolution.name,
            }
        ),
    )

    return read_table(BytesIO(res.content))


@http_retry
def send_eval_judgement(judgement: Judgement) -> None:
    """Send judgements to the server."""
    logger.debug(f"Submitting judgement {judgement.shown}:{judgement.endorsed} ")
    CLIENT.post("/eval/judgements", json=judgement.model_dump())


@http_retry
def download_eval_data(tag: str | None = None) -> tuple[Table, Table]:
    """Download all judgements from the server."""
    logger.debug("Retrieving all judgements.")
    res = CLIENT.get("/eval/judgements", params=url_params({"tag": tag}))

    zip_bytes = BytesIO(res.content)
    with zipfile.ZipFile(zip_bytes, "r") as zip_file:
        with zip_file.open(JudgementsZipFilenames.JUDGEMENTS) as f1:
            judgements = read_table(f1)

        with zip_file.open(JudgementsZipFilenames.EXPANSION) as f2:
            expansion = read_table(f2)

    logger.debug("Finished retrieving judgements.")

    check_schema(SCHEMA_JUDGEMENTS, judgements.schema)
    check_schema(SCHEMA_CLUSTER_EXPANSION, expansion.schema)

    return pl.from_arrow(judgements), pl.from_arrow(expansion)
