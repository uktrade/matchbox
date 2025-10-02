"""Collection of client-side functions in aid of model evaluation."""

import warnings
from typing import Any

import polars as pl
from sqlalchemy.exc import OperationalError

from matchbox.client import _handler
from matchbox.client.dags import DAG
from matchbox.client.results import Results
from matchbox.client.sources import Location, Source
from matchbox.common.dtos import ModelResolutionPath, ResolutionPath, ResolutionType
from matchbox.common.eval import (
    ModelComparison,
    PrecisionRecall,
    precision_recall,
)
from matchbox.common.exceptions import MatchboxSourceTableError


def get_samples(
    n: int,
    dag: DAG,
    user_id: int,
    clients: dict[str, Any] | None = None,
    default_client: Any | None = None,
) -> dict[int, pl.DataFrame]:
    """Retrieve samples enriched with source data, grouped by resolution cluster.

    Args:
        n: Number of clusters to sample
        dag: DAG for which to retrieve samples
        user_id: ID of the user requesting the samples
        clients: Dictionary from location names to valid client for each.
            Locations whose name is missing from the dictionary will be skipped,
            unless a default client is provided.
        default_client: Fallback client to use for all sources.

    Returns:
        Dictionary of cluster ID to dataframe describing the cluster

    Raises:
        MatchboxSourceTableError: If a source cannot be queried from a location using
            provided or default clients.
    """
    if not clients:
        clients = {}

    samples: pl.DataFrame = pl.from_arrow(
        _handler.sample_for_eval(
            n=n,
            resolution=ResolutionPath(
                name=dag.final_step, collection=dag.name, version=dag.version
            ),
            user_id=user_id,
        )
    )

    if not len(samples):
        return {}

    results_by_source = []
    for source_resolution in samples["source"].unique():
        resolution = _handler.get_resolution(
            path=ResolutionPath(
                name=source_resolution, collection=dag.name, version=dag.version
            ),
            validate_type=ResolutionType.SOURCE,
        )
        location_name = resolution.config.location_config.name

        if location_name in clients:
            client = clients[location_name]
        elif default_client:
            client = default_client
        else:
            warnings.warn(
                f"Skipping {source_resolution}, incompatible with given client.",
                UserWarning,
                stacklevel=2,
            )
            continue

        location = Location.from_config(
            resolution.config.location_config, client=client
        )
        source = Source.from_resolution(
            resolution=resolution,
            resolution_name=source_resolution,
            dag=dag,
            location=location,
        )

        samples_by_source = samples.filter(pl.col("source") == source_resolution)
        keys_by_source = samples_by_source["key"].to_list()

        try:
            source_data = pl.concat(
                source.fetch(batch_size=10_000, qualify_names=True, keys=keys_by_source)
            )
        except OperationalError as e:
            raise MatchboxSourceTableError(
                "Could not find source using given client."
            ) from e

        samples_and_source = samples_by_source.join(
            source_data, left_on="key", right_on=source.qualified_key
        )
        desired_columns = ["root", "leaf", "key"] + source.qualified_index_fields
        results_by_source.append(samples_and_source[desired_columns])

    if not results_by_source:
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    results_by_root = {
        root: all_results.filter(pl.col("root") == root).drop("root")
        for root in all_results["root"].unique()
    }

    return results_by_root


class EvalData:
    """Object which caches evaluation data to measure performance of models."""

    def __init__(self):
        """Initialise evaluation data from resolution name."""
        self.judgements, self.expansion = _handler.download_eval_data()

    def precision_recall(self, results: Results, threshold: float) -> PrecisionRecall:
        """Computes precision and recall at one threshold."""
        if not len(results.clusters):
            raise ValueError("No clusters suggested by these results.")

        threshold = int(threshold * 100)

        root_leaf = results.root_leaf().rename({"root_id": "root", "leaf_id": "leaf"})
        return precision_recall([root_leaf], self.judgements, self.expansion)[0]


def compare_models(resolutions: list[ModelResolutionPath]) -> ModelComparison:
    """Compare metrics of models based on evaluation data.

    Args:
        resolutions: List of names of model resolutions to be compared.

    Returns:
        A model comparison object, listing metrics for each model.
    """
    return _handler.compare_models(resolutions)
