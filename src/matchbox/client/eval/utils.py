"""Collection of client-side functions in aid of model evaluation."""

import warnings
from typing import Any

import polars as pl
from matplotlib import pyplot as plt

from matchbox.client import _handler
from matchbox.client.results import Results
from matchbox.common.dtos import ModelResolutionName
from matchbox.common.eval import (
    ModelComparison,
    PrecisionRecall,
    contains_to_pairs,
    eval_data_to_pairs,
    precision_recall,
)


def get_samples(
    n: int,
    resolution: ModelResolutionName,
    user_id: int,
    credentials: Any,
) -> dict[int, pl.DataFrame]:
    """Retrieve samples enriched with source data, grouped by resolution cluster.

    Args:
        n: Number of clusters to sample
        resolution: Model resolution proposing the clusters
        user_id: ID of the user requesting the samples
        credentials: Valid credentials for the source configs.
            Sources that can't be queried with these credentials will be skipped.

    Returns:
        Dictionary of cluster ID to dataframe describing the cluster
    """
    samples: pl.DataFrame = pl.from_arrow(
        _handler.sample_for_eval(n=n, resolution=resolution, user_id=user_id)
    )

    if not len(samples):
        return {}

    results_by_source = []
    for source_resolution in samples["source"].unique():
        source_config = _handler.get_source_config(source_resolution)
        try:
            source_config.location.add_credentials(credentials=credentials)
        except ValueError:
            warnings.warn(
                f"Skipping {source_resolution}, incompatible with given credentials.",
                UserWarning,
                stacklevel=2,
            )
            continue

        samples_by_source = samples.filter(pl.col("source") == source_resolution)
        keys_by_source = samples_by_source["key"].to_list()

        source_data = pl.concat(
            source_config.query(
                batch_size=10_000, qualify_names=True, keys=keys_by_source
            )
        )
        samples_and_source = samples_by_source.join(
            source_data, left_on="key", right_on=source_config.qualified_key
        )
        desired_columns = ["root", "leaf", "key"] + source_config.qualified_fields
        results_by_source.append(samples_and_source[desired_columns])

    if not len(results_by_source):
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    results_by_root = {
        root: all_results.filter(pl.col("root") == root).drop("root")
        for root in all_results["root"].unique()
    }

    return results_by_root


class EvalData:
    """Object to cache evaluation data to measure performance of models."""

    def __init__(self):
        """Initialise evaluation data from resolution name."""
        self.judgements, self.expansion = _handler.download_eval_data()
        self.pairs = eval_data_to_pairs(self.judgements, self.expansion)

    def precision_recall(self, results: Results, threshold: float) -> PrecisionRecall:
        """Computes precision and recall at one threshold."""
        threshold = int(threshold * 100)
        clusters = (
            pl.from_arrow(results.clusters)
            .rename({"child": "leaf"})
            .filter(pl.col("threshold") >= threshold)
            .select(["parent", "leaf"])
        )
        model_pairs = contains_to_pairs(clusters)
        return precision_recall(model_pairs, self.pairs)

    def pr_curve(self, results: Results) -> dict[str, PrecisionRecall]:
        """For each threshold in retults computes precision and recall."""
        all_p = []
        all_r = []

        probs = pl.from_arrow(results.probabilities)
        thresholds = probs.select("probability").unique().to_series()
        for i, t in enumerate(sorted(thresholds)):
            float_thresh = t / 100
            p, r = self.precision_recall(results=results, threshold=float_thresh)
            all_p.append(p)
            all_r.append(r)
            plt.annotate(float_thresh, (all_r[i], all_p[i]))

        plt.plot(all_r, all_p, marker="o")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title("Precision-Recall Curve")
        plt.grid()

        return plt


def compare_models(resolutions: list[ModelResolutionName]) -> ModelComparison:
    """Compare metrics of models based on evaluation data.

    Args:
        resolutions: List of names of model resolutions to be compared.

    Returns:
        A model comparison object, listing metrics for each model.
    """
    return _handler.compare_models(resolutions)
