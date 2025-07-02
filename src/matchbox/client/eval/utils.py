"""Collection of client-side functions in aid of model evaluation."""

import polars as pl
from matplotlib import pyplot as plt

from matchbox.client import _handler
from matchbox.client.results import Results
from matchbox.common.dtos import ModelResolutionName
from matchbox.common.eval import (
    ModelComparison,
    PrecisionRecall,
    contains_to_pairs,
    precision_recall,
)


class EvalData:
    """Object to cache evaluation data to measure performance of models."""

    def __init__(self):
        """Initialise evaluation data from resolution name."""
        self.eval_data = _handler.download_eval_data()
        self.eval_pairs = contains_to_pairs(pl.from_arrow(self.eval_data))

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
        return precision_recall(model_pairs, self.eval_pairs)

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
