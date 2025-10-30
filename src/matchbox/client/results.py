"""Objects representing the results of running a model client-side."""

from collections.abc import Hashable
from typing import ParamSpec, TypeVar

import polars as pl
from pydantic import ConfigDict

from matchbox.common.arrow import SCHEMA_RESULTS
from matchbox.common.hash import IntMap
from matchbox.common.transform import to_clusters

T = TypeVar("T", bound=Hashable)
P = ParamSpec("P")
R = TypeVar("R")


class Results:
    """Results of a model run.

    Contains:

    * The probabilities of each pair being a match
    * (Optional) The clusters of connected components at each threshold
    * (Optional) The leaf_id mapping to trace results back to source clusters

    Allows users to easily interrogate the outputs of models, explore decisions on
    choosing thresholds for clustering, and upload the results to Matchbox.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    probabilities: pl.DataFrame
    _clusters: pl.DataFrame | None = None

    def __init__(
        self,
        probabilities: pl.DataFrame,
        left_root_leaf: pl.DataFrame | None = None,
        right_root_leaf: pl.DataFrame | None = None,
    ) -> None:
        """Initialises and validates results."""
        self.left_root_leaf = None
        self.right_root_leaf = None

        if left_root_leaf is not None:
            self.left_root_leaf = left_root_leaf
        if right_root_leaf is not None:
            self.right_root_leaf = right_root_leaf

        if not isinstance(probabilities, pl.DataFrame):
            raise ValueError(f"Expected a polars DataFrame, got {type(probabilities)}.")

        expected_fields = set(SCHEMA_RESULTS.names)
        if set(probabilities.columns) != expected_fields:
            raise ValueError(
                f"Expected {expected_fields}.\nFound {set(probabilities.column_names)}."
            )

        # Handle empty tables
        if probabilities.height == 0:
            probabilities = pl.DataFrame(schema=pl.Schema(SCHEMA_RESULTS))

        # Process probability field if it contains floating-point or decimal values
        probability_type = probabilities["probability"].dtype
        if probability_type.is_float() or probability_type.is_decimal():
            probability_uint8 = pl.Series(
                probabilities.select(
                    pl.col("probability").mul(100).round(0).cast(pl.UInt8)
                )
            )

            # Check max value only if the table is not empty
            max_prob = probability_uint8.max()
            if max_prob is not None and max_prob > 100:
                p_max = max_prob
                p_min = probability_uint8.min()
                raise ValueError(f"Probability range misconfigured: [{p_min}, {p_max}]")

            probabilities = probabilities.replace_column(
                probabilities.get_column_index("probability"), probability_uint8
            )

        # need schema in format recognised by polars
        self.probabilities = probabilities.cast(pl.Schema(SCHEMA_RESULTS))

    @property
    def clusters(self) -> pl.DataFrame:
        """Retrieve new clusters implied by these results."""
        if self._clusters is None:
            im = IntMap()
            self._clusters = to_clusters(
                results=self.probabilities, dtype=pl.Int64, hash_func=im.index
            )
        return self._clusters

    def root_leaf(self) -> pl.DataFrame:
        """Returns all roots and leaves implied by these results."""
        if self.left_root_leaf is None:
            raise RuntimeError(
                "This Results object wasn't instantiated for validation features."
            )

        parents_root_leaf = self.left_root_leaf.select(["id", "leaf_id"])
        if self.right_root_leaf is not None:
            parents_root_leaf = pl.concat(
                [
                    parents_root_leaf,
                    self.right_root_leaf.select(["id", "leaf_id"]),
                ]
            )

        # Go from parent-child (where child could be the root of another model)
        # to root-leaf, where leaf is a source cluster ID
        root_leaf_res = (
            self.clusters.rename({"parent": "root_id"})
            .join(parents_root_leaf, left_on="child", right_on="id")
            .select(["root_id", "leaf_id"])
            .unique()
        )

        # Generate root-leaf for those input rows that weren't merged by this model
        unmerged_ids_rows = (
            parents_root_leaf.select("id", "leaf_id")
            .join(
                self.clusters.select("child"),
                left_on="id",
                right_on="child",
                how="anti",
            )
            .rename({"id": "root_id"})
            .select(["root_id", "leaf_id"])
            .unique()
        )

        return pl.concat([root_leaf_res, unmerged_ids_rows])
