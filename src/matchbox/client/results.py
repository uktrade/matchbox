"""Objects representing the results of running a model client-side."""

from collections.abc import Hashable
from typing import ParamSpec, TypeVar

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
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
    * (Optional) The input data to the model that generated the probabilities


    Allows users to easily interrogate the outputs of models, explore decisions on
    choosing thresholds for clustering, and upload the results to Matchbox.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    probabilities: pa.Table
    _clusters: pa.Table | None = None

    def __init__(
        self,
        probabilities: pa.Table | pl.DataFrame,
        left_root_leaf: pa.Table | None = None,
        right_root_leaf: pa.Table | None = None,
    ) -> None:
        """Initialises and validates results."""
        self.left_root_leaf = None
        self.right_root_leaf = None

        if left_root_leaf is not None:
            self.left_root_leaf = left_root_leaf
        if right_root_leaf is not None:
            self.right_root_leaf = right_root_leaf

        if isinstance(probabilities, pl.DataFrame):
            probabilities = probabilities.to_arrow()

        if not isinstance(probabilities, pa.Table):
            raise ValueError("Expected a polars DataFrame or pyarrow Table.")

        expected_fields = set(SCHEMA_RESULTS.names)
        if set(probabilities.column_names) != expected_fields:
            raise ValueError(
                f"Expected {expected_fields}.\nFound {set(probabilities.column_names)}."
            )

        # Handle empty tables
        if probabilities.num_rows == 0:
            empty_arrays = [pa.array([], type=field.type) for field in SCHEMA_RESULTS]
            probabilities = pa.Table.from_arrays(
                empty_arrays, names=[field.name for field in SCHEMA_RESULTS]
            )

        # Process probability field if it contains floating-point or decimal values
        probability_type = probabilities["probability"].type
        if pa.types.is_floating(probability_type) or pa.types.is_decimal(
            probability_type
        ):
            probability_uint8 = pc.cast(
                pc.round(pc.multiply(probabilities["probability"], 100)),
                options=pc.CastOptions(
                    target_type=pa.uint8(),
                    allow_float_truncate=True,
                    allow_decimal_truncate=True,
                ),
            )

            # Check max value only if the table is not empty
            max_prob = pc.max(probability_uint8)
            if max_prob is not None and max_prob.as_py() > 100:
                p_max = pc.max(probabilities["probability"]).as_py()
                p_min = pc.min(probabilities["probability"]).as_py()
                raise ValueError(f"Probability range misconfigured: [{p_min}, {p_max}]")

            probabilities = probabilities.set_column(
                i=probabilities.schema.get_field_index("probability"),
                field_="probability",
                column=probability_uint8,
            )

        self.probabilities = probabilities.cast(SCHEMA_RESULTS)

    @property
    def clusters(self):
        """Retrieve new clusters implied by these results."""
        if not self._clusters:
            im = IntMap()
            self._clusters = to_clusters(
                results=self.probabilities, dtype=pa.int64, hash_func=im.index
            )
        return self._clusters

    def _merge_with_source_data(
        self,
        base_df: pl.DataFrame,
        base_df_cols: list[str],
        left_data: pl.DataFrame,
        left_key: str,
        right_data: pl.DataFrame,
        right_key: str,
        left_merge_col: str,
        right_merge_col: str,
    ) -> pl.DataFrame:
        """Helper method to merge results with source data frames."""
        return (
            base_df.select(base_df_cols)
            .join(
                left_data,
                how="left",
                left_on=left_merge_col,
                right_on=left_key,
            )
            .join(
                right_data,
                how="left",
                left_on=right_merge_col,
                right_on=right_key,
            )
        )

    def inspect_probabilities(
        self,
        left_data: pl.DataFrame,
        left_key: str,
        right_data: pl.DataFrame,
        right_key: str,
    ) -> pl.DataFrame:
        """Enriches the probability results with the source data."""
        return self._merge_with_source_data(
            base_df=pl.from_arrow(self.probabilities),
            base_df_cols=["left_id", "right_id", "probability"],
            left_data=left_data,
            left_key=left_key,
            right_data=right_data,
            right_key=right_key,
            left_merge_col="left_id",
            right_merge_col="right_id",
        )

    def inspect_clusters(
        self,
        left_data: pl.DataFrame,
        left_key: str,
        right_data: pl.DataFrame,
        right_key: str,
    ) -> pl.DataFrame:
        """Enriches the cluster results with the source data."""
        return self._merge_with_source_data(
            base_df=pl.from_arrow(self.clusters),
            base_df_cols=["parent", "child", "probability"],
            left_data=left_data,
            left_key=left_key,
            right_data=right_data,
            right_key=right_key,
            left_merge_col="child",
            right_merge_col="child",
        )

    def root_leaf(self):
        """Returns all roots and leaves implied by these results."""
        if self.left_root_leaf is None:
            raise RuntimeError(
                "This Results object wasn't instantiated for validation features."
            )

        parents_root_leaf = pl.from_arrow(self.left_root_leaf.select(["id", "leaf_id"]))
        if self.right_root_leaf is not None:
            parents_root_leaf = pl.concat(
                [
                    parents_root_leaf,
                    pl.from_arrow(self.right_root_leaf.select(["id", "leaf_id"])),
                ]
            )

        # Go from parent-child (where child could be the root of another model)
        # to root-leaf, where leaf is a source cluster ID
        root_leaf_res = (
            pl.from_arrow(self.clusters)
            .rename({"parent": "root_id"})
            .join(parents_root_leaf, left_on="child", right_on="id")
            .select(["root_id", "leaf_id"])
            .unique()
        )

        # Generate root-leaf for those input rows that weren't merged by this model
        unmerged_ids_rows = (
            parents_root_leaf.select("id", "leaf_id")
            .join(
                pl.from_arrow(self.clusters).select("child"),
                left_on="id",
                right_on="child",
                how="anti",
            )
            .rename({"id": "root_id"})
            .select(["root_id", "leaf_id"])
            .unique()
        )

        return pl.concat([root_leaf_res, unmerged_ids_rows])
