"""Objects representing the results of running a model client-side."""

from typing import TYPE_CHECKING, Any, Hashable, ParamSpec, TypeVar

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, field_validator

from matchbox.common.arrow import SCHEMA_RESULTS
from matchbox.common.dtos import ModelConfig
from matchbox.common.hash import IntMap
from matchbox.common.transform import to_clusters

if TYPE_CHECKING:
    from matchbox.client.models.models import Model
else:
    Model = Any

T = TypeVar("T", bound=Hashable)
P = ParamSpec("P")
R = TypeVar("R")


class Results(BaseModel):
    """Results of a model run.

    Contains:

    * The probabilities of each pair being a match
    * (Optional) The clusters of connected components at each threshold

    Model is required during construction and calculation, but not when loading
    from storage.

    Allows users to easily interrogate the outputs of models, explore decisions on
    choosing thresholds for clustering, and upload the results to Matchbox.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    probabilities: pa.Table
    _clusters: pa.Table | None = None
    model: Model | None = None
    metadata: ModelConfig

    @field_validator("probabilities", mode="before")
    @classmethod
    def check_probabilities(cls, value: pa.Table | pl.DataFrame) -> pa.Table:
        """Verifies the probabilities table contains the expected fields."""
        if isinstance(value, pl.DataFrame):
            value = value.to_arrow()

        if not isinstance(value, pa.Table):
            raise ValueError("Expected a polars DataFrame or pyarrow Table.")

        expected_fields = set(SCHEMA_RESULTS.names)
        if set(value.column_names) != expected_fields:
            raise ValueError(
                f"Expected {expected_fields}. \nFound {set(value.column_names)}."
            )

        # Handle empty tables
        if value.num_rows == 0:
            empty_arrays = [pa.array([], type=field.type) for field in SCHEMA_RESULTS]
            return pa.Table.from_arrays(
                empty_arrays, names=[field.name for field in SCHEMA_RESULTS]
            )

        # Process probability field if it contains floating-point or decimal values
        probability_type = value["probability"].type
        if pa.types.is_floating(probability_type) or pa.types.is_decimal(
            probability_type
        ):
            probability_uint8 = pc.cast(
                pc.round(pc.multiply(value["probability"], 100)),
                options=pc.CastOptions(
                    target_type=pa.uint8(),
                    allow_float_truncate=True,
                    allow_decimal_truncate=True,
                ),
            )

            # Check max value only if the table is not empty
            max_prob = pc.max(probability_uint8)
            if max_prob is not None and max_prob.as_py() > 100:
                p_max = pc.max(value["probability"]).as_py()
                p_min = pc.min(value["probability"]).as_py()
                raise ValueError(f"Probability range misconfigured: [{p_min}, {p_max}]")

            value = value.set_column(
                i=value.schema.get_field_index("probability"),
                field_="probability",
                column=probability_uint8,
            )

        return value.cast(SCHEMA_RESULTS)

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
            .drop(left_key)
            .join(
                right_data,
                how="left",
                left_on=right_merge_col,
                right_on=right_key,
            )
            .drop(right_key)
        )

    def probabilities_to_polars(self) -> pl.DataFrame:
        """Returns the probability results as a polars DataFrame."""
        df = (
            pl.from_arrow(self.probabilities)
            .with_columns(
                [
                    pl.lit(self.model.model_config.left_resolution).alias("left"),
                    pl.lit(self.model.model_config.right_resolution).alias("right"),
                    pl.lit(self.metadata.name).alias("model"),
                ]
            )
            .select(["model", "left", "left_id", "right", "right_id", "probability"])
        )

        return df

    def inspect_probabilities(
        self,
        left_data: pl.DataFrame,
        left_key: str,
        right_data: pl.DataFrame,
        right_key: str,
    ) -> pl.DataFrame:
        """Enriches the probability results with the source data."""
        return self._merge_with_source_data(
            base_df=self.probabilities_to_polars(),
            base_df_cols=["left_id", "right_id", "probability"],
            left_data=left_data,
            left_key=left_key,
            right_data=right_data,
            right_key=right_key,
            left_merge_col="left_id",
            right_merge_col="right_id",
        )

    def clusters_to_polars(self) -> pl.DataFrame:
        """Returns the cluster results as a polars DataFrame."""
        return pl.from_arrow(self.clusters)

    def inspect_clusters(
        self,
        left_data: pl.DataFrame,
        left_key: str,
        right_data: pl.DataFrame,
        right_key: str,
    ) -> pl.DataFrame:
        """Enriches the cluster results with the source data."""
        return self._merge_with_source_data(
            base_df=self.clusters_to_polars(),
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
        if ("leaf_id" not in self.model.left_data.columns) or (
            self.model.right_data is not None
            and ("leaf_id" not in self.model.right_data.columns)
        ):
            raise RuntimeError(
                "To compute root-leaf, model inputs must contain leaf IDs."
            )

        parents_root_leaf = self.model.left_data.select(["id", "leaf_id"])
        if self.model.right_data is not None:
            parents_root_leaf = pl.concat(
                [parents_root_leaf, self.model.right_data.select(["id", "leaf_id"])]
            )

        # Go from parent-child (where child could be the root of another model)
        # to root-leaf, where leaf is a source cluster ID
        root_leaf_res = (
            self.clusters_to_polars()
            .rename({"parent": "root_id"})
            .join(parents_root_leaf, left_on="child", right_on="id")
            .select(["root_id", "leaf_id"])
            .unique()
        )

        # Generate root-leaf for those input rows that weren't merged by this model
        unmerged_ids_rows = (
            parents_root_leaf.select("id", "leaf_id")
            .join(
                self.clusters_to_polars().select("child"),
                left_on="id",
                right_on="child",
                how="anti",
            )
            .rename({"id": "root_id"})
            .select(["root_id", "leaf_id"])
            .unique()
        )

        return pl.concat([root_leaf_res, unmerged_ids_rows])

    def to_matchbox(self) -> None:
        """Writes the results to the Matchbox database."""
        self.model.insert_model()
        self.model.results = self
