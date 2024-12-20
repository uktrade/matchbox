import logging
from enum import StrEnum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Hashable, ParamSpec, TypeVar

import pyarrow as pa
import pyarrow.compute as pc
from dotenv import find_dotenv, load_dotenv
from pandas import ArrowDtype, DataFrame
from pydantic import BaseModel, ConfigDict, field_validator

from matchbox.common.transform import to_clusters
from matchbox.server.base import MatchboxDBAdapter, inject_backend

if TYPE_CHECKING:
    from matchbox.client.models.models import Model, ModelMetadata
else:
    Model = Any

T = TypeVar("T", bound=Hashable)
P = ParamSpec("P")
R = TypeVar("R")

logic_logger = logging.getLogger("mb_logic")

dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)


class ModelType(StrEnum):
    """Enumeration of supported model types."""

    LINKER = "linker"
    DEDUPER = "deduper"


class ModelMetadata(BaseModel):
    """Metadata for a model."""

    name: str
    description: str
    type: ModelType
    left_source: str
    right_source: str | None = None  # Only used for linker models


def calculate_clusters(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to calculate clusters if it hasn't been already."""

    @wraps(func)
    def wrapper(self: "Results", *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.clusters:
            self.clusters = to_clusters(self.probabilities)
        return func(self, *args, **kwargs)

    return wrapper


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
    clusters: pa.Table | None = None
    model: Model | None = None
    metadata: ModelMetadata

    @field_validator("probabilities", mode="before")
    @classmethod
    def check_probabilities(cls, value: pa.Table | DataFrame) -> pa.Table:
        """Verifies the probabilities table contains the expected fields."""
        if isinstance(value, DataFrame):
            value = pa.Table.from_pandas(value)

        if not isinstance(value, pa.Table):
            raise ValueError("Expected a pandas DataFrame or pyarrow Table.")

        table_fields = set(value.column_names)
        expected_fields = {"left_id", "right_id", "probability"}

        if table_fields != expected_fields:
            raise ValueError(f"Expected {expected_fields}. \n" f"Found {table_fields}.")

        def _check_range(arr: pa.Array, min: float, max: float) -> bool:
            """Helper to check if an array falls in a range."""
            return pa.compute.all(
                pa.compute.and_(
                    pa.compute.greater_equal(arr, min), pa.compute.less_equal(arr, max)
                )
            ).as_py()

        if pa.types.is_floating(value["probability"].type):
            if _check_range(value["probability"], 0, 1):
                mult = 100
            elif _check_range(value["probability"], 0, 100):
                mult = 0
            else:
                p_max = pc.max(value["probability"])
                p_min = pc.min(value["probability"])
                raise ValueError(f"Probability range misconfigured: [{p_min}, {p_max}]")

            probability_uint8 = pc.cast(
                pc.multiply(value["probability"], mult),
                options=pc.CastOptions(
                    target_type=pa.uint8(), allow_float_truncate=True
                ),
            )
            value.set_column(
                i=value.schema.get_field_index("probability"),
                field_="probability",
                column=probability_uint8,
            )

        return value.cast(
            pa.schema(
                [
                    ("left_id", pa.uint64()),
                    ("right_id", pa.uint64()),
                    ("probability", pa.uint8()),
                ]
            )
        )

    def _merge_with_source_data(
        self,
        base_df: DataFrame,
        base_df_cols: list[str],
        left_data: DataFrame,
        left_key: str,
        right_data: DataFrame,
        right_key: str,
        left_merge_col: str,
        right_merge_col: str,
    ) -> DataFrame:
        """Helper method to merge results with source data frames."""
        return (
            base_df.filter(base_df_cols)
            .merge(
                left_data,
                how="left",
                left_on=left_merge_col,
                right_on=left_key,
            )
            .drop(columns=[left_key])
            .merge(
                right_data,
                how="left",
                left_on=right_merge_col,
                right_on=right_key,
            )
            .drop(columns=[right_key])
        )

    def probabilities_to_pandas(self) -> DataFrame:
        """Returns the probability results as a DataFrame."""
        df = (
            self.probabilities.to_pandas(types_mapper=ArrowDtype)
            .assign(
                left=self.model.metadata.left_source,
                right=self.model.metadata.right_source,
                model=self.metadata.name,
            )
            .convert_dtypes(dtype_backend="pyarrow")[
                ["model", "left", "left_id", "right", "right_id", "probability"]
            ]
        )

        return df

    def inspect_probabilities(
        self, left_data: DataFrame, left_key: str, right_data: DataFrame, right_key: str
    ) -> DataFrame:
        """Enriches the probability results with the source data."""
        return self._merge_with_source_data(
            base_df=self.probabilities_to_pandas(),
            base_df_cols=["left_id", "right_id", "probability"],
            left_data=left_data,
            left_key=left_key,
            right_data=right_data,
            right_key=right_key,
            left_merge_col="left_id",
            right_merge_col="left_id",
        )

    @calculate_clusters
    def clusters_to_pandas(self) -> DataFrame:
        """Returns the cluster results as a DataFrame."""
        return self.clusters.to_pandas(types_mapper=ArrowDtype)

    @calculate_clusters
    def inspect_clusters(
        self,
        left_data: DataFrame,
        left_key: str,
        right_data: DataFrame,
        right_key: str,
    ) -> DataFrame:
        """Enriches the cluster results with the source data."""
        return self._merge_with_source_data(
            base_df=self.clusters_to_pandas(),
            base_df_cols=["parent", "child", "probability"],
            left_data=left_data,
            left_key=left_key,
            right_data=right_data,
            right_key=right_key,
            left_merge_col="child",
            right_merge_col="child",
        )

    @inject_backend
    def to_matchbox(self, backend: MatchboxDBAdapter) -> None:
        """Writes the results to the Matchbox database."""
        self.model.insert_model()
        self.model.results = self
