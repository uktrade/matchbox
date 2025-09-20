"""A linking methodology that applies different weights to field comparisons."""

import duckdb
import polars as pl
from pydantic import BaseModel, Field, field_validator

from matchbox.client.models import comparison
from matchbox.client.models.linkers.base import Linker, LinkerSettings


class WeightedComparison(BaseModel):
    """A valid comparison and a weight to give it."""

    comparison: str = Field(
        description="""
            A valid ON clause to compare fields between the left and 
            the right data.

            Use left.field and right.field to refer to fields in the 
            respective sources.

            For example:

            "left.company_name = right.company_name"
        """
    )
    weight: float = Field(
        description="""
            A weight to give this comparison. Use 1 for all comparisons to give
            uniform weight to each.
        """
    )

    @field_validator("comparison")
    @classmethod
    def validate_comparison(cls, v: str) -> str:
        """Validate the comparison string."""
        comp_val = comparison(v)
        return comp_val


class WeightedDeterministicSettings(LinkerSettings):
    """A data class to enforce the Weighted linker's settings dictionary shape.

    Example:
        >>> {
        ...     left_id: "hash",
        ...     right_id: "hash",
        ...     weighted_comparisons: [
        ...         ("l.company_name = r.company_name", 0.7),
        ...         ("l.postcode = r.postcode", 0.7),
        ...         ("l.company_id = r.company_id", 1),
        ...     ],
        ...     threshold: 0.8,
        ... }
    """

    weighted_comparisons: list[WeightedComparison] = Field(
        description="A list of tuples in the form of a comparison, and a weight."
    )
    threshold: float = Field(
        description="""
            The probability above which matches will be kept. 
            
            Inclusive, so a value of 1 will keep only exact matches across all 
            comparisons.
        """,
        ge=0,
        le=1,
    )


class WeightedDeterministicLinker(Linker):
    """A deterministic linker that applies different weights to field comparisons."""

    settings: WeightedDeterministicSettings

    _id_dtype_l: pl.DataType
    _id_dtype_r: pl.DataType

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Prepare the linker for linking."""
        pass

    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """Link the left and right dataframes."""
        self._id_dtype_l = left[self.settings.left_id].dtype
        self._id_dtype_r = right[self.settings.right_id].dtype

        # Used below but ruff can't detect
        left_df = left.clone()  # noqa: F841
        right_df = right.clone()  # noqa: F841

        match_subquery = []
        weights = []

        for weighted_comparison in self.settings.weighted_comparisons:
            match_subquery.append(
                f"""
                    select distinct on (list_sort([raw.left_id, raw.right_id]))
                        raw.left_id,
                        raw.right_id,
                        1.0 * {weighted_comparison.weight} as probability
                    from (
                        select
                            l.{self.settings.left_id} as left_id,
                            r.{self.settings.right_id} as right_id,
                        from
                            left_df l
                        inner join right_df r on
                            {weighted_comparison.comparison}
                    ) raw
                """
            )
            weights.append(weighted_comparison.weight)

        match_subquery = " union all ".join(match_subquery)
        total_weight = sum(weights)

        sql = f"""
            select
                matches.left_id,
                matches.right_id,
                sum(matches.probability) / {total_weight} as probability
            from
                ({match_subquery}) matches
            group by
                matches.left_id,
                matches.right_id
            having
                sum(matches.probability) / 
                    {total_weight} >= {self.settings.threshold};
        """

        return (
            duckdb.sql(sql)
            .pl()
            .with_columns(
                [
                    pl.col("left_id").cast(self._id_dtype_l),
                    pl.col("right_id").cast(self._id_dtype_r),
                    pl.col("probability").cast(pl.Float32),
                ]
            )
        )
