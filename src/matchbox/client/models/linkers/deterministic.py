"""A linking methodology based on a deterministic set of conditions."""

from typing import Type

import duckdb
from pandas import ArrowDtype, DataFrame
from pydantic import Field, field_validator

from matchbox.client.helpers import comparison
from matchbox.client.models.linkers.base import Linker, LinkerSettings


class DeterministicSettings(LinkerSettings):
    """A data class to enforce the Deterministic linker's settings dictionary shape."""

    comparisons: str = Field(
        description="""
            A valid ON clause to compare fields between the left and 
            the right data.

            Use left.field and right.field to refer to columns in the 
            respective sources.

            For example:

            "left.name = right.name and left.company_id = right.id"
        """
    )

    @field_validator("comparisons")
    @classmethod
    def validate_comparison(cls, v: str) -> str:
        """Validate the comparison string."""
        comp_val = comparison(v)
        return comp_val


class DeterministicLinker(Linker):
    """A deterministic linker that links based on a set of boolean conditions."""

    settings: DeterministicSettings

    _id_dtype_l: Type = None
    _id_dtype_r: Type = None

    @classmethod
    def from_settings(
        cls, left_id: str, right_id: str, comparisons: str
    ) -> "DeterministicLinker":
        """Create a DeterministicLinker from a settings dictionary."""
        settings = DeterministicSettings(
            left_id=left_id, right_id=right_id, comparisons=comparisons
        )
        return cls(settings=settings)

    def prepare(self, left: DataFrame, right: DataFrame) -> None:
        """Prepare the linker for linking."""
        pass

    def link(self, left: DataFrame, right: DataFrame) -> DataFrame:
        """Link the left and right dataframes."""
        self._id_dtype_l = type(left[self.settings.left_id][0])
        self._id_dtype_r = type(right[self.settings.right_id][0])

        left_df = left.copy()  # noqa: F841
        right_df = right.copy()  # noqa: F841

        sql = f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1 as probability
            from (
                select
                    l.{self.settings.left_id} as left_id,
                    r.{self.settings.right_id} as right_id,
                from
                    left_df l
                inner join right_df r on
                    {self.settings.comparisons}
            ) raw;
        """
        df_arrow = duckdb.sql(sql).arrow()
        res = df_arrow.to_pandas(
            split_blocks=True, self_destruct=True, types_mapper=ArrowDtype
        )
        del df_arrow

        # Convert bytearray back to bytes
        return res.assign(
            left_id=lambda df: df.left_id.apply(self._id_dtype_l),
            right_id=lambda df: df.right_id.apply(self._id_dtype_r),
        )
