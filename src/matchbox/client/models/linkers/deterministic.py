"""A linking methodology based on a deterministic set of conditions."""

import polars as pl
from pydantic import Field, field_validator

from matchbox.client.models.linkers.base import Linker, LinkerSettings


class DeterministicSettings(LinkerSettings):
    """A data class to enforce the Deterministic linker's settings dictionary shape."""

    comparisons: list[str] = Field(
        description="""
            A list of valid ON clause to compare fields between the left and 
            the right data.

            Use left.field and right.field to refer to columns in the respective 
            sources.

            Each comparison will be treated as OR logic, but more efficiently than using
            an OR condition in the SQL WHERE clause.

            For example:

            [   
                "left.company_number = right.company_number",
                "left.name = right.name and left.postcode = right.postcode",
            ]
        """,
    )

    @field_validator("comparisons", mode="before")
    @classmethod
    def validate_comparison(cls, v: str | list[str]) -> list[str]:
        """Turn single string into list of one string."""
        if isinstance(v, str):
            return [v]
        return v


class DeterministicLinker(Linker):
    """A deterministic linker that links based on a set of boolean conditions."""

    settings: DeterministicSettings

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Prepare the linker for linking."""
        pass

    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """Link the left and right dataframes."""
        left_pl = left.lazy()  # noqa: F841
        right_pl = right.lazy()  # noqa: F841

        subqueries = []
        for i, condition in enumerate(self.settings.comparisons):
            subquery = f"""
                SELECT
                    l_{i}.{self.settings.left_id} AS left_id,
                    r_{i}.{self.settings.right_id} AS right_id,
                    1.0 AS probability
                FROM left_pl l_{i}
                INNER JOIN right_pl r_{i}
                    ON {condition}
            """
            subqueries.append(subquery)

        union_query = " UNION ALL ".join(subqueries)

        final_query = f"""
            SELECT DISTINCT 
                final.left_id, 
                final.right_id, 
                final.probability
            FROM ({union_query}) as final
        """

        return (
            pl.sql(final_query)
            .with_columns(
                [
                    pl.col("probability").cast(pl.Float32),
                ]
            )
            .collect()
        )
