"""A deduplication methodology based on a deterministic set of conditions."""

import duckdb
import polars as pl
from pydantic import Field

from matchbox.client.models.dedupers.base import Deduper, DeduperSettings


class NaiveSettings(DeduperSettings):
    """A data class to enforce the Naive deduper's settings dictionary shape."""

    unique_fields: list[str] = Field(
        description="A list of fields that will form a unique, deduplicated record"
    )


class NaiveDeduper(Deduper):
    """A simple deduper that deduplicates based on a set of boolean conditions."""

    settings: NaiveSettings

    _id_dtype: pl.DataType

    def prepare(self, data: pl.DataFrame) -> None:
        """Prepare the deduper for deduplication."""
        pass

    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Deduplicate the dataframe."""
        self._id_dtype = data[self.settings.id].dtype

        df = data.clone()

        join_clause = []
        for field in self.settings.unique_fields:
            join_clause.append(f"l.{field} = r.{field}")
        join_clause_compiled = " and ".join(join_clause)

        # Generate a key to remove row self-match but ALLOW true duplicate
        # rows where all data items are identical in the source
        df = df.with_row_index("_unique_e4003b")

        # We also need to suppress raw.left_id = raw.right_id in cases where
        # we're deduplicating an unnested array of primary keys
        sql = f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1.0 as probability
            from (
                select
                    l.{self.settings.id} as left_id,
                    r.{self.settings.id} as right_id
                from
                    df l
                inner join df r on
                    (
                        {join_clause_compiled}
                    ) and
                        l._unique_e4003b != r._unique_e4003b
            ) raw
                where raw.left_id != raw.right_id;
        """

        return (
            duckdb.sql(sql)
            .pl()
            .with_columns(
                [
                    pl.col("left_id").cast(self._id_dtype),
                    pl.col("right_id").cast(self._id_dtype),
                    pl.col("probability").cast(pl.Float32),
                ]
            )
        )
