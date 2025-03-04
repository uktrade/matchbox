"""A deduplication methodology based on a deterministic set of conditions."""

from typing import Type

import duckdb
from pandas import ArrowDtype, DataFrame
from pydantic import Field

from matchbox.client.models.dedupers.base import Deduper, DeduperSettings


class NaiveSettings(DeduperSettings):
    """A data class to enforce the Naive deduper's settings dictionary shape"""

    unique_fields: list[str] = Field(
        description="A list of columns that will form a unique, deduplicated record"
    )


class NaiveDeduper(Deduper):
    settings: NaiveSettings

    _id_dtype: Type = None

    @classmethod
    def from_settings(cls, id: str, unique_fields: list[str]) -> "NaiveDeduper":
        settings = NaiveSettings(id=id, unique_fields=unique_fields)
        return cls(settings=settings)

    def prepare(self, data: DataFrame) -> None:
        pass

    def dedupe(self, data: DataFrame) -> DataFrame:
        self._id_dtype = type(data[self.settings.id][0])

        df = data.copy()

        join_clause = []
        for field in self.settings.unique_fields:
            join_clause.append(f"l.{field} = r.{field}")
        join_clause_compiled = " and ".join(join_clause)

        # Generate a new PK to remove row self-match but ALLOW true duplicate
        # rows where all data items are identical in the source
        df["_unique_e4003b"] = range(df.shape[0])

        # We also need to suppress raw.left_id = raw.right_id in cases where
        # we're deduplicating an unnested array of primary keys
        sql = f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1 as probability
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
        df_arrow = duckdb.sql(sql).arrow()
        res = df_arrow.to_pandas(
            split_blocks=True, self_destruct=True, types_mapper=ArrowDtype
        )
        del df_arrow

        # Convert bytearray back to bytes
        return res.assign(
            left_id=lambda df: df.left_id.apply(self._id_dtype),
            right_id=lambda df: df.right_id.apply(self._id_dtype),
        )
