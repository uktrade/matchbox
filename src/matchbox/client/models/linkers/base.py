"""Base class for linkers."""

import warnings
from abc import ABC, abstractmethod

from pandas import DataFrame
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class LinkerSettings(BaseModel):
    """A data class to enforce basic settings dictionary shapes"""

    left_id: str = Field(description="The unique ID column in the left dataset")
    right_id: str = Field(description="The unique ID column in the right dataset")

    @field_validator("left_id", "right_id")
    @classmethod
    def _id_for_cmf(cls, v: str, info: ValidationInfo) -> str:
        enforce = "id"
        if v != enforce:
            warnings.warn(
                f"For offline deduplication, {info.field_name} can be any field. \n\n"
                "When deduplicating to write back to the Company Matching "
                f"Framework database, the ID must be '{enforce}', generated by "
                "retrieving data with matchbox.query().",
                stacklevel=3,
            )
        return v


class Linker(BaseModel, ABC):
    settings: LinkerSettings

    @classmethod
    @abstractmethod
    def from_settings(cls) -> "Linker":
        raise NotImplementedError(
            """\
            Must implement method to instantiate from settings \
            -- consider creating a pydantic model to enforce shape.
        """
        )

    @abstractmethod
    def prepare(self, left: DataFrame, right: DataFrame) -> None:
        return

    @abstractmethod
    def link(self, left: DataFrame, right: DataFrame) -> DataFrame:
        return
