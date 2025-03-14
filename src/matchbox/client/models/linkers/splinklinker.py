"""A linking methodology leveraging Splink."""

import ast
import inspect
from typing import Any, Type

import pyarrow as pa
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from splink import DuckDBAPI, SettingsCreator
from splink import Linker as SplinkLibLinkerClass
from splink.internals.linker_components.training import LinkerTraining

from matchbox.client.models.linkers.base import Linker, LinkerSettings
from matchbox.common.logging import logger


class SplinkLinkerFunction(BaseModel):
    """A method of splink.Linker.training used to train the linker."""

    function: str
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_function_and_arguments(self) -> "SplinkLinkerFunction":
        """Ensure the function and arguments are valid."""
        if not hasattr(LinkerTraining, self.function):
            raise ValueError(
                f"Function {self.function} not found as method of Splink Linker class"
            )

        splink_linker_func = getattr(LinkerTraining, self.function)
        splink_linker_func_param_set = set(
            inspect.signature(splink_linker_func).parameters.keys()
        )
        current_func_param_set = set(self.arguments.keys())

        if not current_func_param_set <= splink_linker_func_param_set:
            raise ValueError(
                f"Function {self.function} given incorrect arguments: "
                f"{current_func_param_set.difference(splink_linker_func_param_set)}. "
                "Consider referring back to the Splink documentation: "
                "https://moj-analytical-services.github.io/splink/linker.html"
            )

        return self


class SplinkSettings(LinkerSettings):
    """A data class to enforce the Splink linker's settings dictionary shape."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    database_api: Type[DuckDBAPI] = Field(
        default=DuckDBAPI,
        description="""
            The Splink DB API, to choose between DuckDB (default) and Spark (untested)
        """,
    )

    linker_training_functions: list[SplinkLinkerFunction] = Field(
        description="""
            A list of dictionaries where keys are the names of methods for
            splink.Linker.training and values are dictionaries encoding the arguments of
            those methods. Each function will be run in the order supplied.

            Example:
            
                >>> linker_training_functions=[
                ...     {
                ...         "function": "estimate_probability_two_random_records_match",
                ...         "arguments": {
                ...             "deterministic_matching_rules": \"""
                ...                 l.company_name = r.company_name
                ...             \""",
                ...             "recall": 0.7,
                ...         },
                ...     },
                ...     {
                ...         "function": "estimate_u_using_random_sampling",
                ...         "arguments": {"max_pairs": 1e6},
                ...     }
                ... ]
            
        """
    )
    linker_settings: SettingsCreator = Field(
        description="""
            A valid Splink SettingsCreator.

            See Splink's documentation for a full description of available settings.
            https://moj-analytical-services.github.io/splink/api_docs/settings_dict_guide.html

            * link_type must be set to "link_only"
            * unique_id_column_name is overridden to the value of left_id and right_id,
                which must match

            Example:

                >>> from splink import SettingsCreator, block_on
                ... import splink.comparison_library as cl
                ... import splink.comparison_template_library as ctl
                ... 
                ... splink_settings = SettingsCreator(
                ...     retain_matching_columns=False,
                ...     retain_intermediate_calculation_columns=False,
                ...     blocking_rules_to_generate_predictions=[
                ...         block_on("company_name"),
                ...         block_on("postcode"),
                ...     ],
                ...     comparisons=[
                ...         cl.jaro_winkler_at_thresholds(
                ...             "company_name", 
                ...             [0.9, 0.6], 
                ...             term_frequency_adjustments=True
                ...         ),
                ...         ctl.postcode_comparison("postcode"), 
                ...     ]
                ... )         
        """
    )
    threshold: float | None = Field(
        default=None,
        description="""
            The probability above which matches will be kept.

            None is used to indicate no threshold.
            
            Inclusive, so a value of 1 will keep only exact matches across all 
            comparisons.
        """,
        gt=0,
        le=1,
    )

    @model_validator(mode="after")
    def check_ids_match(self) -> "SplinkSettings":
        """Ensure left_id and right_id match."""
        l_id = self.left_id
        r_id = self.right_id
        if l_id is not None and r_id is not None and l_id != r_id:
            raise ValueError(
                "Left and right ID do not match. "
                "left_id and right_id must match in a Splink linker."
            )
        return self

    @model_validator(mode="after")
    def check_link_only(self) -> "SplinkSettings":
        """Ensure link_type is set to "link_only"."""
        if self.linker_settings.link_type != "link_only":
            raise ValueError('link_type must be set to "link_only"')
        return self

    @model_validator(mode="after")
    def add_enforced_settings(self) -> "SplinkSettings":
        """Ensure ID is the only field we link on."""
        self.linker_settings.unique_id_column_name = self.left_id
        return self


class SplinkLinker(Linker):
    """A linker that leverages Bayesian record linkage using Splink."""

    settings: SplinkSettings

    _linker: SplinkLibLinkerClass = None
    _id_dtype_l: Type = None
    _id_dtype_r: Type = None

    @classmethod
    def from_settings(
        cls,
        left_id: str,
        right_id: str,
        linker_training_functions: list[dict[str, Any]],
        linker_settings: SettingsCreator,
        threshold: float,
    ) -> "SplinkLinker":
        """Create a SplinkLinker from a settings dictionary."""
        settings = SplinkSettings(
            left_id=left_id,
            right_id=right_id,
            linker_training_functions=[
                SplinkLinkerFunction(**func) for func in linker_training_functions
            ],
            linker_settings=linker_settings,
            threshold=threshold,
        )
        return cls(settings=settings)

    def prepare(self, left: DataFrame, right: DataFrame) -> None:
        """Prepare the linker for linking."""
        if (set(left.columns) != set(right.columns)) or not left.dtypes.equals(
            right.dtypes
        ):
            raise ValueError(
                "SplinkLinker requires input datasets to be conformant, meaning they "
                "share the same column names and data formats."
            )

        self._id_dtype_l = type(left[self.settings.left_id][0])
        self._id_dtype_r = type(right[self.settings.right_id][0])

        # Deal with converting back to bytes from string b-representation,
        # the most common datatype we expect
        if self._id_dtype_l.__name__ == "bytes":
            self._id_dtype_l = ast.literal_eval
        if self._id_dtype_r.__name__ == "bytes":
            self._id_dtype_r = ast.literal_eval

        left[self.settings.left_id] = left[self.settings.left_id].apply(str)
        right[self.settings.right_id] = right[self.settings.right_id].apply(str)

        self._linker = SplinkLibLinkerClass(
            input_table_or_tables=[left, right],
            input_table_aliases=["l", "r"],
            settings=self.settings.linker_settings,
            db_api=self.settings.database_api(),
        )

        for func in self.settings.linker_training_functions:
            proc_func = getattr(self._linker.training, func.function)
            proc_func(**func.arguments)

    def link(self, left: DataFrame = None, right: DataFrame = None) -> DataFrame:
        """Link the left and right dataframes."""
        if left is not None or right is not None:
            logger.warning(
                "Left and right data are declared in .prepare() for SplinkLinker. "
                "These values will be ignored"
            )

        res = self._linker.inference.predict(
            threshold_match_probability=self.settings.threshold
        )

        df = res.as_pandas_dataframe().drop_duplicates()

        return pa.table(
            [
                pa.array(
                    df[f"{self.settings.left_id}_l"].apply(self._id_dtype_l),
                    type=pa.uint64(),
                ),
                pa.array(
                    df[f"{self.settings.right_id}_r"].apply(self._id_dtype_r),
                    type=pa.uint64(),
                ),
                pa.array(df["match_probability"], type=pa.float32()),
            ],
            names=["left_id", "right_id", "probability"],
        )
