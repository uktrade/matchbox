from typing import Any, Callable

import splink.comparison_library as cl
from pydantic import BaseModel, Field
from splink import SettingsCreator
from splink import blocking_rule_library as brl

from matchbox.models.dedupers import NaiveDeduper
from matchbox.models.dedupers.base import Deduper
from matchbox.models.linkers import (
    DeterministicLinker,
    SplinkLinker,
    WeightedDeterministicLinker,
)
from matchbox.models.linkers.base import Linker


class DedupeTestParams(BaseModel):
    """Data class for raw dataset testing parameters and attributes."""

    source: str = Field(description="Reference for the source table")
    fixture: str = Field(description="pytest fixture of the clean data")
    fields: dict[str, str] = Field(
        description=(
            "Data fields to select and work with. The key is the name of the "
            "field as it comes out of the database, the value is what it should be "
            "renamed to for models that require field names be identical, like the "
            "SplinkLinker."
        )
    )
    unique_n: int = Field(description="Unique items in this data")
    curr_n: int = Field(description="Current row count of this data")
    tgt_prob_n: int = Field(description="Expected count of generated probabilities")
    tgt_clus_n: int = Field(description="Expected count of resolved clusters")


class LinkTestParams(BaseModel):
    """Data class for deduped dataset testing parameters and attributes."""

    source_l: str = Field(description="Reference for the left source model")
    fixture_l: str = Field(description="pytest fixture of the clean left data")
    fields_l: dict[str, str] = Field(
        description=(
            "Left data fields to select and work with. The key is the name of the "
            "field as it comes out of the database, the value is what it should be "
            "renamed to for models that require field names be identical, like the "
            "SplinkLinker."
        )
    )
    curr_n_l: int = Field(description="Current row count of the left data")

    source_r: str = Field(description="Reference for the right source model")
    fixture_r: str = Field(description="pytest fixture of the clean right data")
    fields_r: dict[str, str] = Field(
        description=(
            "Right data fields to select and work with. The key is the name of the "
            "field as it comes out of the database, the value is what it should be "
            "renamed to for models that require field names be identical, like the "
            "SplinkLinker."
        )
    )
    curr_n_r: int = Field(description="Current row count of the right data")

    unique_n: int = Field(description="Unique items in the merged data")
    tgt_prob_n: int = Field(description="Expected count of generated probabilities")
    tgt_clus_n: int = Field(description="Expected count of resolved clusters")


Model = type[Deduper | Linker]
DataSettings = Callable[[DedupeTestParams | LinkTestParams], dict[str, Any]]


class ModelTestParams(BaseModel):
    """Data class for model testing parameters and attributes."""

    name: str = Field(description="Model name")
    cls: Model = Field(description="Model class")
    build_settings: DataSettings = Field(
        description=(
            "A function that takes an object of type DedupeTestParams or "
            "LinkTestParams and returns an appropriate settings dictionary "
            "for this deduper."
        )
    )
    rename_fields: bool = Field(
        description=(
            "Whether fields should be coerced to have matching names, as required "
            "by some Linkers and Dedupers."
        )
    )


dedupe_data_test_params = [
    DedupeTestParams(
        source="test.crn",
        fixture="query_clean_crn",
        fields={
            "test_crn_company_name": "company_name",
            "test_crn_crn": "crn",
        },
        # 1000 unique items repeated three times with minor perturbations
        unique_n=1000,
        curr_n=3000,
        # Unordered pairs of sets of three, so (3 choose 2) = 3, * 1000 = 3000
        tgt_prob_n=3000,
        tgt_clus_n=1000,
    ),
    DedupeTestParams(
        source="test.duns",
        fixture="query_clean_duns",
        fields={
            "test_duns_company_name": "company_name",
            "test_duns_duns": "duns",
        },
        # 500 unique items with no duplication
        unique_n=500,
        curr_n=500,
        # No duplicates
        tgt_prob_n=0,
        tgt_clus_n=0,
    ),
    DedupeTestParams(
        source="test.cdms",
        fixture="query_clean_cdms",
        fields={
            "test_cdms_crn": "crn",
            "test_cdms_cdms": "cdms",
        },
        # 1000 unique items repeated two times, completely identical
        unique_n=1000,
        curr_n=2000,
        # Because the repeated items are identical, they generate the same hash
        # This means they're actually unordered pairs of sets of one
        # So (1 choose 2) = 0, * 1000 = 0
        tgt_prob_n=0,
        tgt_clus_n=0,
    ),
]


link_data_test_params = [
    LinkTestParams(
        # Left
        source_l="naive_test.crn",
        fixture_l="query_clean_crn_deduped",
        fields_l={"test_crn_company_name": "company_name"},
        curr_n_l=3000,
        # Right
        source_r="naive_test.duns",
        fixture_r="query_clean_duns_deduped",
        fields_r={"test_duns_company_name": "company_name"},
        curr_n_r=500,
        # Check
        unique_n=1000,
        # Remember these are deduped: 1000 unique in the left, 500 in the right
        tgt_prob_n=500,
        tgt_clus_n=500,
    ),
    LinkTestParams(
        # Left
        source_l="naive_test.cdms",
        fixture_l="query_clean_cdms_deduped",
        fields_l={
            "test_cdms_crn": "crn",
        },
        curr_n_l=2000,
        # Right
        source_r="naive_test.crn",
        fixture_r="query_clean_crn_deduped",
        fields_r={"test_crn_crn": "crn"},
        curr_n_r=3000,
        # Check
        unique_n=1000,
        # Remember these are deduped: 1000 unique in the left, 1000 in the right
        tgt_prob_n=1000,
        tgt_clus_n=1000,
    ),
]


def make_naive_dd_settings(data: DedupeTestParams) -> dict[str, Any]:
    return {"id": "id", "unique_fields": list(data.fields.keys())}


dedupe_model_test_params = [
    ModelTestParams(
        name="naive",
        cls=NaiveDeduper,
        build_settings=make_naive_dd_settings,
        rename_fields=False,
    )
]


def make_deterministic_li_settings(data: LinkTestParams) -> dict[str, Any]:
    comparisons = []

    for field_l, field_r in zip(
        data.fields_l.keys(), data.fields_r.keys(), strict=False
    ):
        comparisons.append(f"l.{field_l} = r.{field_r}")

    return {
        "left_id": "id",
        "right_id": "id",
        "comparisons": " and ".join(comparisons),
    }


def make_splink_li_settings(data: LinkTestParams) -> dict[str, Any]:
    fields_l = data.fields_l.values()
    fields_r = data.fields_r.values()
    if set(fields_l) != set(fields_r):
        raise ValueError("Splink requires fields have identical names")
    else:
        fields = list(fields_l)

    comparisons = []

    for field in fields:
        comparisons.append(f"l.{field} = r.{field}")

    linker_training_functions = [
        {
            "function": "estimate_probability_two_random_records_match",
            "arguments": {
                "deterministic_matching_rules": comparisons,
                "recall": 1,
            },
        },
        {
            "function": "estimate_u_using_random_sampling",
            "arguments": {"max_pairs": 1e4},
        },
    ]

    # The m parameter is 1 because we're testing in a deterministic system, and
    # many of these tests only have one field, so we can't use expectation
    # maximisation to estimate. For testing raw functionality, fine to use 1
    linker_settings = SettingsCreator(
        link_type="link_only",
        retain_matching_columns=False,
        retain_intermediate_calculation_columns=False,
        blocking_rules_to_generate_predictions=[
            brl.block_on(field) for field in fields
        ],
        comparisons=[
            cl.ExactMatch(field).configure(m_probabilities=[1, 0]) for field in fields
        ],
    )

    return {
        "left_id": "id",
        "right_id": "id",
        "linker_training_functions": linker_training_functions,
        "linker_settings": linker_settings,
        "threshold": None,
    }


def make_weighted_deterministic_li_settings(data: LinkTestParams) -> dict[str, Any]:
    weighted_comparisons = []

    for field_l, field_r in zip(data.fields_l, data.fields_r, strict=False):
        weighted_comparisons.append((f"l.{field_l} = r.{field_r}", 1))

    return {
        "left_id": "id",
        "right_id": "id",
        "weighted_comparisons": weighted_comparisons,
        "threshold": 1,
    }


link_model_test_params = [
    ModelTestParams(
        name="deterministic",
        cls=DeterministicLinker,
        build_settings=make_deterministic_li_settings,
        rename_fields=False,
    ),
    ModelTestParams(
        name="weighted_deterministic",
        cls=WeightedDeterministicLinker,
        build_settings=make_weighted_deterministic_li_settings,
        rename_fields=False,
    ),
    ModelTestParams(
        name="splink",
        cls=SplinkLinker,
        build_settings=make_splink_li_settings,
        rename_fields=True,
    ),
]
