import pytest
from pandas import DataFrame

from matchbox import make_model, query
from matchbox.common.db import Source, SourceWarehouse
from matchbox.server.postgresql import MatchboxPostgres

from ..fixtures.db import AddIndexedDataCallable
from ..fixtures.models import (
    DedupeTestParams,
    ModelTestParams,
    dedupe_data_test_params,
    dedupe_model_test_params,
)


@pytest.mark.parametrize("fx_data", dedupe_data_test_params)
@pytest.mark.parametrize("fx_deduper", dedupe_model_test_params)
def test_dedupers(
    # Fixtures
    matchbox_postgres: MatchboxPostgres,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse: SourceWarehouse,
    warehouse_data: list[Source],
    # Parameterised data classes
    fx_data: DedupeTestParams,
    fx_deduper: ModelTestParams,
    # Pytest
    request: pytest.FixtureRequest,
):
    """Runs all deduper methodologies over exemplar tables.

    Tests:
        1. That the input data is as expected
        2. That the data is deduplicated correctly
        3. That the deduplicated probabilities are inserted correctly
        4. That the correct number of clusters are resolved and inserted correctly
    """
    # i. Ensure database is ready, collect fixtures, perform any special
    # deduper cleaning

    db_add_indexed_data(backend=matchbox_postgres, warehouse_data=warehouse_data)

    select: dict[Source, list[str]]
    df: DataFrame

    select, df = request.getfixturevalue(fx_data.fixture)

    fields = list(fx_data.fields.keys())

    if fx_deduper.rename_fields:
        df_renamed = df.copy().rename(columns=fx_data.fields)
        fields_renamed = list(fx_data.fields.values())
        df_renamed = df_renamed.filter(["id"] + fields_renamed)

    # 1. Input data is as expected

    if fx_deduper.rename_fields:
        assert isinstance(df_renamed, DataFrame)
        assert df_renamed.shape[0] == fx_data.curr_n
    else:
        assert isinstance(df, DataFrame)
        assert df.shape[0] == fx_data.curr_n

    # 2. Data is deduplicated correctly

    deduper_name = f"{fx_deduper.name}_{fx_data.source}"
    deduper_settings = fx_deduper.build_settings(fx_data)

    model = make_model(
        model_name=deduper_name,
        description=f"Testing dedupe of {fx_data.source} with {fx_deduper.name} method",
        model_class=fx_deduper.cls,
        model_settings=deduper_settings,
        left_data=df_renamed if fx_deduper.rename_fields else df,
        left_source=fx_data.source,
    )

    results = model.run()

    deduped_df = results.probabilities.to_df()
    deduped_df_with_source = results.probabilities.inspect_with_source(
        left_data=df, left_key="id", right_data=df, right_key="id"
    )

    assert isinstance(deduped_df, DataFrame)
    assert deduped_df.shape[0] == fx_data.tgt_prob_n

    assert isinstance(deduped_df_with_source, DataFrame)
    for field in fields:
        assert deduped_df_with_source[field + "_x"].equals(
            deduped_df_with_source[field + "_y"]
        )

    # 3. Correct number of clusters are resolved

    clusters_dupes_df = results.clusters.to_df()
    clusters_dupes_df_with_source = results.clusters.inspect_with_source(
        left_data=df, left_key="id", right_data=df, right_key="id"
    )

    assert isinstance(clusters_dupes_df, DataFrame)
    assert clusters_dupes_df.parent.nunique() == fx_data.tgt_clus_n

    assert isinstance(clusters_dupes_df_with_source, DataFrame)
    for field in fields:
        assert clusters_dupes_df_with_source[field + "_x"].equals(
            clusters_dupes_df_with_source[field + "_y"]
        )

    # 4. Probabilities and clusters are inserted correctly

    results.to_matchbox(backend=matchbox_postgres)

    model = matchbox_postgres.get_model(model=deduper_name)
    assert model.probabilities.dataframe.shape[0] == fx_data.tgt_prob_n

    model.truth = 0.0

    clusters = query(
        selector=select,
        backend=matchbox_postgres,
        return_type="pandas",
        resolution=deduper_name,
    )

    assert isinstance(clusters, DataFrame)
    assert clusters.id.nunique() == fx_data.unique_n
