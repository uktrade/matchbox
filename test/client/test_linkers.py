import pytest
from pandas import DataFrame
from splink import SettingsCreator

from matchbox import make_model, query
from matchbox.common.db import Source, SourceWarehouse
from matchbox.helpers import selectors
from matchbox.models.linkers.splinklinker import SplinkLinkerFunction, SplinkSettings
from matchbox.server.postgresql import MatchboxPostgres

from ..fixtures.db import AddDedupeModelsAndDataCallable, AddIndexedDataCallable
from ..fixtures.models import (
    LinkTestParams,
    ModelTestParams,
    dedupe_data_test_params,
    dedupe_model_test_params,
    link_data_test_params,
    link_model_test_params,
)


@pytest.mark.parametrize("fx_data", link_data_test_params)
@pytest.mark.parametrize("fx_linker", link_model_test_params)
def test_linkers(
    # Fixtures
    matchbox_postgres: MatchboxPostgres,
    db_add_dedupe_models_and_data: AddDedupeModelsAndDataCallable,
    db_add_indexed_data: AddIndexedDataCallable,
    warehouse: SourceWarehouse,
    warehouse_data: list[Source],
    # Parameterised data classes
    fx_data: LinkTestParams,
    fx_linker: ModelTestParams,
    # Pytest
    request: pytest.FixtureRequest,
):
    """Runs all linker methodologies over exemplar tables.

    Tests:
        1. That the input data is as expected
        2. That the data is linked correctly
        3. That the linked probabilities are inserted correctly
        4. That the correct number of clusters are resolved and inserted correctly
    """
    # i. Ensure database is ready, collect fixtures, perform any special linker cleaning

    db_add_dedupe_models_and_data(
        db_add_indexed_data=db_add_indexed_data,
        backend=matchbox_postgres,
        warehouse_data=warehouse_data,
        dedupe_data=dedupe_data_test_params,
        dedupe_models=[dedupe_model_test_params[0]],  # Naive deduper,
        request=request,
    )

    select_l: dict[Source, list[str]]
    select_r: dict[Source, list[str]]
    df_l: DataFrame
    df_r: DataFrame

    select_l, df_l = request.getfixturevalue(fx_data.fixture_l)
    select_r, df_r = request.getfixturevalue(fx_data.fixture_r)

    fields_l = list(fx_data.fields_l.keys())
    fields_r = list(fx_data.fields_r.keys())

    if fx_linker.rename_fields:
        df_l_renamed = df_l.copy().rename(columns=fx_data.fields_l)
        df_r_renamed = df_r.copy().rename(columns=fx_data.fields_r)
        fields_l_renamed = list(fx_data.fields_l.values())
        fields_r_renamed = list(fx_data.fields_r.values())
        df_l_renamed = df_l_renamed.filter(["id"] + fields_l_renamed)
        df_r_renamed = df_r_renamed.filter(["id"] + fields_r_renamed)
        assert set(df_l_renamed.columns) == set(df_r_renamed.columns)
        assert df_l_renamed.dtypes.equals(df_r_renamed.dtypes)

    # 1. Input data is as expected

    if fx_linker.rename_fields:
        assert isinstance(df_l_renamed, DataFrame)
        assert df_l_renamed.shape[0] == fx_data.curr_n_l
    else:
        assert isinstance(df_l, DataFrame)
        assert df_l.shape[0] == fx_data.curr_n_l

    if fx_linker.rename_fields:
        assert isinstance(df_r_renamed, DataFrame)
        assert df_r_renamed.shape[0] == fx_data.curr_n_r
    else:
        assert isinstance(df_r, DataFrame)
        assert df_r.shape[0] == fx_data.curr_n_r

    # 2. Data is linked correctly

    linker_name = f"{fx_linker.name}_{fx_data.source_l}_{fx_data.source_r}"
    linker_settings = fx_linker.build_settings(fx_data)

    model = make_model(
        model_name=linker_name,
        description=(
            f"Testing link of {fx_data.source_l} and {fx_data.source_r} "
            f"with {fx_linker.name} method."
        ),
        model_class=fx_linker.cls,
        model_settings=linker_settings,
        left_data=df_l_renamed if fx_linker.rename_fields else df_l,
        left_source=fx_data.source_l,
        right_data=df_r_renamed if fx_linker.rename_fields else df_r,
        right_source=fx_data.source_r,
    )

    results = model.run()

    linked_df = results.probabilities.to_df()
    linked_df_with_source = results.probabilities.inspect_with_source(
        left_data=df_l,
        left_key="id",
        right_data=df_r,
        right_key="id",
    )

    assert isinstance(linked_df, DataFrame)
    assert linked_df.shape[0] == fx_data.tgt_prob_n

    assert isinstance(linked_df_with_source, DataFrame)
    for field_l, field_r in zip(fields_l, fields_r, strict=True):
        assert linked_df_with_source[field_l].equals(linked_df_with_source[field_r])

    # 3. Correct number of clusters are resolved

    clusters_links_df = results.clusters.to_df()
    clusters_links_df_with_source = results.clusters.inspect_with_source(
        left_data=df_l,
        left_key="id",
        right_data=df_r,
        right_key="id",
    )

    assert isinstance(clusters_links_df, DataFrame)
    assert clusters_links_df.parent.nunique() == fx_data.tgt_clus_n

    assert isinstance(clusters_links_df_with_source, DataFrame)
    for field_l, field_r in zip(fields_l, fields_r, strict=True):
        # When we enrich the ClusterResults in a deduplication job, every child
        # id will match something in the source data, because we're only using
        # one dataset. NaNs are therefore impossible.
        # When we enrich the ClusterResults in a link job, some child ids
        # will match something in the left data, and others in the right data.
        # NaNs are therefore guaranteed.
        # We therefore coalesce by parent to unique joined values, which
        # we can expect to equal the target cluster number, and have matching
        # rows of data
        def unique_non_null(s):
            return s.dropna().unique()

        cluster_vals = (
            clusters_links_df_with_source.filter(["parent", field_l, field_r])
            .groupby("parent")
            .agg(
                {
                    field_l: unique_non_null,
                    field_r: unique_non_null,
                }
            )
            .explode(column=[field_l, field_r])
            .reset_index()
        )

        assert cluster_vals[field_l].equals(cluster_vals[field_r])
        assert cluster_vals.parent.nunique() == fx_data.tgt_clus_n
        assert cluster_vals.shape[0] == fx_data.tgt_clus_n

    # 4. Probabilities and clusters are inserted correctly

    results.to_matchbox(backend=matchbox_postgres)

    model = matchbox_postgres.get_model(model=linker_name)
    assert model.probabilities.dataframe.shape[0] == fx_data.tgt_prob_n

    model.truth = 0.0

    l_r_selector = selectors(select_l, select_r)

    clusters = query(
        selector=l_r_selector,
        backend=matchbox_postgres,
        return_type="pandas",
        resolution=linker_name,
    )

    assert isinstance(clusters, DataFrame)
    assert clusters.id.nunique() == fx_data.unique_n


def test_splink_training_functions():
    # You can create a valid SplinkLinkerFunction
    SplinkLinkerFunction(
        function="estimate_u_using_random_sampling",
        arguments={"max_pairs": 1e4},
    )
    # You can't reference a function that doesn't exist
    with pytest.raises(ValueError):
        SplinkLinkerFunction(function="made_up_funcname", arguments=dict())
    # You can't pass arguments that don't exist
    with pytest.raises(ValueError):
        SplinkLinkerFunction(
            function="estimate_u_using_random_sampling", arguments={"foo": "bar"}
        )


def test_splink_settings():
    valid_settings = SplinkSettings(
        left_id="id",
        right_id="id",
        linker_training_functions=[],
        linker_settings=SettingsCreator(link_type="link_only"),
        threshold=None,
    )
    assert valid_settings.linker_settings.unique_id_column_name == "id"
    # Can only use "link_only"
    with pytest.raises(ValueError):
        valid_settings = SplinkSettings(
            left_id="id",
            right_id="id",
            linker_training_functions=[],
            linker_settings=SettingsCreator(link_type="dedupe_only"),
            threshold=None,
        )
    # Left and right ID must coincide
    with pytest.raises(ValueError):
        valid_settings = SplinkSettings(
            left_id="id",
            right_id="id2",
            linker_training_functions=[],
            linker_settings=SettingsCreator(link_type="link_only"),
            threshold=None,
        )
