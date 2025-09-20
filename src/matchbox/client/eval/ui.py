"""Interface for submitting evaluation data."""

import polars as pl
import streamlit as st
from sqlalchemy import Engine, create_engine

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.dags import DAG
from matchbox.client.eval import get_samples
from matchbox.common.eval import Judgement
from matchbox.common.exceptions import MatchboxClientSettingsException


@st.cache_resource
def init_engine() -> Engine:
    """Create a single SQLAlchemy engine."""
    engine = create_engine(settings.default_warehouse)
    return engine


def fetch_samples():
    """Download same samples to evaluate and add to session."""
    st.session_state.dag_name = st.session_state.get("dag_name")
    if not st.session_state.dag_name_input:
        raise RuntimeError("A DAG name must be set in environment variables.")

    dag = DAG(st.session_state.dag_name)

    with st.spinner("Loading samples..."):
        st.session_state.samples = get_samples(
            dag=dag,
            n=100,
            user_id=st.session_state.user_id,
            default_client=init_engine(),
        )
    st.session_state.step = "eval" if st.session_state.samples else "done"


st.title("Matchbox evaluation session")

if "step" not in st.session_state:
    if not settings.user:
        raise MatchboxClientSettingsException("User name is unset.")
    st.session_state.user_name = settings.user
    st.session_state.user_id = _handler.login(user_name=st.session_state.user_name)
    st.session_state.step = "ready"


if st.session_state.step == "ready":
    st.text_input("DAG to sample from", key="dag_name_input")
    st.button("Fetch samples", type="primary", on_click=fetch_samples)


if st.session_state.step == "eval":
    if "df" not in st.session_state:
        st.session_state.shown_cluster = next(iter(st.session_state.samples.keys()))

        original_df = st.session_state.samples.pop(st.session_state.shown_cluster)

        st.session_state.judgement = []
        st.session_state.orig_columns = list(set(original_df.columns) - {"leaf"})
        st.session_state.df = original_df.with_columns(select=False)

    st.markdown(
        f"Welcome **{st.session_state.user_name}**. "
        f"Sampling from DAG: `{st.session_state.dag_name}`"
    )
    edited_df = st.data_editor(
        st.session_state.df,
        disabled=st.session_state.orig_columns,
        column_order=("select", *st.session_state.orig_columns),
    )

    def splinter():
        """Endorse subset of cluster."""
        st.session_state.df = edited_df.filter(~pl.col("select"))
        selected_ids = (
            edited_df.filter(pl.col("select")).select("leaf").to_series().to_list()
        )
        st.session_state.judgement.append(selected_ids)
        if st.session_state.df.shape[0] == 1:
            looks_good()

    def looks_good():
        """Confirm cluster as is."""
        all_ids = st.session_state.df.select("leaf").to_series().to_list()
        st.session_state.judgement.append(all_ids)
        _handler.send_eval_judgement(
            judgement=Judgement(
                shown=st.session_state.shown_cluster,
                endorsed=st.session_state.judgement,
                user_id=st.session_state.user_id,
            ),
        )

        del st.session_state.df

        if not len(st.session_state.samples):
            st.session_state.step = "done"

    if (edited_df.select("select").to_series().any()) and (
        not edited_df.select("select").to_series().all()
    ):
        st.button("Splinter", icon="✂️", on_click=splinter)
    else:
        st.button("Looks good to me.", icon="✅", on_click=looks_good)

if st.session_state.step == "done":
    st.header("You're all done 🔥")
