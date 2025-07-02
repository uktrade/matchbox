"""Interface for submitting evaluation data."""

import polars as pl
import streamlit as st

from matchbox.client import _handler
from matchbox.common.eval import Judgement

st.title("Matchbox evaluation session")


if "step" not in st.session_state:
    st.session_state.step = "eval"
    st.session_state.user_name = "scott.mcgregor@example.com"
    st.session_state.user_id = _handler.login(user_name=st.session_state.user_name)
    st.session_state.resolution = "__DEFAULT__"

if st.session_state.step == "eval":
    if "df" not in st.session_state:
        original_df = pl.from_arrow(
            _handler.sample_one(
                user_id=st.session_state.user_id,
                resolution=st.session_state.resolution,
            )
        )
        st.session_state.judgement = []
        st.session_state.orig_columns = list(set(original_df.columns) - {"id"})
        st.session_state.df = original_df.with_columns(select=False)

    st.html(
        f"Welcome <b>{st.session_state.user_name}</b>. "
        f"Sampling from resolution: <b>{st.session_state.resolution}.</b>"
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
            edited_df.filter(pl.col("select")).select("id").to_series().to_list()
        )
        st.session_state.judgement.append(selected_ids)
        if st.session_state.df.shape[0] == 1:
            looks_good()

    def looks_good():
        """Confirm cluster as is."""
        st.session_state.step = "done"
        all_ids = st.session_state.df.select("id").to_series().to_list()
        st.session_state.judgement.append(all_ids)
        _handler.send_eval_judgement(
            judgement=Judgement(
                clusters=st.session_state.judgement, user_id=st.session_state.user_id
            ),
        )

    if (edited_df.select("select").to_series().any()) and (
        not edited_df.select("select").to_series().all()
    ):
        st.button("Splinter", icon="✂️", on_click=splinter)
    else:
        st.button("Looks good to me.", icon="✅", on_click=looks_good)

else:
    st.header("You're all done 🔥")
