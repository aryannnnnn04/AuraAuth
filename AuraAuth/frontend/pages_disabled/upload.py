"""Dataset upload page."""

import pandas as pd
import streamlit as st


def page_upload() -> None:
    """Upload CSV dataset and choose target/task settings."""
    st.title("Upload Dataset")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.dataset = df
            st.success(f"Loaded dataset with shape {df.shape}.")
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            return

    df = st.session_state.get("dataset")
    if df is None:
        st.warning("Upload a CSV file to continue.")
        return

    st.dataframe(df.head(20), use_container_width=True)

    target = st.selectbox(
        "Target column",
        options=list(df.columns),
        index=list(df.columns).index(st.session_state.target_column)
        if st.session_state.get("target_column") in df.columns
        else len(df.columns) - 1,
    )
    task_type = st.selectbox(
        "Task type",
        options=["classification", "regression"],
        index=0 if st.session_state.get("task_type") != "regression" else 1,
    )

    st.session_state.target_column = target
    st.session_state.task_type = task_type

    if st.button("Mark dataset ready", use_container_width=True):
        st.session_state.pipeline_ready = True
        st.success("Dataset and settings saved. Proceed to Run Pipeline.")
