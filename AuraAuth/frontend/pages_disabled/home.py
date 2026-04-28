"""Home/dashboard page."""

import streamlit as st


def page_home() -> None:
    """Show app overview and current workspace state."""
    st.title("AuraAuth AutoML")
    st.markdown(
        '<div class="hero-card"><b>Reliability-first AutoML workflow</b><br/>'
        'Upload data, train candidate models, inspect results, then export documentation.'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset loaded", "Yes" if st.session_state.get("dataset") is not None else "No")
    c2.metric("Pipeline run", "Yes" if st.session_state.get("pipeline_executed") else "No")
    c3.metric("Task", st.session_state.get("task_type") or "Not selected")

    st.info("Use the sidebar to move through each workflow stage.")
