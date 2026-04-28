"""Results display page."""

import streamlit as st


def page_results() -> None:
    """Display leaderboard and best-model summary."""
    st.title("Results")

    results = st.session_state.get("pipeline_results")
    if not results:
        st.warning("Run the pipeline first.")
        return

    st.markdown(
        f'<div class="hero-card">Best model: <b>{results["best_model"]}</b> | '
        f'{results["metric"]}: <b>{results["best_score"]:.4f}</b></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Leaderboard")
    st.dataframe(results["leaderboard"], use_container_width=True)
