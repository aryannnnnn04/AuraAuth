"""Documentation export page."""

from datetime import datetime

import streamlit as st


def page_documentation() -> None:
    """Generate and download a simple pipeline report."""
    st.title("Documentation")

    results = st.session_state.get("pipeline_results")
    if not results:
        st.warning("Run the pipeline first to generate documentation.")
        return

    report = f"""# AuraAuth AutoML Report\n
Generated: {datetime.utcnow().isoformat()}Z\n
## User\n- Username: {st.session_state.get('username', 'unknown')}\n
## Dataset\n- Target column: {st.session_state.get('target_column')}\n- Task type: {st.session_state.get('task_type')}\n
## Results\n- Best model: {results.get('best_model')}\n- Metric: {results.get('metric')}\n- Score: {results.get('best_score'):.4f}\n
## Leaderboard\n{results.get('leaderboard').to_markdown(index=False)}\n"""

    st.markdown("Preview")
    st.code(report, language="markdown")

    st.download_button(
        label="Download report (.md)",
        data=report,
        file_name="auraauth_automl_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
